import logging
import os
import torch
import torch.nn as nn
from transformers import (
    AutoTokenizer,
    AutoModel,
    AutoModelForImageClassification,
    CLIPModel,
)
from torch.optim.lr_scheduler import StepLR
from tqdm import tqdm
import pandas as pd


class MultiModalClassifier(nn.Module):
    def __init__(self, num_classes, model_type="vit+text+mamba",
                 num_hidden_layers=0, hidden_neurons=512):
        """
        model_type: one of "mamba", "vit", "mamba+text", "vit+text", "vit+text+mamba"
        num_hidden_layers: number of hidden layers to use in the combined classifier.
        hidden_neurons: number of neurons per hidden layer.
        """
        super(MultiModalClassifier, self).__init__()
        self.model_type = model_type

        # Combined feature dimension from branches
        combined_feature_dim = 0

        # --- Text Branch ---
        if "text" in model_type:
            self.text_tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
            self.text_model = AutoModel.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
            self.text_feat_dim = self.text_model.config.hidden_size
            self.text_classifier = nn.Linear(self.text_feat_dim, num_classes)
            combined_feature_dim += self.text_feat_dim
        else:
            self.text_tokenizer = None

        # --- Mamba Branch ---
        if "mamba" in model_type:
            mamba_model_name = "nvidia/MambaVision-T-1K"
            try:
                self.mamba_model = AutoModelForImageClassification.from_pretrained(
                    mamba_model_name,
                    trust_remote_code=True,
                    num_labels=num_classes,
                    ignore_mismatched_sizes=True
                )
                if hasattr(self.mamba_model, "classifier"):
                    in_features = self.mamba_model.classifier.in_features
                    self.mamba_model.classifier = nn.Linear(in_features, num_classes)
                    logging.info("Replaced 'classifier' with new head for Mamba branch.")
                elif hasattr(self.mamba_model, "head"):
                    in_features = self.mamba_model.head.in_features
                    self.mamba_model.head = nn.Linear(in_features, num_classes)
                    logging.info("Replaced 'head' with new head for Mamba branch.")
                else:
                    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                    dummy_input = torch.randn(1, 3, 224, 224).to(device)
                    self.mamba_model.to(device)
                    with torch.no_grad():
                        output = self.mamba_model(dummy_input)
                        features = output["logits"] if isinstance(output, dict) else output.logits
                    in_features = features.shape[1]
                    if hasattr(self.mamba_model, "head"):
                        self.mamba_model.head = nn.Linear(in_features, num_classes)
                        logging.info("Replaced 'head' via fallback for Mamba branch.")
                    else:
                        self.mamba_model.classifier = nn.Linear(in_features, num_classes)
                        logging.info("Replaced 'classifier' via fallback for Mamba branch.")
                self.mamba_feat_dim = in_features

                # Freeze parameters for this branch
                for param in self.mamba_model.parameters():
                    param.requires_grad = False
            except Exception as e:
                logging.error(f"Error setting up MambaVision model: {e}")
                raise e

            combined_feature_dim += self.mamba_feat_dim
        else:
            self.mamba_model = None

        # --- ViT Branch ---
        if "vit" in model_type:
            clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
            self.vit_model_vision = clip_model.vision_model
            self.vit_feat_dim = self.vit_model_vision.config.hidden_size
            self.vit_classifier = nn.Linear(self.vit_feat_dim, num_classes)
            for param in self.vit_model_vision.parameters():
                param.requires_grad = False
            combined_feature_dim += self.vit_feat_dim
        else:
            self.vit_model_vision = None

        # --- Combined Classifier ---
        if num_hidden_layers > 0:
            layers = []
            input_dim = combined_feature_dim
            for _ in range(num_hidden_layers):
                layers.append(nn.Linear(input_dim, hidden_neurons))
                layers.append(nn.ReLU())
                input_dim = hidden_neurons
            self.combined_hidden = nn.Sequential(*layers)
            self.combined_classifier = nn.Linear(input_dim, num_classes)
        else:
            self.combined_hidden = None
            self.combined_classifier = nn.Linear(combined_feature_dim, num_classes)

    def forward(self, image, text=None):
        branch_features = {}
        outputs = {}

        # --- Text Branch ---
        if "text" in self.model_type:
            if text is None:
                raise ValueError("Text input is required for model types that use text.")
            encoded = self.text_tokenizer(
                text,
                padding=True,
                truncation=True,
                return_tensors="pt"
            ).to(image.device)
            text_outputs = self.text_model(**encoded)
            text_feat = text_outputs.last_hidden_state[:, 0, :]
            branch_features["text"] = text_feat
            outputs["text"] = self.text_classifier(text_feat)

        # --- Mamba Branch ---
        if "mamba" in self.model_type:
            mamba_out = self.mamba_model(image)
            mamba_feat = mamba_out["logits"] if isinstance(mamba_out, dict) else mamba_out.logits
            branch_features["mamba"] = mamba_feat
            outputs["mamba"] = mamba_feat

        # --- ViT Branch ---
        if "vit" in self.model_type:
            vit_out = self.vit_model_vision(image)
            vit_feat = vit_out.pooler_output
            branch_features["vit"] = vit_feat
            outputs["vit"] = self.vit_classifier(vit_feat)

        # --- Combined Output ---
        if branch_features:
            combined_features = torch.cat(list(branch_features.values()), dim=1)
            if self.combined_hidden is not None:
                hidden = self.combined_hidden(combined_features)
                outputs["combined"] = self.combined_classifier(hidden)
            else:
                outputs["combined"] = self.combined_classifier(combined_features)
        else:
            raise ValueError("No branch selected to build combined features.")

        return outputs


def compute_accuracy(outputs, labels, topk=(1,)):
    with torch.no_grad():
        maxk = max(topk)
        batch_size = labels.size(0)
        _, pred = outputs.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(labels.view(1, -1).expand_as(pred))
        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append((correct_k.mul_(100.0 / batch_size)).item())
        return res


def evaluate(model, dataloader, criterion, device, num_classes, used_branches):
    model.eval()
    losses = []
    total_top1 = total_top3 = total_samples = 0

    per_class = {i: {"top1_correct": 0, "top3_correct": 0, "total": 0} for i in range(num_classes)}

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating", leave=False):
            images = batch["image"].to(device)
            texts = batch.get("text", None)
            labels = batch["label"].to(device)

            outputs = model(images, texts)
            combined_logits = outputs["combined"]
            loss = criterion(combined_logits, labels)
            losses.append(loss.item())

            top1_acc, top3_acc = compute_accuracy(combined_logits, labels, topk=(1, 3))
            total_top1 += top1_acc * images.size(0) / 100.0
            total_top3 += top3_acc * images.size(0) / 100.0
            total_samples += images.size(0)

            top3_preds = combined_logits.topk(3, dim=1, largest=True, sorted=True)[1]
            for i in range(labels.size(0)):
                true_label = labels[i].item()
                per_class[true_label]["total"] += 1
                if top3_preds[i, 0].item() == true_label:
                    per_class[true_label]["top1_correct"] += 1
                if true_label in top3_preds[i].tolist():
                    per_class[true_label]["top3_correct"] += 1

    avg_loss = sum(losses) / len(losses)
    overall_top1 = (total_top1 / total_samples) * 100.0
    overall_top3 = (total_top3 / total_samples) * 100.0

    # Convert per-class counts to percentages
    per_class_acc = {}
    for cls, stats in per_class.items():
        total = stats["total"]
        if total > 0:
            top1_percent = 100 * stats["top1_correct"] / total
            top3_percent = 100 * stats["top3_correct"] / total
        else:
            top1_percent = 0
            top3_percent = 0
        per_class_acc[cls] = {"top1_acc": top1_percent, "top3_acc": top3_percent}
    return avg_loss, overall_top1, overall_top3, per_class_acc


def train_model(model, train_loader, val_loader, criterion, optimizer, scheduler, device,
                num_epochs, early_stop, num_classes, class_map_df, used_branches, args=None):
    """
    Trains the model and logs final statistics.
    If an argparse 'args' object is provided, its parameters will be included in the filename.
    """
    best_val_top1 = 0
    early_stop_counter = 0

    best_model_state = None
    best_epoch = None
    best_val_top3 = None
    best_val_per_class = None

    for epoch in range(1, num_epochs + 1):
        model.train()
        running_loss = 0.0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{num_epochs}", leave=False)
        for batch in pbar:
            images = batch["image"].to(device)
            texts = batch.get("text", None)
            labels = batch["label"].to(device)

            optimizer.zero_grad()
            outputs = model(images, texts)
            loss = sum(criterion(outputs[key], labels) for key in outputs)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            pbar.set_postfix(loss=f"{loss.item():.4f}")

        scheduler.step()

        val_loss, val_top1, val_top3, val_per_class = evaluate(model, val_loader, criterion, device, num_classes,
                                                               used_branches)
        avg_train_loss = running_loss / len(train_loader)
        logging.info(
            f"Epoch [{epoch}/{num_epochs}] | Train Loss: {avg_train_loss:.4f} | "
            f"Val Loss: {val_loss:.4f} | Val Top1: {val_top1:.2f}% | Val Top3: {val_top3:.2f}%"
        )

        if val_top1 > best_val_top1:
            best_val_top1 = val_top1
            best_val_top3 = val_top3
            best_model_state = model.state_dict()
            best_epoch = epoch
            best_val_per_class = val_per_class
            early_stop_counter = 0
            logging.info(f"New best found at epoch {epoch}: Val Top1: {val_top1:.2f}%")
        else:
            early_stop_counter += 1

        if early_stop_counter >= early_stop:
            logging.info(
                f"Early stopping triggered at epoch {epoch}: No improvement for {early_stop} consecutive epochs.")
            break

    train_loss, train_top1, _, _ = evaluate(model, train_loader, criterion, device, num_classes, used_branches)
    logging.info(f"Final Training Loss: {train_loss:.4f} | Final Training Top1: {train_top1:.2f}%")
    logging.info(
        f"Best Epoch: {best_epoch} | Best Val Top1: {best_val_top1:.2f}% | Best Val Top3: {best_val_top3:.2f}%")

    # Build per-class accuracy table and log it.
    table_rows = []
    for cls in sorted(best_val_per_class.keys()):
        top1_acc = best_val_per_class[cls]["top1_acc"]
        top3_acc = best_val_per_class[cls]["top3_acc"]
        if not class_map_df.empty and 'class_name' in class_map_df.columns:
            class_name = class_map_df.loc[class_map_df['class_number'] == cls, 'class_name'].values[0]
        else:
            class_name = str(cls)
        table_rows.append({
            "Class": class_name,
            "Top1 Accuracy (%)": round(top1_acc, 2),
            "Top3 Accuracy (%)": round(top3_acc, 2)
        })
    df_table = pd.DataFrame(table_rows).set_index("Class")
    logging.info("Per-Class Validation Accuracies (Best Epoch):\n%s", df_table.to_string())

    # Save the best model weight in the "weight" folder.
    if best_model_state is not None:
        if args is not None:
            param_str = f"lr{args.learning_rate}_step{args.step}_gamma{args.gamma}_layers{args.num_hidden_layers}_neurons{args.hidden_neurons}"
        else:
            param_str = "default_params"
        filename = f"{model.model_type}_{param_str}_epoch{best_epoch}.pth"
        weight_folder = "weight"
        os.makedirs(weight_folder, exist_ok=True)
        full_path = os.path.join(weight_folder, filename)
        torch.save(best_model_state, full_path)
        logging.info(f"Saved best model weight to: {full_path}")
    else:
        logging.warning("No best model state was recorded.")

    return {
        "final_train_loss": train_loss,
        "final_train_top1": train_top1,
        "best_epoch": best_epoch,
        "best_val_top1": best_val_top1,
        "best_val_top3": best_val_top3,
        "per_class": best_val_per_class,
    }


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    pass
