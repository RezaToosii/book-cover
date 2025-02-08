import os
import logging
import argparse
import pandas as pd
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms

from model import MultiModalClassifier, train_model
from dataloader import BookCoverDataset

# Prevent tokenizer parallelism warnings.
os.environ["TOKENIZERS_PARALLELISM"] = "false"


def setup_logging(log_filename="training.log"):
    """Set up logging to both file and console."""
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s", "%Y-%m-%d %H:%M:%S")

    file_handler = logging.FileHandler(log_filename, mode="w")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)


def save_results(train_stats, args, class_map_df):
    """
    Save final model parameters and training results into a new log file 'result.log'.
    This file will include:
      - Parsed hyperparameters.
      - Final Training Loss & Training Top1.
      - Best Epoch, Best Val Top1, Best Val Top3.
      - Per-class accuracies as a table.
    """
    result_lines = ["\n----- Training Results -----", "", "Parsed Hyperparameters:", str(args), "",
                    f"Final Training Loss: {train_stats['final_train_loss']:.4f}",
                    f"Final Training Top1: {train_stats['final_train_top1']:.2f}%", "",
                    f"Best Epoch: {train_stats['best_epoch']}", f"Best Val Top1: {train_stats['best_val_top1']:.2f}%",
                    f"Best Val Top3: {train_stats['best_val_top3']:.2f}%", "",
                    "Per-Class Validation Accuracies (Best Epoch):"]

    # Build table
    table_rows = []
    for cls in sorted(train_stats['per_class'].keys()):
        top1_acc = train_stats['per_class'][cls]["top1_acc"]
        top3_acc = train_stats['per_class'][cls]["top3_acc"]
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
    result_lines.append(df_table.to_string())
    result_lines.append("")
    result_lines.append("----- End of Results -----")

    # Write results to 'result.log'
    with open("result.log", "a") as f:
        f.write("\n".join(result_lines) + "\n")
    logging.info("Saved final results to 'result.log'.")


def main():
    setup_logging()

    parser = argparse.ArgumentParser(description="Train a multimodal classifier.")
    parser.add_argument("--model_type", type=str, required=True)
    parser.add_argument("--learning_rate", type=float, default=2e-5)
    parser.add_argument("--using_scheduler", type=bool, default=True)
    parser.add_argument("--step", type=int, default=5)
    parser.add_argument("--gamma", type=float, default=0.1)
    parser.add_argument("--epochs", type=int, default=7)
    parser.add_argument("--early_stop", type=int, default=3)
    parser.add_argument("--num_hidden_layers", type=int, default=0)
    parser.add_argument("--hidden_neurons", type=int, default=512)
    args = parser.parse_args()

    logging.info("Parsed arguments: %s", args)

    # Hard-coded dataset parameters
    train_csv = "train_dataset.csv"
    val_csv = "test_dataset.csv"
    class_map = "class_map.csv"
    batch_size = 32

    # Determine if text is used based on the model type.
    use_text = "text" in args.model_type

    # Read class map and count number of classes.
    class_map_df = pd.read_csv(class_map, header=None, names=["class_number", "class_name"])
    num_classes = class_map_df["class_number"].nunique()

    # Image transformations
    image_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    # Create datasets and loaders.
    train_dataset = BookCoverDataset(train_csv, transform=image_transform, use_text=use_text)
    val_dataset = BookCoverDataset(val_csv, transform=image_transform, use_text=use_text)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MultiModalClassifier(num_classes=num_classes, model_type=args.model_type,
                                 num_hidden_layers=args.num_hidden_layers,
                                 hidden_neurons=args.hidden_neurons).to(device)

    logging.info("Model Architecture:\n%s", model)

    optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=args.learning_rate)
    if args.using_scheduler:
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.step, gamma=args.gamma)
        logging.info("Using learning rate scheduler.")
    else:
        scheduler = None
        logging.info("Not using a learning rate scheduler.")
    criterion = torch.nn.CrossEntropyLoss()

    used_branches = []
    if "text" in args.model_type:
        used_branches.append("text")
    if "mamba" in args.model_type:
        used_branches.append("mamba")
    if "vit" in args.model_type:
        used_branches.append("vit")
    used_branches.append("combined")

    # Train the model (pass args for filename construction)
    train_stats = train_model(model, train_loader, val_loader, criterion, optimizer, scheduler, device,
                              num_epochs=args.epochs, early_stop=args.early_stop, num_classes=num_classes,
                              class_map_df=class_map_df, used_branches=used_branches, args=args)

    logging.info("Training complete.")

    # Save final results into a new log file "result.log"
    save_results(train_stats, args, class_map_df)


if __name__ == "__main__":
    main()
