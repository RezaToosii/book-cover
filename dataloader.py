import pandas as pd
from torch.utils.data import Dataset
from PIL import Image


class BookCoverDataset(Dataset):
    """
    Dataset for book covers.
    CSV expected columns: image_path, image_name, class_number, class_name, ocr_text
    """

    def __init__(self, csv_file, transform=None, use_text=True):
        self.df = pd.read_csv(csv_file)
        self.transform = transform
        self.use_text = use_text
        self.df["class_number"] = self.df["class_number"].astype(int)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        image = Image.open(row["image_path"]).convert("RGB")
        if self.transform:
            image = self.transform(image)
        if self.use_text:
            text = row["ocr_text"]
        else:
            text = ""  # or you can omit the key altogether
        label = int(row["class_number"])
        sample = {"image": image, "label": label}
        if self.use_text:
            sample["text"] = text
        return sample
