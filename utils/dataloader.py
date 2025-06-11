# utils/dataloader.py

import os
import pandas as pd
import numpy as np
from PIL import Image, UnidentifiedImageError
from torch.utils.data import Dataset

class ISICDataset(Dataset):
    """
    PyTorch Dataset for ISIC images with optional metadata.
    Returns:
        (image, label) if use_metadata=False
        (image, metadata, label) if use_metadata=True
    """

    def __init__(self, img_dir, metadata_df, transform=None, index_to_label=None, use_metadata=True):
        self.img_dir = img_dir
        self.metadata_df = metadata_df.reset_index(drop=True)
        self.transform = transform
        self.image_ids = self.metadata_df['image'].tolist()
        self.use_metadata = use_metadata
        self.index_to_label = index_to_label

        if self.use_metadata:
            self.meta_features = self.metadata_df.drop(columns=['image', 'label']).values.astype(np.float32)
        self.labels = self.metadata_df['label'].values.astype(np.float32)

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, idx):
        image_id = self.image_ids[idx]
        image_path = os.path.join(self.img_dir, f"{image_id}.jpg")
        image = Image.open(image_path).convert('RGB')

        if self.transform:
            image = self.transform(image)

        label = self.labels[idx]

        if self.use_metadata:
            meta = self.meta_features[idx]
            return image, meta, label
        else:
            return image, label


# canonical class order used everywhere in your codebase
LABEL_ORDER  = ['MEL', 'NV', 'BCC', 'AK', 'BKL', 'DF', 'VASC', 'SCC']
LABEL_TO_IDX = {lbl: i for i, lbl in enumerate(LABEL_ORDER)}
IDX_TO_LABEL = {i: lbl for i, lbl in enumerate(LABEL_ORDER)}

class MIDASDataset(Dataset):
    """
    Dataset for cleaned MIDAS CSV.
    Assumes all images exist and are valid.
    Prioritizes cropped version when available.
    """

    def __init__(self, csv_path: str, image_dir: str, transform=None):
        df = pd.read_csv(csv_path)
        df["isic_label"] = df[LABEL_ORDER].idxmax(axis=1)
        df["label_idx"]  = df["isic_label"].map(LABEL_TO_IDX).astype("int64")

        self.image_files = df["image"].tolist()
        self.labels      = df["label_idx"].tolist()
        self.image_dir   = image_dir
        self.transform   = transform

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        base = self.image_files[idx]
        name, ext = os.path.splitext(base)
        cropped = os.path.join(self.image_dir, f"{name}_cropped{ext}")
        base_path = os.path.join(self.image_dir, base)

        img_path = cropped if os.path.exists(cropped) else base_path
        image = Image.open(img_path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        label = self.labels[idx]
        # used_cropped = os.path.exists(cropped)
        return image, label


