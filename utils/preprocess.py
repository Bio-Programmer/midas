# utils/preprocessing.py

import os
import pandas as pd
from sklearn.model_selection import train_test_split
from torchvision import transforms
import numpy as np

def get_effnet_image_transform():
    """Returns the standard image transform for EfficientNet-B3."""
    img_transform = transforms.Compose([
        transforms.Resize((300, 300)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],  # ImageNet mean
                             [0.229, 0.224, 0.225])  # ImageNet std
    ])
    return img_transform

def get_swin_image_transform():
    """Returns the standard image transform for Swin """
    img_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])
    return img_transform



def load_and_split(labels_path, meta_path, n_samples=None, test_size=0.2, random_state=42):
    ''' 
    Load image labels and metadata to create train and val df 
    Used for ISICDataset constructor: (image_tensor, metadata_tensor, label)
    '''
    labels_df = pd.read_csv(labels_path)
    meta_df = pd.read_csv(meta_path)
    df = pd.merge(meta_df, labels_df, on="image")

    # clean metadata
    df = df.dropna(subset=["age_approx", "sex", "anatom_site_general"])
    df = pd.get_dummies(df, columns=["sex", "anatom_site_general"])
    df["age_approx"] = (df["age_approx"] - df["age_approx"].mean()) / df["age_approx"].std()

    # convert one-hot diagnosis to class index
    label_cols = ['MEL', 'NV', 'BCC', 'AK', 'BKL', 'DF', 'VASC', 'SCC']
    df["label"] = df[label_cols].values.argmax(axis=1)
    df = df.drop(columns=label_cols)
    index_to_label = {idx: name for idx, name in enumerate(label_cols)}

    # handle lesion_id — treat missing values as unique lesions
    if "lesion_id" in df.columns:
        df["lesion_id"] = df["lesion_id"].fillna(df["image"])
    else:
        df["lesion_id"] = df["image"]  # fallback: 1 lesion per image

    # optional downsampling
    if n_samples is not None:
        df = df.sample(n=n_samples, random_state=random_state)

    # group-aware split by lesion_id
    lesion_ids = df["lesion_id"].unique().tolist()
    rng = np.random.default_rng(random_state)
    rng.shuffle(lesion_ids)

    split_idx = int(len(lesion_ids) * (1 - test_size))
    train_lesions = set(lesion_ids[:split_idx])
    val_lesions = set(lesion_ids[split_idx:])

    train_df = df[df["lesion_id"].isin(train_lesions)].reset_index(drop=True)
    val_df = df[df["lesion_id"].isin(val_lesions)].reset_index(drop=True)

    # Logging: lesion and image counts
    print(f"Train split: {train_df.shape[0]} images, {len(train_lesions)} unique lesions")
    print(f"Val split:   {val_df.shape[0]} images, {len(val_lesions)} unique lesions")

    # drop lesion_id so it's not treated as a feature
    train_df = train_df.drop(columns=["lesion_id"])
    val_df = val_df.drop(columns=["lesion_id"])

    return train_df, val_df, index_to_label


def load_full_isic(labels_path, meta_path, n_samples=None):
    ''' 
    Load and preprocess full ISIC training data without splitting.
    Returns full df and label index mapping.
    '''
    labels_df = pd.read_csv(labels_path)
    meta_df = pd.read_csv(meta_path)
    df = pd.merge(meta_df, labels_df, on="image")

    # clean metadata
    df = df.dropna(subset=["age_approx", "sex", "anatom_site_general"])
    df = pd.get_dummies(df, columns=["sex", "anatom_site_general"])
    df["age_approx"] = (df["age_approx"] - df["age_approx"].mean()) / df["age_approx"].std()

    # convert one-hot diagnosis to class index
    label_cols = ['MEL', 'NV', 'BCC', 'AK', 'BKL', 'DF', 'VASC', 'SCC']
    df["label"] = df[label_cols].values.argmax(axis=1)
    df = df.drop(columns=label_cols)

    # optional downsampling
    if n_samples is not None:
        df = df.sample(n=n_samples, random_state=42)

    print(f"Loaded {df.shape[0]} images from full ISIC data set.")
    index_to_label = {idx: name for idx, name in enumerate(label_cols)}

    return df.reset_index(drop=True), index_to_label

