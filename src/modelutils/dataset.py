"""
Dataset classes, vocabulary builder, and data-loader helpers for OCR training.
"""

from __future__ import annotations

import os
import torch
import pandas as pd
from collections import Counter
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as T


# ── Shared image helper ────────────────────────────────────────────────────────

def resize_keep_ratio(img: Image.Image, height: int = 64) -> Image.Image:
    """Resize a PIL image to a fixed height, preserving the aspect ratio."""
    w, h = img.size
    new_w = max(1, int(w * (height / h)))
    return img.resize((new_w, height), Image.BILINEAR)


# ── Vocabulary ─────────────────────────────────────────────────────────────────

def build_vocab(dataset_dir: str) -> tuple[list, dict, dict]:
    """
    Build character vocabulary from all labels.csv files in *dataset_dir*.

    Returns
    -------
    vocab    : list of characters (index 0 = '<blank>' for CTC)
    char2idx : character → index
    idx2char : index → character
    """
    chars = Counter()
    for book in os.listdir(dataset_dir):
        label_path = os.path.join(dataset_dir, book, "labels.csv")
        if not os.path.exists(label_path):
            continue
        df = pd.read_csv(label_path)
        for label in df["text"]:
            chars.update(list(str(label)))

    vocab    = ["<blank>"] + sorted(chars.keys())
    char2idx = {c: i for i, c in enumerate(vocab)}
    idx2char = {i: c for c, i in char2idx.items()}
    return vocab, char2idx, idx2char


# ── CRNN line dataset ──────────────────────────────────────────────────────────

class LineDataset(Dataset):
    """
    Dataset of pre-cropped text-line images for CRNN training.

    Parameters
    ----------
    dataset_dir : root directory containing book sub-folders with labels.csv
    char2idx    : character → index mapping from build_vocab()
    img_height  : target image height (pixels); width is scaled proportionally
    augment     : apply mild augmentation (brightness, blur, affine, perspective)
    """

    _AUG = T.Compose([
        T.RandomApply([T.ColorJitter(brightness=0.35, contrast=0.35)], p=0.5),
        T.RandomApply([T.GaussianBlur(kernel_size=3, sigma=(0.1, 1.2))],  p=0.25),
        T.RandomApply([
            T.RandomAffine(
                degrees=1.5, translate=(0.01, 0.03),
                scale=(0.93, 1.07), shear=1.5, fill=255,
            )
        ], p=0.5),
        T.RandomApply(
            [T.RandomPerspective(distortion_scale=0.05, p=1.0, fill=255)], p=0.2
        ),
    ])

    def __init__(
        self,
        dataset_dir: str,
        char2idx:    dict,
        img_height:  int  = 64,
        augment:     bool = False,
    ):
        self.samples    = []
        self.char2idx   = char2idx
        self.img_height = img_height
        self.augment    = augment
        self.to_tensor  = T.ToTensor()

        for book in os.listdir(dataset_dir):
            label_path = os.path.join(dataset_dir, book, "labels.csv")
            if not os.path.exists(label_path):
                continue
            df = pd.read_csv(label_path)
            for _, row in df.iterrows():
                img_path = os.path.join(dataset_dir, row["image"])
                if os.path.exists(img_path):
                    self.samples.append((img_path, str(row["text"])))

        print(f"LineDataset: {len(self.samples)} samples  (augment={augment})")

    def encode(self, text: str) -> torch.Tensor:
        return torch.tensor([self.char2idx[c] for c in text], dtype=torch.long)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        img_path, label = self.samples[idx]
        img = Image.open(img_path).convert("L")
        img = resize_keep_ratio(img, self.img_height)
        if self.augment:
            img = self._AUG(img)
        img = self.to_tensor(img)   # (1, H, W) in [0, 1]
        return img, self.encode(label), label


def collate_fn(batch):
    """Pad images to the same width and concatenate CTC labels."""
    images, labels_encoded, labels_str = zip(*batch)

    max_w      = max(img.shape[-1] for img in images)
    padded     = [torch.nn.functional.pad(img, (0, max_w - img.shape[-1])) for img in images]
    padded_imgs = torch.stack(padded)

    label_lengths = torch.tensor([len(l) for l in labels_encoded])
    labels_concat = torch.cat(labels_encoded)

    return padded_imgs, labels_concat, label_lengths, labels_str


# ── TrOCR line dataset ─────────────────────────────────────────────────────────

class TrOCRLineDataset(Dataset):
    """
    Dataset of text-line images for TrOCR (VisionEncoderDecoder) training.

    Parameters
    ----------
    dataset_dir    : same folder layout as LineDataset
    processor      : TrOCRProcessor instance
    max_target_len : maximum tokenised label length
    """

    def __init__(self, dataset_dir: str, processor, max_target_len: int = 128):
        self.samples        = []
        self.processor      = processor
        self.max_target_len = max_target_len

        for book in os.listdir(dataset_dir):
            label_path = os.path.join(dataset_dir, book, "labels.csv")
            if not os.path.exists(label_path):
                continue
            df = pd.read_csv(label_path)
            for _, row in df.iterrows():
                img_path = os.path.join(dataset_dir, row["image"])
                if os.path.exists(img_path):
                    self.samples.append((img_path, str(row["text"])))

        print(f"TrOCRLineDataset: {len(self.samples)} samples")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        img_path, text = self.samples[idx]
        img = Image.open(img_path).convert("RGB")

        pixel_values = self.processor(img, return_tensors="pt").pixel_values.squeeze(0)

        labels = self.processor.tokenizer(
            text,
            padding    = "max_length",
            max_length = self.max_target_len,
            truncation = True,
            return_tensors = "pt",
        ).input_ids.squeeze(0)

        # Replace pad ids with -100 so they are ignored in cross-entropy loss
        labels[labels == self.processor.tokenizer.pad_token_id] = -100
        return {"pixel_values": pixel_values, "labels": labels, "text": text}


def trocr_collate(batch):
    """Collate function for TrOCRLineDataset."""
    pixel_values = torch.stack([b["pixel_values"] for b in batch])
    labels       = torch.stack([b["labels"]       for b in batch])
    texts        = [b["text"] for b in batch]
    return {"pixel_values": pixel_values, "labels": labels, "texts": texts}
