"""Torch-датасет для Payload-эксперта: текст запроса → индексы символов + метка."""

import polars as pl
import torch
from torch.utils.data import Dataset

from mlsiem.models.dataset import LabelEncoder
from mlsiem.models.payload.model import CharTokenizer


class PayloadDataset(Dataset):
    def __init__(self, df: pl.DataFrame, tokenizer: CharTokenizer, encoder: LabelEncoder,
                 *, text_col: str = "text", label_col: str = "label"):
        self.texts = df[text_col].to_list()
        self.labels = [encoder.encode(x) for x in df[label_col]]
        self.tokenizer = tokenizer

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, i: int):
        idx = self.tokenizer.encode(self.texts[i])
        return torch.from_numpy(idx), torch.tensor(self.labels[i], dtype=torch.long)
