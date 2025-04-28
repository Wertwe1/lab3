from transformers import BertTokenizerFast
import torch
from torch.utils.data import Dataset

class TextDataset(Dataset):
    def __init__(self, texts, tokenizer: BertTokenizerFast, max_len=128):
        self.texts = texts
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.encodings = tokenizer(
            texts,
            padding="max_length",
            truncation=True,
            max_length=max_len,
            return_tensors="pt"
        )
        # Ensure attention_mask is [B, L]
        self.encodings['attention_mask'] = self.encodings['attention_mask'].squeeze()

    def __len__(self):
        return len(self.encodings["input_ids"])

    def __getitem__(self, idx):
        return {
            "input_ids": self.encodings["input_ids"][idx],
            "token_type_ids": self.encodings["token_type_ids"][idx],
            "attention_mask": self.encodings["attention_mask"][idx]
        }