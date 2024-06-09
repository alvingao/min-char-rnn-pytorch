"""
Dataset shared by all models.
"""

from typing import List

import torch


class TextDataset(torch.utils.data.Dataset):
    """
    Dataset class to load text from a file with given sequence length.
    """

    def __init__(self, filename: str, seq_length: int = 25):
        with open(filename, "r", encoding="utf-8") as f:
            self.text = f.read()

        self.chars = list(set(self.text))
        self.vocab_size = len(self.chars)
        self.char_to_idx = {ch: i for i, ch in enumerate(self.chars)}
        self.idx_to_char = dict(enumerate(self.chars))

        self.seq_length = seq_length

        self.text_vector = self.string_to_vector(self.text)

    def __len__(self):
        return int((len(self.text_vector) - 1) / self.seq_length)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        start = idx * self.seq_length
        end = start + self.seq_length

        x = torch.tensor(self.text_vector[start:end])
        # pylint: disable=E1102
        x = torch.nn.functional.one_hot(x, num_classes=self.vocab_size).float()

        y = torch.tensor(self.text_vector[start + 1 : end + 1]).long()

        return x, y

    def string_to_vector(self, text: str) -> List[int]:
        """
        Convert string to tensor
        """
        return [self.char_to_idx[ch] for ch in text]

    def vector_to_string(self, vector: List[int]) -> str:
        """
        Convert tensor to string
        """
        return "".join([self.idx_to_char[i] for i in vector])
