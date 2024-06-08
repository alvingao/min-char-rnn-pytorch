"""
Dataset shared by all models.
"""

import torch


class TextDataset(torch.utils.data.Dataset):
    """
    Dataset class to load text from a file with given sequence length.
    """

    def __init__(self, filename: str, seq_length: int = 25):
        with open(filename, "r", encoding="utf-8") as f:
            self.data = f.read()

        self.chars = list(set(self.data))
        self.vocab_size = len(self.chars)
        self.char_to_idx = {ch: i for i, ch in enumerate(self.chars)}
        self.idx_to_char = dict(enumerate(self.chars))

        self.seq_length = seq_length

    def __len__(self):
        return int((len(self.data) - 1) / self.seq_length)

    def __getitem__(self, idx):
        start = idx * self.seq_length
        end = start + self.seq_length
        x = self._get_x_tensor(self.data[start:end])
        y = self._get_y_tensor(self.data[start + 1 : end + 1])
        return x, y

    # Convert input string to a one-hot encoded tensor with shape (len(text), 1, vocab_size)
    def _get_x_tensor(self, text: str) -> torch.Tensor:
        tensor = torch.zeros(len(text), 1, self.vocab_size)
        for i, ch in enumerate(text):
            tensor[i][0][self.char_to_idx[ch]] = 1
        return tensor

    # Convert target string to a tensor with shape (len(text)) with each element
    # as the index of the character
    def _get_y_tensor(self, text: str) -> torch.Tensor:
        tensor = torch.zeros(len(text))
        for i, ch in enumerate(text):
            tensor[i] = self.char_to_idx[ch]
        return tensor
