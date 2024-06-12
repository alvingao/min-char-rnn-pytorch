import os

from typing import List

import torch
from torch import nn, optim, Tensor
from torch.distributions import Categorical
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from config import FILENAME, CHECKPOINT_DIR, LOG_DIR
from dataset import TextDataset


class Network(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers):
        super().__init__()

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
        )

        self.out = nn.Linear(hidden_size, output_size)

    def forward(self, x, state):
        lstm_out, state = self.lstm(x, state)
        output = self.out(lstm_out)
        output = F.log_softmax(output, dim=-1)
        return output, state


def generate_text(
    model: Network, dataset: TextDataset, seed: torch.Tensor, n: int
) -> str:
    """
    Generate text output using the model.
    """
    with torch.no_grad():
        indices = []
        idx = seed

        state = None

        for _ in range(n):
            # pylint: disable=W0622,E1102
            x = F.one_hot(idx, num_classes=dataset.vocab_size).float()
            x = x.unsqueeze(0)
            output, state = model(x, state)

            # Construct categorical distribution and sample a (random) character
            dist = Categorical(logits=output)
            next_idx = dist.sample().item()

            indices.append(next_idx)
            idx = torch.tensor(next_idx)

        text = dataset.vector_to_string(indices)
        return text


def save_model_weights(model: nn.Module, path: str):
    """Save model weights to disk."""
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    torch.save(model.state_dict(), path)


def load_model_weights(model: nn.Module, path: str):
    """Load previously saved model weights."""
    state_dict = torch.load(path)
    model.load_state_dict(state_dict)


def train(
    dataloader: DataLoader,
    model: Network,
    loss_fn,
    optimizer: optim.Optimizer,
    writer: SummaryWriter,
    num_epochs: int = 5,
):
    """
    Train the network
    """
    for epoch in range(num_epochs):
        for batch_idx, (batch_inputs, batch_targets) in enumerate(dataloader):
            optimizer.zero_grad()
            loss = 0

            output, _ = model(batch_inputs, None)
            loss = loss_fn(output.transpose(1, 2), batch_targets)
            loss.backward()

            # Fix the gradient exploding problem with RNNs
            nn.utils.clip_grad_norm_(model.parameters(), 5)

            optimizer.step()

            if batch_idx % 100 == 0:
                print(f"Epoch {epoch}, Iteration {batch_idx}, Loss {loss.item()}")
                writer.add_scalar(
                    "Training Loss", loss.item(), epoch * len(dataloader) + batch_idx
                )

            if batch_idx % 500 == 0:
                text = generate_text(
                    model,
                    dataloader.dataset,
                    batch_inputs[0][0].argmax(),
                    1000,
                )
                print("---------------")
                print("Sampled text:")
                print(text)
                print("---------------")

        print(f"Epoch: {epoch} finished. Saving model weights.")
        save_model_weights(model, f"{CHECKPOINT_DIR}/lstm_epoch_{epoch}.pth")


def setup_device():
    """
    Setup default device preferring GPUs.
    """
    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )
    torch.set_default_device(device)

    return device


def main():
    """
    Main function
    """
    device = setup_device()

    dataset = TextDataset(FILENAME, seq_length=50)
    dataloader = DataLoader(
        dataset, batch_size=256, shuffle=True, generator=torch.Generator(device=device)
    )

    model = Network(dataset.vocab_size, 256, dataset.vocab_size, num_layers=3)
    loss_fn = nn.NLLLoss()
    optimizer = optim.Adam(model.parameters(), lr=2e-3)
    writer = SummaryWriter(LOG_DIR)

    train(dataloader, model, loss_fn, optimizer, writer, num_epochs=50)

    save_model_weights(model, f"{CHECKPOINT_DIR}/lstm_torch_final.pth")


def test():
    """
    Main function to evaulate the trained model.
    """
    setup_device()

    dataset = TextDataset(FILENAME, seq_length=50)
    model = Network(dataset.vocab_size, 256, dataset.vocab_size, num_layers=3)
    load_model_weights(model, f"{CHECKPOINT_DIR}/lstm_torch_final.pth")

    seed = torch.tensor(dataset.char_to_idx["f"])
    text = generate_text(model, dataset, seed, 1000)
    print(text)


if __name__ == "__main__":
    main()
    # test()
