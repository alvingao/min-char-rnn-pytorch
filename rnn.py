"""
Vanilla RNN implementation of min-char-rnn in PyTorch
"""

import torch
from torch import nn, optim
from torch.distributions import Categorical
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from config import FILENAME, CHECKPOINT_DIR, LOG_DIR
from dataset import TextDataset


class RNN(nn.Module):
    """
    Vanilla RNN implementation of min-char-rnn in PyTorch
    """

    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size

        self.i2h = nn.Linear(input_size, hidden_size)
        self.h2h = nn.Linear(hidden_size, hidden_size)
        self.h2o = nn.Linear(hidden_size, output_size)
        self.softmax = nn.LogSoftmax(dim=1)

    # pylint: disable=W0622
    def forward(self, input, hidden):
        """
        Forward pass of the RNN
        """
        hidden = F.tanh(self.i2h(input) + self.h2h(hidden))
        output = self.h2o(hidden)
        output = self.softmax(output)
        return output, hidden.detach()

    def init_hidden(self, batch_size: int = 1):
        """
        Returns a new hidden state of the RNN for given batch size
        """
        return torch.zeros(batch_size, self.hidden_size)


def generate_text(rnn: RNN, dataset: TextDataset, seed: torch.Tensor, n: int) -> str:
    """
    Generate text output using the model.
    """
    with torch.no_grad():
        # When generating text sequence, batch size is always 1
        hidden = rnn.init_hidden(1)

        indices = []
        idx = seed

        for _ in range(n):
            # pylint: disable=W0622,E1102
            input = F.one_hot(idx, num_classes=dataset.vocab_size).float()

            output, hidden = rnn(input, hidden)

            # Construct categorical distribution and sample a (random) character
            dist = Categorical(logits=output)
            next_idx = dist.sample().item()

            indices.append(next_idx)
            idx = torch.tensor(next_idx)

        text = dataset.vector_to_string(indices)
        return text


def save_model_weights(rnn: RNN, path: str = "rnn.pth"):
    """Save model weights to disk."""
    torch.save(rnn.state_dict(), path)


def load_model_weights(rnn: RNN, path: str = "rnn.pth"):
    """Load previously saved model weights."""
    state_dict = torch.load(path)
    rnn.load_state_dict(state_dict)


def train(
    dataloader: DataLoader,
    model: RNN,
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
            # Make the first dimension as the sequence length and second dimension as batch size
            batch_inputs = batch_inputs.transpose(0, 1).float()
            batch_targets = batch_targets.transpose(0, 1).long()

            seq_length = batch_inputs.size(0)
            batch_size = batch_inputs.size(1)

            hidden = model.init_hidden(batch_size)

            optimizer.zero_grad()
            loss = 0

            for idx in range(seq_length):
                output, hidden = model(batch_inputs[idx], hidden)
                loss += loss_fn(output, batch_targets[idx])

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
        save_model_weights(model, f"{CHECKPOINT_DIR}/rnn_epoch_{epoch}.pth")


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

    dataset = TextDataset(FILENAME)
    dataloader = DataLoader(
        dataset, batch_size=64, shuffle=True, generator=torch.Generator(device=device)
    )

    rnn = RNN(dataset.vocab_size, 128, dataset.vocab_size)

    loss_fn = nn.NLLLoss()
    optimizer = optim.Adam(rnn.parameters(), lr=1e-3)

    writer = SummaryWriter(LOG_DIR)

    train(dataloader, rnn, loss_fn, optimizer, writer)

    save_model_weights(rnn, f"{CHECKPOINT_DIR}/rnn_final.pth")


def test():
    """
    Main function to evaulate the trained model.
    """
    setup_device()

    dataset = TextDataset(FILENAME)
    rnn = RNN(dataset.vocab_size, 128, dataset.vocab_size)
    load_model_weights(rnn, f"{CHECKPOINT_DIR}/rnn_final.pth")

    seed = torch.tensor(dataset.char_to_idx["A"])
    text = generate_text(rnn, dataset, seed, 1000)
    print(text)


if __name__ == "__main__":
    main()
    # test()
