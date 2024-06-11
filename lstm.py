from typing import List

import torch
from torch import nn, optim, Tensor
from torch.distributions import Categorical
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from config import FILENAME, CHECKPOINT_DIR, LOG_DIR
from dataset import TextDataset


class LSTMCell(nn.Module):
    def __init__(self, input_size, hidden_size):
        super().__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size

        self.f = nn.Linear(input_size + hidden_size, hidden_size)
        self.i = nn.Linear(input_size + hidden_size, hidden_size)
        self.c = nn.Linear(input_size + hidden_size, hidden_size)
        self.o = nn.Linear(input_size + hidden_size, hidden_size)

    def forward(self, x: Tensor, c: Tensor, h: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        xh = torch.cat([x, h], dim=1)
        c = c * F.sigmoid(self.f(xh)) + F.sigmoid(self.i(xh)) * F.tanh(self.c(xh))
        h = F.sigmoid(self.o(xh)) * F.tanh(c)
        return h, c.detach(), h.detach()

    def init_hidden(self, batch_size: int = 1):
        """
        Returns a new hidden state of the RNN for given batch size
        """
        return torch.zeros(batch_size, self.hidden_size)

    init_cell = init_hidden


class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers: int = 1):
        super().__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size

        cells = [LSTMCell(hidden_size, hidden_size) for _ in range(1, num_layers)]
        cells.insert(0, LSTMCell(input_size, hidden_size))
        self.cells = nn.ModuleList(cells)

        self.h2o = nn.Linear(hidden_size, output_size)

    def forward(
        self, x: Tensor, state: tuple[List[Tensor], List[Tensor]]
    ) -> tuple[Tensor, tuple[List[Tensor], List[Tensor]]]:
        if state is None:
            batch_size = x.size(0)
            c = [cell.init_cell(batch_size) for cell in self.cells]
            h = [cell.init_hidden(batch_size) for cell in self.cells]
        else:
            (c, h) = state

        for idx, cell in enumerate(self.cells):
            x, c[idx], h[idx] = cell(x, c[idx], h[idx])

        y = self.h2o(x)
        y = F.log_softmax(y, dim=-1)

        return y, (c, h)


def generate_text(lstm: LSTM, dataset: TextDataset, seed: torch.Tensor, n: int) -> str:
    """
    Generate text output using the model.
    """
    with torch.no_grad():
        # When generating text sequence, batch size is always 1
        state = None

        indices = []
        idx = seed

        for _ in range(n):
            # pylint: disable=W0622,E1102
            input = F.one_hot(idx, num_classes=dataset.vocab_size).float()
            input = input.unsqueeze(0)

            output, state = lstm(input, state)

            # Construct categorical distribution and sample a (random) character
            dist = Categorical(logits=output)
            next_idx = dist.sample().item()

            indices.append(next_idx)
            idx = torch.tensor(next_idx)

        text = dataset.vector_to_string(indices)
        return text


def save_model_weights(model: nn.Module, path: str):
    """Save model weights to disk."""
    torch.save(model.state_dict(), path)


def load_model_weights(model: nn.Module, path: str):
    """Load previously saved model weights."""
    state_dict = torch.load(path)
    model.load_state_dict(state_dict)


def train(
    dataloader: DataLoader,
    model: LSTM,
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

            state = None

            optimizer.zero_grad()
            loss = 0

            for idx in range(seq_length):
                output, state = model(batch_inputs[idx], state)
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

        print(f"Epoch: {0} finished. Saving model weights.")
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
        dataset, batch_size=128, shuffle=True, generator=torch.Generator(device=device)
    )

    lstm = LSTM(dataset.vocab_size, 128, dataset.vocab_size, num_layers=2)
    nn.LSTM

    loss_fn = nn.NLLLoss()
    optimizer = optim.Adam(lstm.parameters(), lr=2e-3)

    writer = SummaryWriter(LOG_DIR)

    train(dataloader, lstm, loss_fn, optimizer, writer, num_epochs=50)

    save_model_weights(lstm, f"{CHECKPOINT_DIR}/lstm_final.pth")


def test():
    """
    Main function to evaulate the trained model.
    """
    setup_device()

    dataset = TextDataset(FILENAME, seq_length=50)
    rnn = LSTM(dataset.vocab_size, 128, dataset.vocab_size, num_layers=2)
    load_model_weights(rnn, f"{CHECKPOINT_DIR}/lstm_final.pth")

    seed = torch.tensor(dataset.char_to_idx["f"])
    text = generate_text(rnn, dataset, seed, 1000)
    print(text)


if __name__ == "__main__":
    main()
    # test()
