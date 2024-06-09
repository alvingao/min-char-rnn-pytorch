"""Configurations for the training/testing."""

from dataclasses import dataclass


@dataclass
class Config:
    """Wraps configuration items."""

    filename: str
    checkpoint_dir: str
    log_dir: str


SHERLOCK = Config(
    "data/sherlock.txt", "model_weights/rnn_sherlock", "runs/min-char-rnn-sherlock"
)
SHAKESPEAR = Config(
    "data/shakespeare.txt",
    "model_weights/rnn_shakespeare",
    "runs/min-char-rnn-shakespeare",
)
DJANGO = Config(
    "data/django.txt", "model_weights/rnn_django", "runs/min-char-rnn-django"
)

CONFIG = SHERLOCK

FILENAME = CONFIG.filename
CHECKPOINT_DIR = CONFIG.checkpoint_dir
LOG_DIR = CONFIG.log_dir
