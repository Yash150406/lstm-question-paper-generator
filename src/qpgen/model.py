"""Conditional word-level LSTM language model."""
import torch
from torch import nn


class QuestionLSTM(nn.Module):
    """Embedding, 2-layer LSTM and a linear decoder tied to the embedding."""

    def __init__(self, vocab_size: int, embed_dim: int, hidden_dim: int, num_layers: int,
                 dropout: float, tie_weights: bool, init_range: float, forget_bias: float) -> None:
        super().__init__()
        if tie_weights and embed_dim != hidden_dim:
            raise ValueError("tie_weights needs embed_dim == hidden_dim")
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.drop = nn.Dropout(dropout)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, num_layers, dropout=dropout, batch_first=True)
        self.decoder = nn.Linear(hidden_dim, vocab_size)
        if tie_weights:
            self.decoder.weight = self.embedding.weight
        self._init_weights(init_range, forget_bias, hidden_dim)

    @classmethod
    def from_config(cls, model_cfg: dict, vocab_size: int) -> "QuestionLSTM":
        """Build from the `model` section of the config."""
        return cls(vocab_size, **model_cfg)

    def _init_weights(self, init_range: float, forget_bias: float, hidden_dim: int) -> None:
        """Uniform embeddings, zero decoder bias, forget gate bias set to forget_bias."""
        nn.init.uniform_(self.embedding.weight, -init_range, init_range)
        nn.init.zeros_(self.decoder.bias)
        for name, param in self.lstm.named_parameters():
            if name.startswith("bias"):
                nn.init.zeros_(param)
                if name.startswith("bias_ih"):
                    # PyTorch gate order is input, forget, cell, output.
                    param.data[hidden_dim:2 * hidden_dim] = forget_bias

    def forward(self, x: torch.Tensor, hidden: tuple | None = None) -> tuple[torch.Tensor, tuple]:
        """x is (batch, seq_len) token ids. Returns logits (batch, seq_len, vocab) and hidden."""
        out, hidden = self.lstm(self.drop(self.embedding(x)), hidden)
        return self.decoder(self.drop(out)), hidden
