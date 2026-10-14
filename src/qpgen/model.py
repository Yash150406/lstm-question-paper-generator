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
        # One single-layer LSTM per layer with our own dropout between them. nn.LSTM(dropout=...)
        # does the same maths, but its cuDNN dropout state crashes the process on exit after GPU
        # training with torch 2.14.1+cu130 on Windows (issue #3).
        self.lstms = nn.ModuleList(nn.LSTM(embed_dim if i == 0 else hidden_dim, hidden_dim, batch_first=True)
                                   for i in range(num_layers))
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
        for name, param in self.lstms.named_parameters():
            if "bias" in name:
                nn.init.zeros_(param)
                if "bias_ih" in name:
                    # PyTorch gate order is input, forget, cell, output.
                    param.data[hidden_dim:2 * hidden_dim] = forget_bias

    def forward(self, x: torch.Tensor, hidden: tuple | None = None) -> tuple[torch.Tensor, tuple]:
        """x is (batch, seq_len) token ids. Returns logits (batch, seq_len, vocab) and hidden,
        where hidden is (h, c), each (num_layers, batch, hidden_dim)."""
        out = self.drop(self.embedding(x))
        hs, cs = [], []
        for i, lstm in enumerate(self.lstms):
            state = None if hidden is None else (hidden[0][i:i + 1], hidden[1][i:i + 1])
            out, (h, c) = lstm(out, state)
            hs.append(h)
            cs.append(c)
            if i < len(self.lstms) - 1:
                out = self.drop(out)      # dropout between layers
        return self.decoder(self.drop(out)), (torch.cat(hs), torch.cat(cs))
