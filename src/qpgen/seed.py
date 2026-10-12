"""One place to fix every random seed."""
import random

import numpy as np
import torch


def set_seed(seed: int) -> None:
    """Seed Python, NumPy and PyTorch so runs repeat."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
