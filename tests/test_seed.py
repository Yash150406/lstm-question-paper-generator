import random

import numpy as np
import torch

from qpgen.seed import set_seed


def test_same_seed_gives_same_numbers():
    set_seed(42)
    first = (random.random(), np.random.rand(), torch.rand(1).item())
    set_seed(42)
    second = (random.random(), np.random.rand(), torch.rand(1).item())
    assert first == second
