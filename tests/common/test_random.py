import numpy as np
import torch

from cookie_test.common.random import set_random_seed


def test_set_random_seed():
    seed = 123
    deterministic = True

    set_random_seed(seed, deterministic)

    # Check NumPy random seed
    assert np.random.get_state()[1][0] == seed

    # Check PyTorch random seed
    assert torch.initial_seed() == seed

    # Check PyTorch CUDA random seed if CUDA is available
    if torch.cuda.is_available():
        assert torch.cuda.initial_seed() == seed

    # Check PyTorch CUDNN backend settings if deterministic=True and CUDA is available
    if deterministic and torch.cuda.is_available():
        assert not torch.backends.cudnn.benchmark
        assert torch.backends.cudnn.deterministic
