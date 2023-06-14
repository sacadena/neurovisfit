from typing import Optional
from typing import Tuple

import numpy as np
from numpy.typing import NDArray


def get_train_val_split(n: int, train_frac: float, seed: Optional[int] = None) -> Tuple[NDArray, NDArray]:
    """
    Splits the total number of images into train and test set.
    This ensures that in every session, the same train and validation images are being used.

    Args:
        n: Total number of images. These will be plit into train and validation set
        train_frac: fraction of images used for the training set
        seed: random seed

    Returns: Two arrays, containing image IDs of the whole imageset, split into train and validation

    """
    np.random.seed(seed)
    train_idx, val_idx = np.split(np.random.permutation(int(n)), [int(n * train_frac)])
    assert not np.any(np.isin(train_idx, val_idx)), "train_set and val_set are overlapping sets"
    return train_idx, val_idx
