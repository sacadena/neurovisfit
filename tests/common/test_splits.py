import numpy as np

from cookie_test.common.splits import get_train_val_split


def test_get_train_val_split():
    n = 1000
    train_frac = 0.8
    seed = 123

    train_idx, val_idx = get_train_val_split(n, train_frac, seed)

    assert isinstance(train_idx, np.ndarray)
    assert isinstance(val_idx, np.ndarray)

    assert len(train_idx) + len(val_idx) == n

    # Check that the same seed produces the same split
    train_idx2, val_idx2 = get_train_val_split(n, train_frac, seed)
    assert np.array_equal(train_idx, train_idx2)
    assert np.array_equal(val_idx, val_idx2)

    # Check that the train and validation indices are non-overlapping
    assert not np.any(np.isin(train_idx, val_idx))
    assert not np.any(np.isin(val_idx, train_idx))
