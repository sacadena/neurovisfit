import torch
from torch.utils.data import DataLoader

from cookie_test.common.dimensions import get_dims_for_loader_dict
from cookie_test.common.dimensions import get_io_dims


def test_get_io_dims():
    data = torch.tensor([1, 2, 3, 4])  # Example data
    batch_size = 2
    data_loader = DataLoader(data, batch_size=batch_size)

    result = get_io_dims(data_loader)

    assert isinstance(result, tuple)
    assert len(result) == 2


def test_get_dims_for_loader_dict():
    data1 = torch.tensor([1, 2, 3, 4])  # Example data 1
    data2 = torch.tensor([5, 6, 7, 8])  # Example data 2
    batch_size = 2
    data_loader1 = DataLoader(data1, batch_size=batch_size)
    data_loader2 = DataLoader(data2, batch_size=batch_size)

    dataloaders = {"loader1": data_loader1, "loader2": data_loader2}

    result = get_dims_for_loader_dict(dataloaders)

    assert isinstance(result, dict)
    assert len(result) == 2

    assert "loader1" in result
    assert isinstance(result["loader1"], tuple)
    assert len(result["loader1"]) == 2
