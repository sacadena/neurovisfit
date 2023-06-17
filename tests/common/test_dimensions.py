from collections import namedtuple
from typing import Dict
from typing import Tuple

import pytest
import torch
from torch.utils.data import DataLoader
from torch.utils.data import Dataset

from neurovisfit.common.dimensions import get_dims_for_loader_dict
from neurovisfit.common.dimensions import get_io_dims


@pytest.fixture
def mock_named_tuple_dataset():
    class MockDataset(Dataset):
        def __init__(self) -> None:
            self.inputs = torch.tensor([1, 2, 3, 4])
            self.responses = torch.tensor([5, 6, 7, 8])

        def __len__(self) -> int:
            return len(self.inputs)

        def __getitem__(self, key: int) -> Tuple[torch.Tensor, torch.Tensor]:
            SomeData = namedtuple("SomeData", ["inputs", "responses"])
            return SomeData(
                inputs=self.inputs[key],
                responses=self.responses[key],
            )

    return MockDataset()


@pytest.fixture
def mock_dict_dataset():
    class MockDataset(Dataset):
        def __init__(self) -> None:
            self.inputs = torch.tensor([1, 2, 3, 4])
            self.responses = torch.tensor([5, 6, 7, 8])

        def __len__(self) -> int:
            return len(self.inputs)

        def __getitem__(self, key: int) -> Dict[str, torch.Tensor]:
            return {
                "inputs": self.inputs[key],
                "responses": self.responses[key],
            }

    return MockDataset()


def test_get_io_dims(mock_named_tuple_dataset, mock_dict_dataset):
    data = torch.tensor([1, 2, 3, 4])  # Example data
    batch_size = 2
    data_loader = DataLoader(data, batch_size=batch_size)

    result = get_io_dims(data_loader)

    assert isinstance(result, tuple)
    assert len(result) == 2

    data_loader = DataLoader(mock_named_tuple_dataset, batch_size=batch_size)
    result = get_io_dims(data_loader)
    assert isinstance(result, dict)

    data_loader = DataLoader(mock_dict_dataset, batch_size=batch_size)
    result = get_io_dims(data_loader)
    assert isinstance(result, dict)


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
