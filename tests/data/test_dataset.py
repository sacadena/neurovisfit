import numpy as np
import pytest
import torch

from neurovisfit.data.dataset import DataSplit
from neurovisfit.data.dataset import ImageResponseDataset
from neurovisfit.data.dataset import InputResponseSelector
from neurovisfit.data.dataset import NamedDataSplit


@pytest.fixture
def mock_input_response_selector():
    image_ids = np.array([1, 2, 3, 4, 5])
    responses = np.array([0, 1, 0, 1, 0])
    image_ids_to_keep = np.array([1, 3, 5])
    fraction_config = None
    kwargs = {"feature1": np.array([0, 1, 0, 1, 0]), "feature2": np.array([1, 0, 1, 0, 1])}

    selector = InputResponseSelector(
        image_ids=image_ids,
        responses=responses,
        image_ids_to_keep=image_ids_to_keep,
        fraction_config=fraction_config,
        **kwargs,
    )

    return selector


def test_input_response_selector_select(mock_input_response_selector):
    assert (mock_input_response_selector._select(np.array([1, 2, 3, 4, 5])) == np.array([1, 3, 5])).all()
    assert (mock_input_response_selector._select(np.array([0, 1, 0, 1, 0])) == np.array([0, 0, 0])).all()
    assert (mock_input_response_selector._select(np.array([1, 0, 1, 0, 1])) == np.array([1, 1, 1])).all()


def test_input_response_selector_image_ids(mock_input_response_selector):
    assert (mock_input_response_selector.image_ids == np.array([1, 3, 5])).all()


def test_input_response_selector_responses(mock_input_response_selector):
    assert (mock_input_response_selector.responses == np.array([0, 0, 0])).all()


def test_named_data_split():
    data_split = DataSplit.TEST
    selector = InputResponseSelector(np.array([1, 2, 3]), np.array([0, 1, 0]))
    named_data_split = NamedDataSplit(name=data_split.value, data=selector)

    assert named_data_split.name == "test"
    assert named_data_split.data is selector


def test_image_response_dataset(mock_image_cache, mock_responses):
    data_split = DataSplit.TEST
    selector = InputResponseSelector(np.array([1, 2, 3]), mock_responses)
    named_data_split = NamedDataSplit(name=data_split.value, data=selector)
    order = ["image_ids", "responses"]

    dataset = ImageResponseDataset(named_data_split, order, mock_image_cache)

    assert len(dataset) == 3

    image_ids, responses = dataset[0]
    assert isinstance(image_ids, torch.Tensor)
    assert isinstance(responses, torch.Tensor)
