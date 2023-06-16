from neurovisfit.data.dataset import DataSplit
from neurovisfit.data.loaders import _get_dataloaders_from_params
from neurovisfit.data.loaders import get_loader_split
from neurovisfit.data.params import DataLoaderParams


def test_get_loader_split(mock_image_cache, mock_toml_config):
    params = DataLoaderParams(**mock_toml_config.get_dict("some_data_name"))
    data_split = DataSplit.TRAIN
    result = get_loader_split(params, mock_image_cache, DataSplit.TRAIN)
    assert isinstance(result, dict)
    assert len(result) == 2
    assert data_split.value in result.keys()


def test__get_dataloaders_from_params(mock_toml_config):
    params = DataLoaderParams(**mock_toml_config.get_dict("some_data_name"))
    result = _get_dataloaders_from_params(params)
    assert isinstance(result, dict)
    assert len(result) == 3
    assert "train" in result.keys()
    assert "test" in result.keys()
    assert isinstance(result["train"], dict)
    assert isinstance(result["test"], dict)
