from importlib.resources import open_text

import pytest

from neurovisfit.common.config import TomlConfig


def test_toml_config_get_dict(mock_toml_config, mock_data_path):
    database_config = mock_toml_config.get_dict("some_data_name")
    print(database_config)
    assert isinstance(database_config, dict)
    assert database_config["dataset"] == "some_data_name"
    assert database_config["batch_size"] == 32
    assert database_config["validation_fraction"] == 0.2
    assert database_config["seed"] == 42
    assert database_config["include_prev_image"] is False
    assert database_config["include_trial_id"] is False
    assert database_config["data"]["path"] == str(mock_data_path)
    assert database_config["data"]["exclude"] == []
    assert database_config["image_transform"]["subsample"] == 1
    assert database_config["image_transform"]["crop"] == [2, 2, 2, 2]
    assert database_config["image_transform"]["scale"] == 1.0
    assert database_config["process_time_bins"] == {
        "bin_duration_ms": 10,
        "num_bins": 12,
        "offset_first_bin_ms": 40,
        "window_range_ms": [40, 70],
        "agg_operation": "mean",
    }
    assert database_config["train_with_fraction_of_images"] == {
        "fraction": 1.0,
        "randomize_selection": False,
    }


def test_toml_config_available_keys(mock_toml_config):
    available_keys = mock_toml_config.available_keys
    assert isinstance(available_keys, list)
    assert "some_data_name" in available_keys


def test_toml_config_invalid_key(mock_toml_config):
    with pytest.raises(ValueError):
        mock_toml_config.get_dict("invalid_key")


def test_toml_config_invalid_file():
    with pytest.raises(ValueError):
        TomlConfig("nonexistent_file.toml")


def test_get_config_file():
    with open_text("neurovisfit.data", "config.toml") as f:
        config_file = f.read()
    assert isinstance(config_file, str)
