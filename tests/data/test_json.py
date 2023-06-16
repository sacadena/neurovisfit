import json
from pathlib import Path

import numpy as np
import pytest

from cookie_test.data.json import load_json
from cookie_test.data.json import load_json_stats
from cookie_test.data.json import save_json_stats


@pytest.fixture
def mock_json_file(tmpdir):
    temp_dir = Path(tmpdir.mkdir("test"))
    file_path = temp_dir / "test_stats.json"
    return file_path


def test_save_json_stats(mock_json_file):
    stats = {
        "value1": np.float32(0.12345),
        "value2": np.float32(0.98765),
        "value3": 42,
    }

    save_json_stats(mock_json_file, stats)

    with mock_json_file.open("r") as f:
        saved_stats = json.load(f)

    assert saved_stats["value1"] == "0.12345"
    assert saved_stats["value2"] == "0.98765"
    assert saved_stats["value3"] == 42


def test_load_json(mock_json_file):
    some_json = {
        "hey": "hello",
        "value": 42,
    }

    with mock_json_file.open("w") as f:
        json.dump(some_json, f)

    loaded_stats = load_json(mock_json_file)

    assert loaded_stats == some_json


def test_load_json_stats(mock_json_file):
    stats = {
        "value1": np.float32(0.12345),
        "value2": np.float32(0.98765),
        "value3": 42.4,
    }

    save_json_stats(mock_json_file, stats)

    loaded_stats = load_json_stats(mock_json_file)

    assert loaded_stats == stats
