import csv
import json
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from neurovisfit.common.config import TomlConfig
from neurovisfit.data.cache import ImageCache


@pytest.fixture
def mock_image_cache(tmpdir) -> ImageCache:
    # Create a temporary directory for testing
    temp_dir = Path(tmpdir.mkdir("images"))

    # Create some test images
    image1 = Image.fromarray((np.random.randn(10, 10) * 255).astype(np.uint8))
    image1.save(temp_dir / "000001.png")

    image2 = Image.fromarray((np.random.randn(10, 10) * 255).astype(np.uint8))
    image2.save(temp_dir / "000002.png")

    image3 = Image.fromarray((np.random.randn(10, 10) * 255).astype(np.uint8))
    image3.save(temp_dir / "000003.png")

    # Create an instance of ImageCache for testing
    image_cache = ImageCache(temp_dir, filename_precision=6, parallelized=False)
    _ = image_cache.images

    return image_cache


@pytest.fixture
def mock_responses() -> np.ndarray:
    return np.array([[0, 1, 0], [0, 0, 1], [1, 1, 1]])


@pytest.fixture
def mock_session_path(tmpdir):
    temp_dir = Path(tmpdir.mkdir("test"))
    session_path = temp_dir / "session"
    session_path.mkdir()

    responses_csv = session_path / "responses.csv"
    with responses_csv.open("w") as f:
        writer = csv.writer(f)
        writer.writerow(["responses", "image_id"])
        writer.writerow(["[0, 1, 0]", "1"])
        writer.writerow(["[0, 0, 1]", "2"])
        writer.writerow(["[1, 1, 1]", "3"])

    meta_data_json = session_path / "meta_data.json"
    with meta_data_json.open("w") as f:
        json.dump({"subject_id": "001", "session_id": "123"}, f)

    return session_path


@pytest.fixture
def mock_data_path(tmpdir):
    from pathlib import Path

    temp_dir = Path(tmpdir.mkdir("data"))
    (temp_dir / "images").mkdir()
    image = Image.fromarray((np.random.randn(10, 10) * 255).astype(np.uint8))
    image.save(temp_dir / "images/000001.png")
    image = Image.fromarray((np.random.randn(10, 10) * 255).astype(np.uint8))
    image.save(temp_dir / "images/000002.png")
    for session in range(1, 4):
        for split in ("train", "test"):
            session_path = temp_dir / f"{split}/session{session}"
            session_path.mkdir(parents=True)
            with (session_path / "responses.csv").open("w") as f:
                writer = csv.writer(f)
                writer.writerow(["responses", "image_id"])
                writer.writerow(["[[0, 1, 0],[1, 1, 0]]", "1"])
                writer.writerow(["[[1, 1, 0],[0, 1, 0]]", "2"])
            with (session_path / "meta_data.json").open("w") as f:
                json.dump({"subject_id": "001", "session_id": "123"}, f)
    return temp_dir


@pytest.fixture
def mock_toml_config(tmp_path, mock_data_path):
    # Create a temporary file with test data
    toml_data = f"""
    [some_data_name]
    dataset = "some_data_name"
    batch_size = 32
    validation_fraction = 0.2
    seed = 42
    include_prev_image = false
    include_trial_id = false

    [some_data_name.data]
    path = "{str(mock_data_path)}"
    exclude = []

    [some_data_name.image_transform]
    subsample = 1
    crop = [2, 2, 2, 2]
    scale = 1.0

    [some_data_name.process_time_bins]
    bin_duration_ms = 10
    num_bins = 12
    offset_first_bin_ms = 40
    window_range_ms = [40, 70]
    agg_operation = "mean"

    [some_data_name.train_with_fraction_of_images]
    fraction = 1.0
    randomize_selection = false
    """
    file_path = tmp_path / "test_config.toml"
    with open(file_path, "w") as f:
        f.write(toml_data)

    # Create an instance of TomlConfig for testing
    toml_config = TomlConfig(file_path)

    return toml_config

    # Create an instance of TomlConfig for testing
    toml_config = TomlConfig(file_path)

    return toml_config
