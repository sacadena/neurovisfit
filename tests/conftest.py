import csv
import json
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from cookie_test.data.cache import ImageCache


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
    for session in range(1, 4):
        for split in ("train", "test"):
            session_path = temp_dir / f"{split}/session{session}"
            session_path.mkdir(parents=True)
            with (session_path / "responses.csv").open("w") as f:
                writer = csv.writer(f)
                writer.writerow(["responses", "image_id"])
                writer.writerow(["[0, 1, 0]", "1"])
            with (session_path / "meta_data.json").open("w") as f:
                json.dump({"subject_id": "001", "session_id": "123"}, f)
    return temp_dir
