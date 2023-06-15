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
