from pathlib import Path

import numpy as np
import pytest
import torch
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


def test_image_cache_len(mock_image_cache) -> None:
    assert len(mock_image_cache) == 3


def test_image_cache_contains(mock_image_cache) -> None:
    assert 1 in mock_image_cache
    assert 2 in mock_image_cache
    assert 3 in mock_image_cache
    assert 4 not in mock_image_cache


def test_image_cache_getitem(mock_image_cache) -> None:
    image = mock_image_cache[1]
    assert isinstance(image, torch.Tensor)


def test_image_cache_from_image_id_to_filename(mock_image_cache) -> None:
    assert mock_image_cache.from_image_id_to_filename(1) == "000001"


def test_image_cache_from_filename_to_image_id(mock_image_cache) -> None:
    assert mock_image_cache.from_filename_to_image_id("000001") == 1


def test_image_cache_zscore_images(mock_image_cache) -> None:
    mean, std = mock_image_cache.zscore_images()
    assert isinstance(mean, float)
    assert isinstance(std, float)
