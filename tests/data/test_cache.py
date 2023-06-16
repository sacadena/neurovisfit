import torch


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
