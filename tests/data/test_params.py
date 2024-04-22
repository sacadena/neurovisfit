import numpy as np
from PIL import Image

from neurovisfit.data.params import AggregateBins
from neurovisfit.data.params import Crop
from neurovisfit.data.params import DataLoaderParams
from neurovisfit.data.params import ImageTransform
from neurovisfit.data.params import ProcessTimeBins
from neurovisfit.data.params import TrainWithFractionOfImages


def test_crop_from_sequence():
    crop_sequence = [1, 2, 3, 4]
    crop = Crop.from_sequence(crop_sequence)

    assert crop.top == 1
    assert crop.bottom == 2
    assert crop.left == 3
    assert crop.right == 4


def test_aggregate_bins():
    assert AggregateBins.SUM.func == np.sum
    assert AggregateBins.MEAN.func == np.mean


def test_process_time_bins():
    bins = ProcessTimeBins(
        bin_duration_ms=10,
        num_bins=10,
        offset_first_bin_ms=50.0,
        window_range_ms=(60.0, 110.0),
        agg_operation=AggregateBins.MEAN,
    )

    assert bins.window_bins == (
        1,
        2,
        3,
        4,
        5,
    )


def test_image_transform():
    transform = ImageTransform(
        subsample=2,
        crop=Crop(10, 10, 10, 10),
        scale=0.5,
        mean_images=0.0,
        std_images=1.0,
    )

    image = Image.new("RGB", (100, 100))
    transformed_image = transform.transform(image)

    assert transformed_image.shape == (1, 3, 20, 20)
    assert transformed_image.mean() == np.array(image).mean()
    assert transformed_image.std() == np.array(image).std()


def test_train_with_fraction_of_images():
    train = TrainWithFractionOfImages(
        fraction=0.5,
        randomize_selection=True,
        selection_seed=42,
    )

    n_images = 10
    indices = train.get_indices(n_images)
    assert len(indices) == 5


def test_data_loader_params(mock_data_path):
    params = DataLoaderParams(
        data_path=mock_data_path,
        image_transform=ImageTransform(
            subsample=2,
            crop=Crop(10, 10, 10, 10),
            scale=0.5,
            mean_images=0.0,
            std_images=1.0,
        ),
        process_time_bins=ProcessTimeBins(
            bin_duration_ms=10,
            num_bins=10,
            offset_first_bin_ms=50.0,
            window_range_ms=(50.0, 150.0),
            agg_operation=AggregateBins.MEAN,
        ),
        batch_size=32,
        validation_fraction=0.2,
        seed=42,
        include_prev_image=False,
        include_trial_id=False,
        train_with_fraction_of_images=TrainWithFractionOfImages(
            fraction=0.5,
            randomize_selection=False,
            selection_seed=42,
        ),
    )

    assert params.data_path == mock_data_path
    assert params.image_transform.subsample == 2
    assert params.image_transform.crop.left == 10
    assert params.process_time_bins.bin_duration_ms == 10.0
    assert params.batch_size == 32
    assert params.validation_fraction == 0.2
    assert params.seed == 42
    assert params.include_prev_image is False
    assert params.include_trial_id is False
    assert params.train_with_fraction_of_images.fraction == 0.5
    assert params.train_with_fraction_of_images.randomize_selection is False
    assert params.train_with_fraction_of_images.selection_seed == 42
