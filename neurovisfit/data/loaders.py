from collections import defaultdict
from typing import Any
from typing import Dict
from typing import List
from typing import Optional

import numpy as np
from neuralpredictors.data.samplers import RepeatsBatchSampler
from pydantic import NonNegativeInt
from pydantic import PositiveFloat
from torch.utils.data import DataLoader
from tqdm import tqdm

from .cache import ImageCache
from .dataset import DataSplit
from .dataset import ImageResponseDataset
from .dataset import InputResponseSelector
from .dataset import NamedDataSplit
from .json import load_json_stats
from .json import save_json_stats
from .params import DataLoaderParams
from .params import get_params_from_config
from .params import ImageTransform
from .params import PathLike
from .params import ProcessTimeBins
from .params import TrainWithFractionOfImages
from .session import Session
from neurovisfit.common.hash import make_hash
from neurovisfit.common.splits import get_train_val_split


def get_loader_split(
    params: DataLoaderParams,
    image_cache: ImageCache,
    data_split: DataSplit,
    repeat_condition: Optional[bool] = None,
    batch_size: Optional[int] = None,
    shuffle: bool = True,
) -> Dict[str, Dict[str, DataLoader]]:
    """Get a list of data loaders for the given data split.

    Args:
        params: The data loader parameters.
        image_cache: The image cache.
        data_split: The data split to use.
        repeat_condition: Whether to repeat the data loader.
        batch_size: The batch size to use.
        shuffle: Whether to shuffle the data loader.

    Returns:
        A dictionary with split containing a dictionary with data loaders
        for each session
    """

    # Select images for train and validation
    train_image_ids, validation_image_ids = None, None
    batch_size = batch_size or params.batch_size
    if data_split.value == "train":
        train_image_ids, validation_image_ids = get_train_val_split(
            n=len(image_cache),
            train_frac=1 - params.validation_fraction,
            seed=params.seed,
        )

    session_paths = params.train_sessions if data_split.value == "train" else params.test_sessions

    dataloaders: Dict[str, Dict[str, DataLoader]] = defaultdict(dict)
    print(f"Building {data_split.value} dataloaders for all sessions ...")
    for session_path in tqdm(session_paths):
        session = Session.from_path(session_path)
        session.responses = params.process_time_bins.process(session.responses)

        extra_arrays: Dict[str, np.ndarray] = {}
        if params.include_prev_image and session.previous_image_ids is not None:
            extra_arrays["previous_image_ids"] = session.previous_image_ids
        if params.include_trial_id and session.trial_ids is not None:
            extra_arrays["trial_ids"] = session.trial_ids

        named_data_splits: List[NamedDataSplit] = []
        if data_split.value == "train":
            for sub_split, sub_split_ids in zip(("train", "validation"), (train_image_ids, validation_image_ids)):
                named_data_splits.append(
                    NamedDataSplit(
                        sub_split,
                        InputResponseSelector(
                            image_ids=session.image_ids,
                            responses=session.responses,
                            image_ids_to_keep=sub_split_ids,
                            fraction_config=params.train_with_fraction_of_images,
                            **extra_arrays,
                        ),
                    )
                )
        else:
            named_data_splits.append(
                NamedDataSplit(
                    "test",
                    InputResponseSelector(
                        image_ids=session.image_ids,
                        responses=session.responses,
                        image_ids_to_keep=None,
                        fraction_config=None,
                        **extra_arrays,
                    ),
                ),
            )

        order = ["image_ids"]
        if params.include_prev_image:
            order.append("previous_image_ids")
        if params.include_trial_id:
            order.append("trial_id")
        order.append("responses")

        for named_data_split in named_data_splits:
            dataset = ImageResponseDataset(named_data_split, order=order, image_cache=image_cache)
            sampler = RepeatsBatchSampler(repeat_condition) if repeat_condition is not None else None
            loader = (
                DataLoader(dataset, batch_sampler=sampler)
                if batch_size is None
                else DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
            )
            dataloaders[named_data_split.name].update({session.session_id: loader})

    return dict(dataloaders)


def _get_dataloaders_from_params(
    params: DataLoaderParams,
    shuffle: bool = True,
) -> Dict[str, Dict[str, DataLoader]]:
    """
    Create a dataloader from a DataLoaderParams object
    Args:
        params: DataLoaderParams object
        shuffle: Shuffle samples in data loader
    Returns:
        Dictionary with keys "train", "test", and "validation" containing session dataloaders
    """

    # Load image statistics if present
    params_hash = make_hash(params.dict())
    stats_path = params.data_path / f"cached_stats_config={params_hash}.json"

    if stats_path.is_file():
        stats_config = load_json_stats(stats_path)
        params.image_transform.mean_images = float(np.float32(stats_config.get("mean_images")))
        params.image_transform.std_images = float(np.float32(stats_config.get("std_images")))

    image_cache = ImageCache(
        path=params.data_path / "images",
        image_transformation=params.image_transform,
        filename_precision=6,
    )

    if not stats_path.is_file():
        stats_data: Dict[str, Any] = {}
        mean, std = image_cache.zscore_images()  # zscore images and get mean and std
        params.image_transform.mean_images = mean
        params.image_transform.std_images = std
        stats_data["mean_images"] = mean
        stats_data["std_images"] = std
        save_json_stats(stats_path, stats_data)

    test_loader = get_loader_split(
        params=params,
        image_cache=image_cache,
        data_split=DataSplit("test"),
        shuffle=shuffle,
    )
    train_loader = get_loader_split(
        params=params,
        image_cache=image_cache,
        data_split=DataSplit("train"),
        shuffle=shuffle,
    )

    return {**train_loader, **test_loader}


def get_dataloaders(
    dataset_name: Optional[str] = "default",
    data_path: Optional[PathLike] = None,
    image_transform: Optional[ImageTransform] = None,
    process_time_bins: Optional[ProcessTimeBins] = None,
    batch_size: Optional[NonNegativeInt] = None,
    validation_fraction: Optional[PositiveFloat] = None,
    seed: Optional[NonNegativeInt] = None,
    train_with_fraction_of_images: Optional[TrainWithFractionOfImages] = None,
    include_prev_image: Optional[bool] = None,
    include_trial_id: Optional[bool] = None,
) -> Dict[str, Dict[str, DataLoader]]:
    """
    This function loads the content of the config.toml with key `dataset_name` and parses the parameters with the
    DataLoaderParams schema. If `dataset_name` is None, the parameters are taken from arguments to this function.
    These arguments override the parameters in the config.toml file if they are not None.
    Args:
        dataset_name: name of the dataset to load from the config.toml file
        data_path: PathLike object
        image_transform: ImageTransform object
        process_time_bins: ProcessTimeBins object
        batch_size: batch size
        validation_fraction: fraction of the data to use for validation
        seed: seed for the random number generator
        train_with_fraction_of_images: TrainWithFractionOfImages object
        include_prev_image: whether to include the previous image in the input
        include_trial_id: whether to include the trial id in the input
    Returns:
        Dictionary with keys "train", "test", and "validation" containing session dataloaders
    """

    # Get params from config
    params_config = get_params_from_config(dataset_name) if dataset_name is not None else {}

    # Create input params dict
    params_input: Dict[str, Any] = {}
    for name, field in zip(
        ("image_transform", "process_time_bins", "train_with_fraction_of_images"),
        (image_transform, process_time_bins, train_with_fraction_of_images),
    ):
        if field is not None:
            params_input[name] = field.dict()

    for name, field_num in zip(
        ("data_path", "batch_size", "validation_fraction", "seed", "include_prev_image", "include_trial_id"),
        (data_path, batch_size, validation_fraction, seed, include_prev_image, include_trial_id),
    ):
        if field_num is not None:
            params_input[name] = field_num

    # Override config params with input params
    params = {**params_config, **params_input}

    if params == {}:
        raise ValueError("Missing required parameters")

    # Check compatibility with schema
    if not set(params.keys()).issuperset(set(DataLoaderParams.schema().get("required", []))):
        raise ValueError("Missing required parameters")
    data_loader_params = DataLoaderParams(**params)

    return _get_dataloaders_from_params(data_loader_params)
