from __future__ import annotations

import os
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any
from typing import Callable
from typing import Dict
from typing import List
from typing import Optional
from typing import Sequence
from typing import Tuple
from typing import Union

import numpy as np
import torch
from PIL import Image
from pydantic import BaseModel
from pydantic import NonNegativeFloat
from pydantic import NonNegativeInt
from pydantic import PositiveFloat
from pydantic import PositiveInt
from pydantic import validator
from skimage.transform import rescale

from cookie_test import data
from cookie_test.common.config import get_config_file
from cookie_test.common.config import TomlConfig

PathLike = Union[str, os.PathLike]


@dataclass
class Crop:
    left: int
    right: int
    top: int
    bottom: int

    @staticmethod
    def from_sequence(crop_sequence: Sequence[int]) -> Crop:
        return Crop(
            left=crop_sequence[0],
            right=crop_sequence[1],
            top=crop_sequence[2],
            bottom=crop_sequence[3],
        )


class AggregateBins(Enum):
    SUM = "sum"
    MEAN = "mean"

    @property
    def func(self) -> Callable:
        return getattr(np, self.value)


class ProcessTimeBins(BaseModel):
    bin_duration_ms: NonNegativeFloat
    num_bins: NonNegativeInt
    offset_first_bin_ms: NonNegativeFloat
    window_range_ms: Tuple[NonNegativeFloat, NonNegativeFloat]
    agg_operation: AggregateBins

    @validator("window_range_ms")
    def validate_window_range_ms(cls, v: Tuple[float, float]) -> Tuple[float, float]:
        if v[0] < 0 or v[1] < 0:
            raise ValueError("Window range must be positive")
        if v[0] > v[1]:
            raise ValueError("Window range start must be smaller than end")
        return v

    @property
    def window_bins(self) -> Tuple[int, ...]:
        offset = int(self.offset_first_bin_ms / self.bin_duration_ms)
        return tuple(
            range(
                int(self.window_range_ms[0] / self.bin_duration_ms) - offset,
                int(self.window_range_ms[1] / self.bin_duration_ms) - offset,
            )
        )

    def process(self, responses: np.ndarray) -> np.ndarray:
        """Aggregate bins in the window range"""
        return self.agg_operation.func(responses[:, :, self.window_bins], axis=-1)


class ImageTransform(BaseModel):
    subsample: PositiveInt
    crop: Crop
    scale: PositiveFloat
    mean_images: float = 0.0
    std_images: float = 1.0

    @validator("crop")
    def validate_crop(cls, v: Any) -> Crop:
        if isinstance(v, int):
            return Crop(left=v, right=v, top=v, bottom=v)
        if isinstance(v, list):
            return Crop.from_sequence(v)
        if isinstance(v, tuple):
            return Crop.from_sequence(v)
        if isinstance(v, Crop):
            return v
        raise ValueError("crop must be an int, list, tuple or Crop")

    @staticmethod
    def _rescale_fn(
        x: np.ndarray,
        scale: float,
        channel_axis: Optional[int] = None,
    ) -> np.ndarray:
        return rescale(
            x,
            scale,
            mode="reflect",
            channel_axis=channel_axis,
            anti_aliasing=False,
            preserve_range=True,
        ).astype(x.dtype)

    @staticmethod
    def normalize(x: torch.Tensor, mean: float, std: float) -> torch.Tensor:
        return (x - mean) / std

    def transform(self, image: Image) -> torch.Tensor:
        """
        Transforms the image in the following order: crop, subsample, rescale,
        add channel axis, convert to torch tensor.
        Although subsampling achieves resizing, it is done before rescaling for legacy reasons.
        Ignore subsampling if it is 1.
        Args:
            image: a PIL image
        Returns:
            A transformed image as a numpy array
        """
        image = np.array(
            image.crop(
                (
                    self.crop.left,
                    self.crop.top,
                    image.width - self.crop.right,
                    image.height - self.crop.bottom,
                )
            )
        )
        if self.subsample > 1:
            image = image[
                :: self.subsample,
                :: self.subsample,
            ]

        if self.scale != 1:
            channel_axis = 2 if image.ndim == 3 else None
            image = self._rescale_fn(image, self.scale, channel_axis=channel_axis)

        if image.ndim == 2:
            image = image[None, ...]
        else:
            image = image[None, ...].transpose(0, 3, 1, 2)

        return self.normalize(torch.tensor(image), self.mean_images, self.std_images)


class TrainWithFractionOfImages(BaseModel):
    fraction: PositiveFloat = 1.0
    randomize_selection: bool = False
    selection_seed: Optional[NonNegativeInt] = None

    def get_indices(self, n_images: int) -> np.ndarray:
        """Returns indices of images to use for training"""
        image_indices = np.arange(n_images)

        if self.fraction == 1.0:
            return image_indices

        seed = (
            int(self.selection_seed * self.fraction)
            if (self.randomize_selection and self.selection_seed is not None)
            else self.selection_seed
        )

        np.random.seed(seed)
        indices = np.random.choice(
            image_indices,
            int(n_images * self.fraction),
            replace=False,
        )
        return indices


class DataPath(BaseModel):
    path: Path
    exclude_sessions: Optional[List[str]]

    @validator("path")
    def validate_data_path(cls, val: Path) -> Path:
        """Checks that within data_path there is a folder called "images", and folders "train" and "test".
        Also checks that within train and test, there are the same number of folders with the same names.
        Each sub-folder in train and test must at least contain a file with name responses.csv
        """
        if not val.is_dir():
            raise ValueError("data_path must be a directory")
        if not (val / "images").is_dir():
            raise ValueError("data_path must contain a folder called 'images'")
        if not (val / "train").is_dir():
            raise ValueError("data_path must contain a folder called 'train'")
        if not (val / "test").is_dir():
            raise ValueError("data_path must contain a folder called 'test'")
        train_folders = [f for f in (val / "train").iterdir() if f.is_dir()]
        test_folders = [f for f in (val / "test").iterdir() if f.is_dir()]
        if len(train_folders) != len(test_folders):
            raise ValueError("train and test must contain the same number of folders")
        for train_folder, test_folder in zip(train_folders, test_folders):
            if train_folder.name != test_folder.name:
                raise ValueError("train and test must contain the same folders")
            if not (train_folder / "responses.csv").is_file():
                raise ValueError("train and test folders must contain a file called 'responses.csv'")
            if not (test_folder / "responses.csv").is_file():
                raise ValueError("train and test folders must contain a file called 'responses.csv'")
        return val

    @property
    def train_sessions(self) -> List[Path]:
        return [f for f in (self.path / "train").iterdir() if f.is_dir()]

    @property
    def test_sessions(self) -> List[Path]:
        return [f for f in (self.path / "test").iterdir() if f.is_dir()]


class DataLoaderParams(BaseModel):
    data: DataPath
    image_transform: ImageTransform
    process_time_bins: ProcessTimeBins
    batch_size: NonNegativeInt
    validation_fraction: PositiveFloat
    seed: Optional[NonNegativeInt] = None
    include_prev_image: bool = False
    include_trial_id: bool = False
    train_with_fraction_of_images: Optional[TrainWithFractionOfImages] = None


def get_params_from_config(dataset_name: str) -> Dict[str, Any]:
    config = TomlConfig(file_path=get_config_file(package=data))
    if dataset_name not in config.available_keys:
        raise ValueError(f"Dataset {dataset_name} not found in file")
    return config.get_dict(dataset_name)
