from collections import namedtuple
from enum import Enum
from typing import List
from typing import NamedTuple
from typing import Optional
from typing import Tuple

import numpy as np
import torch
from torch.utils.data import Dataset

from .cache import ImageCache
from .params import TrainWithFractionOfImages


class DataSplit(Enum):
    TRAIN = "train"
    TEST = "test"


class InputResponseSelector:
    def __init__(
        self,
        image_ids: np.ndarray,
        responses: np.ndarray,
        image_ids_to_keep: Optional[np.ndarray] = None,
        fraction_config: Optional[TrainWithFractionOfImages] = None,
        **kwargs: np.ndarray,
    ) -> None:
        self._image_ids = image_ids
        self._responses = responses
        self.fraction_config = fraction_config
        self.image_ids_to_keep = image_ids_to_keep

        for k, v in kwargs.items():
            if not isinstance(v, np.ndarray):
                raise ValueError(f"Expected numpy array for {k}, got {type(v)}")
            setattr(self, f"_{k}", v)
            setattr(self, k, self._select(v))

    def _select(self, array: np.ndarray) -> np.ndarray:
        return array[self._indices_image_ids][self._indices_to_keep]

    @property
    def _indices_image_ids(self) -> np.ndarray:
        if self.image_ids_to_keep is not None:
            return np.isin(self._image_ids, self.image_ids_to_keep)
        return np.ones(len(self._image_ids), dtype=bool)

    @property
    def _indices_to_keep(self) -> np.ndarray:
        if self.fraction_config:
            return self.fraction_config.get_indices(sum(self._indices_image_ids))
        return np.ones(sum(self._indices_image_ids), dtype=bool)

    @property
    def image_ids(self) -> np.ndarray:
        return self._select(self._image_ids)

    @property
    def responses(self) -> np.ndarray:
        return self._select(self._responses)


class NamedDataSplit(NamedTuple):
    name: str
    data: InputResponseSelector


class ImageResponseDataset(Dataset):
    def __init__(
        self,
        named_data_split: NamedDataSplit,
        order: List[str],
        image_cache: ImageCache,
    ) -> None:
        self.named_data_split = named_data_split
        self.order = order
        self.image_cache = image_cache
        self.DataPoint = namedtuple("DataPoint", self.order)  # type: ignore[misc]

    def __len__(self) -> int:
        return len(self.named_data_split.data.image_ids)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, ...]:
        tensors = []
        for key in self.order:
            if key not in self.named_data_split.data.__dir__():
                raise ValueError(f"Key {key} not in data split")

            val = getattr(self.named_data_split.data, key)
            if key in ("image_ids", "previous_image_ids"):
                tensors.append(torch.stack(list(self.image_cache[val[idx]])))  # Turn ids into images
            else:
                tensors.append(torch.from_numpy(val[idx]).to(torch.float))
        return self.DataPoint(*tensors)
