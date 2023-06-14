from pathlib import Path
from typing import Any
from typing import Dict
from typing import Optional
from typing import Tuple

import numpy as np
import torch
from multiprocess import Pool
from PIL import Image
from tqdm import tqdm

from .params import ImageTransform


class ImageCache:
    """A class to cache images from a directory"""

    def __init__(
        self,
        path: Path,
        image_transformation: Optional[ImageTransform] = None,
        filename_precision: int = 6,
    ) -> None:
        self.path = path
        self.image_transformation = image_transformation
        self._cache: Dict[int, torch.Tensor] = {}
        self.filename_precision = filename_precision

    def __len__(self) -> int:
        """List all files in self.path that end with .png"""
        return len([file for file in self.path.iterdir() if file.name.endswith(".png")])

    def __contains__(self, item: str) -> bool:
        return item in self._cache

    def __getitem__(self, item: int) -> Any:
        if item not in self._cache:
            self._cache[item] = self._load_image(item)
        return self._cache[item]

    def from_image_id_to_filename(self, item: int) -> str:
        return str(item).zfill(self.filename_precision)

    @staticmethod
    def from_filename_to_image_id(name: str) -> int:
        return int(name)

    def _load_image(self, item: int) -> torch.Tensor:
        """Load an image from the cache, or from disk if it is not yet in the cache"""
        filename = str(self.path / f"{self.from_image_id_to_filename(item)}.png")
        image = Image.open(filename)
        image = self.image_transformation.transform(image) if self.image_transformation else image
        image = torch.Tensor(image).to(torch.float)
        return image

    @property
    def images(self) -> torch.Tensor:
        items = [
            self.from_filename_to_image_id(file.stem) for file in self.path.iterdir() if file.name.endswith(".png")
        ]
        # If all images are loaded in cache, return them as a tensor
        if set(self._cache.keys()).issuperset(set(items)):
            return torch.stack([self[item] for item in items])

        print("Loading and caching transformed images ...")
        # Use multiprocessing to load and transform images in parallel
        with Pool() as pool, tqdm(total=len(items), position=0, leave=True) as pbar:
            # This does not update the cache
            outs = list(tqdm(pool.imap(self.__getitem__, items), total=len(items), position=0, leave=True))
            pbar.update()
        # Update the cache
        for item, out in zip(items, outs):
            self._cache[item] = out

        return torch.stack([self[item] for item in items])

    def zscore_images(self) -> Tuple[float, float]:
        """
        zscore images in cache and returns mean and std of images
        """
        images = self.images
        img_mean = images.mean()
        img_std = images.std()

        for item in self._cache:
            self._cache[item] = ImageTransform.normalize(self._cache[item], img_mean, img_std)

        img_mean_array, img_std_array = np.float32(img_mean.item()), np.float32(img_std.item())

        if self.image_transformation is not None:
            self.image_transformation.mean_images = img_mean_array
            self.image_transformation.std_images = img_std_array

        return img_mean_array, img_std_array