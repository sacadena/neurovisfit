import io
import os
from contextlib import contextmanager
from importlib.resources import Package
from importlib.resources import path
from pathlib import Path
from typing import Any
from typing import BinaryIO
from typing import Dict
from typing import Iterator
from typing import List
from typing import Union

import tomli

PathLike = Union[str, os.PathLike]


class TomlConfig:
    def __init__(self, file_path: PathLike) -> None:
        self._file_path = Path(file_path)
        with self._get_stream() as stream:
            self._dict = tomli.load(stream)

    @contextmanager
    def _get_stream(self) -> Iterator[BinaryIO]:
        if self._file_path.is_file():
            with open(self._file_path, mode="rb") as stream:
                yield io.BytesIO(stream.read())
        else:
            raise ValueError(f"Could not find configuration file {self._file_path}")

    def get_dict(self, key: str) -> Dict[str, Any]:
        d = self._dict.get(key, {})
        if not isinstance(d, dict):
            raise ValueError(f"Not a dictionary: {key}")
        return d

    @property
    def available_keys(self) -> List[str]:
        return list(self._dict.keys())


def get_config_file(package: Package) -> Path:
    with path(package, "config.toml") as f:
        return f
