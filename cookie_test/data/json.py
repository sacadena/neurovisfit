import json
from pathlib import Path
from typing import Any
from typing import Dict

import numpy as np


def save_json_stats(file: Path, stats: Dict[str, Any]) -> None:
    for k, v in stats.items():
        if type(v) is np.float32:
            stats[k] = str(round(v, 5))
    with file.open("w") as f:
        json.dump(stats, f)


def load_json(file: Path) -> Dict[str, Any]:
    with file.open("r") as f:
        return json.load(f)


def load_json_stats(file: Path) -> Dict[str, Any]:
    with file.open("r") as f:
        return json.load(f)
