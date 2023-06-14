from __future__ import annotations

import ast
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from .json import load_json


@dataclass
class Session:
    subject_id: str
    session_id: str
    responses: np.ndarray
    image_ids: np.ndarray
    previous_image_ids: Optional[np.ndarray]
    trial_ids: Optional[np.ndarray]

    @staticmethod
    def _parse_responses_from_file(responses_col: pd.Series) -> np.ndarray:
        return np.stack([np.array(ast.literal_eval(r)) for r in responses_col]).astype(np.float32)

    @classmethod
    def from_path(cls, session_path: Path) -> Session:
        session_responses = pd.read_csv(str(session_path / "responses.csv"))
        session_metadata = load_json(session_path / "meta_data.json")
        subject_id = session_metadata["subject_id"]
        session_id = str(session_metadata["session_id"])
        responses = cls._parse_responses_from_file(session_responses["responses"])
        image_ids = session_responses["image_id"].values
        trial_ids = session_responses["trial_id"].values if "trial_id" in session_responses.columns else None
        previous_image_ids = (
            session_responses["previous_image_id"].values if "previous_image_id" in session_responses.columns else None
        )
        return Session(
            subject_id=subject_id,
            session_id=session_id,
            responses=responses,
            image_ids=image_ids,
            previous_image_ids=previous_image_ids,
            trial_ids=trial_ids,
        )
