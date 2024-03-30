from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any
from typing import Dict
from typing import Optional
from typing import Sequence

import numpy as np
from pydantic import BaseModel


class TaskDrivenCoreParams(BaseModel):
    input_channels: int
    model_name: str
    layer_name: str
    pretrained: bool = True
    bias: bool = False
    final_batchnorm: bool = True
    final_nonlinearity: bool = True
    momentum: float = 0.1
    fine_tune: bool = False


class GaussianType(Enum):
    ISOTROPIC = "isotropic"
    UNCORRELATED = "uncorrelated"
    FULL = "full"


class FullGaussian2dReadoutParams(BaseModel):
    in_shape: Sequence
    outdims: int
    bias: bool
    init_mu_range: float
    init_sigma: float
    batch_sample: bool
    align_corners: bool
    gauss_type: GaussianType = GaussianType.FULL
    grid_mean_predictor: Optional[Any] = None
    shared_features: Optional[Any] = None
    shared_grid: Optional[Any] = None
    source_grid: Optional[np.ndarray] = None
    mean_activity: Optional[Sequence[int]] = None
    feature_reg_weight: float = 1.0


@dataclass
class SessionsDataLoaderSettings:
    num_neurons_per_session: Dict[str, int]
    input_shape_per_session: Dict[str, Sequence[int]]

    @classmethod
    def from_dataloader(
        cls,
        dataloaders: Dict[str, Any],
    ) -> SessionsDataLoaderSettings:
        def _get_dimensions_per_loader(data_loader: Sequence[Any]) -> Any:
            session_loader = next(iter(data_loader))
            if isinstance(session_loader, tuple) and hasattr(session_loader, "_fields"):  # namedtuple
                session_loader = getattr(session_loader, "_asdict")()
            if hasattr(session_loader, "items"):  # if dict like
                return {key: loader.shape for key, loader in session_loader.items()}
            return (loader.shape for loader in session_loader)

        dataloaders = dataloaders if "train" not in dataloaders else dataloaders["train"]
        # Obtain the named tuple fields from the first entry of the first dataloader in the dictionary
        example_dataloader_val = next(iter(list(dataloaders.values())[0]))
        in_name, out_name = getattr(example_dataloader_val, "_fields")[:2]

        session_dimensions = {session: _get_dimensions_per_loader(loader) for session, loader in dataloaders.items()}
        return cls(
            num_neurons_per_session={session: v[out_name][1] for session, v in session_dimensions.items()},
            input_shape_per_session={session: v[in_name] for session, v in session_dimensions.items()},
        )

    @classmethod
    def from_cached_data_info(cls, cached_data_info: Dict[str, Any]) -> SessionsDataLoaderSettings:
        return cls(
            num_neurons_per_session={k: v["output_dimension"] for k, v in cached_data_info.items()},
            input_shape_per_session={k: v["input_dimensions"] for k, v in cached_data_info.items()},
        )


class MultiSessionReadoutParams(BaseModel):
    sessions_dataloader_settings: SessionsDataLoaderSettings  # needed to shape readouts
    base_readout_params: FullGaussian2dReadoutParams
    mean_activity: Optional[Dict[str, Sequence[int]]]
    clone_readout: bool = False


class EluNonLinearityParams(BaseModel):
    offset: float = -1.0


class TaskDrivenModelParams(BaseModel):
    seed: int
    core_params: TaskDrivenCoreParams
    readout_params: MultiSessionReadoutParams
    non_linearity_params: EluNonLinearityParams
