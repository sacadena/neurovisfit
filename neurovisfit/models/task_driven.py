from typing import Optional

from neuralpredictors.layers.readouts import FullGaussian2d
from neuralpredictors.layers.readouts import MultiReadoutBase

from ..common.random import set_random_seed
from .core import TaskDrivenCore
from .model import Model
from .nonlinearity import EluNonLinearity
from .params import TaskDrivenModelParams


def task_driven_core_gauss_readout(params: TaskDrivenModelParams, seed: Optional[int] = None) -> Model:
    core = TaskDrivenCore(**params.core_params.dict())
    if seed is not None:
        set_random_seed(seed)
    core.initialize()

    base_readout = FullGaussian2d(**params.readout_params.base_readout_params.dict())
    sessions_dataloader_settings = params.readout_params.sessions_dataloader_settings
    readout = MultiReadoutBase(
        in_shape_dict=sessions_dataloader_settings.input_shape_per_session,
        n_neurons_dict=sessions_dataloader_settings.num_neurons_per_session,
        base_readout=base_readout,
        mean_activity_dict=params.readout_params.mean_activity,
        clone_readout=params.readout_params.clone_readout,
    )
    non_linearity = EluNonLinearity(offset=params.non_linearity_params.offset)

    return Model(core=core, readout=readout, non_linearity=non_linearity)
