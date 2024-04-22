from typing import Optional

from neuralpredictors.utils import get_module_output

from ..common.random import set_random_seed
from .core import TaskDrivenCore
from .model import Model
from .nonlinearity import EluNonLinearity
from .params import SessionsDataLoaderSettings
from .params import TaskDrivenModelGaussianReadoutParams
from .readout import MultiReadoutFullGaussian2d


def task_driven_core_gauss_readout(
    params: TaskDrivenModelGaussianReadoutParams,
    sessions_dataloader_settings: SessionsDataLoaderSettings,
    seed: Optional[int] = None,
) -> Model:
    # Set up core
    core = TaskDrivenCore(**params.core_params.dict())
    if seed is not None:
        set_random_seed(seed)
    core.initialize()

    # Core output shapes per sessions
    core_output_shape_per_session = {
        session: get_module_output(core, input_shape)[1:]
        for session, input_shape in sessions_dataloader_settings.input_shape_per_session.items()
    }

    # Set up readout for all sessions
    readout = MultiReadoutFullGaussian2d(
        in_shape_dict=core_output_shape_per_session,
        n_neurons_dict=sessions_dataloader_settings.num_neurons_per_session,
        mean_activity_dict=params.readout_params.mean_activity,
        clone_readout=params.readout_params.clone_readout,
        **params.readout_params.base_readout_params.dict(),
    )

    # Add non-linearity
    non_linearity = EluNonLinearity(offset=params.non_linearity_params.offset)

    # Build and return model
    return Model(core=core, readout=readout, non_linearity=non_linearity)
