from typing import Any
from typing import Dict
from typing import Optional

from ..common.module import dynamic_import
from ..common.module import split_module_name
from ..data.loaders import get_dataloaders
from .model import Model
from .params import get_params_from_config
from .params import SessionsDataLoaderSettings
from .params import TaskDrivenModelGaussianReadoutParams


def build_model(
    model_config_name: Optional[str] = None,
    sessions_data_loader_settings: Optional[SessionsDataLoaderSettings] = None,
    dataloaders: Optional[Dict[str, Any]] = None,
    seed: Optional[int] = None,
) -> Model:
    config = get_params_from_config(model_config_name) if model_config_name is not None else {}
    if sessions_data_loader_settings is None:
        dataloaders = dataloaders or get_dataloaders(config["dataset"])
        sessions_data_loader_settings = SessionsDataLoaderSettings.from_dataloaders(
            dataloaders,
        )

    module_path, class_name = split_module_name(config["model_function"])
    model_function = dynamic_import(module_path, class_name)

    model_function_params = TaskDrivenModelGaussianReadoutParams(**config)
    return model_function(model_function_params, sessions_data_loader_settings, seed=seed)
