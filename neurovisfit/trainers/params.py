from enum import Enum
from typing import Callable
from typing import Optional

from neuralpredictors.measures import modules as ml_measures
from pydantic import BaseModel
from pydantic import PositiveFloat
from pydantic import PositiveInt
from torch import nn

from ..common.config import get_config_file
from ..common.config import TomlConfig
from ..scorers.scores import ScorerNames
from neurovisfit import trainers


class MeasureLossNames(Enum):
    """Names of measure losses implemented
    in neuralpredictors.measures.modules
    """

    CORR = "Corr"
    AVG_CORR = "AvgCorr"
    POISSON_LOSS = "PoissonLoss"
    POISSON_LOSS_3D = "PoissonLoss3d"
    EXPONENTIAL_LOSS = "ExponentialLoss"
    ANSCOMBE_MSE = "AnscombeMSE"

    @property
    def measure(self) -> nn.Module:
        return getattr(ml_measures, self.value)


class TrainerParams(BaseModel):
    avg_loss: bool = False
    scale_loss: bool = False
    loss_function: MeasureLossNames = MeasureLossNames.POISSON_LOSS
    stop_function: ScorerNames = ScorerNames.CORRELATION
    loss_accum_batch_n: Optional[PositiveInt] = None
    verbose: bool = True
    interval: int = 1
    patience: PositiveInt = 5
    epoch: int = 0
    lr_init: PositiveFloat = 0.005
    max_iter: PositiveInt = 100
    maximize: bool = True
    tolerance: float = 1e-6
    restore_best: bool = True
    lr_decay_steps: PositiveInt = 3
    lr_decay_factor: PositiveFloat = 0.3
    min_lr: PositiveFloat = 0.0001
    call_back_function: Optional[Callable] = None
    track_training: bool = False
    return_test_score: bool = False
    batch_ping: int = 1000


def get_trainer_params_from_config(trainer_name: str) -> TrainerParams:
    config = TomlConfig(file_path=get_config_file(package=trainers))
    if trainer_name not in config.available_keys:
        raise ValueError(f"Trainer {trainer_name} not found in config.toml file")
    return TrainerParams(**config.get_dict(trainer_name))
