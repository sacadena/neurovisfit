from typing import Any
from typing import Dict
from typing import Optional
from typing import Union

import torch
from neuralpredictors.layers.readouts import MultiReadoutBase
from torch import nn

from .core import Core2d
from .nonlinearity import NonLinearity


class Model(nn.Module):
    def __init__(
        self,
        core: Core2d,
        readout: MultiReadoutBase,
        non_linearity: NonLinearity,
    ) -> None:
        super().__init__()
        self.core = core
        self.readout = readout
        self.non_linearity = non_linearity

    def forward(
        self,
        x: torch.Tensor,
        data_key: Optional[str] = None,
        repeat_channel_dim: Optional[int] = None,
        **kwargs: Dict[str, Any],
    ) -> torch.Tensor:
        if repeat_channel_dim is not None:
            x = x.repeat(1, repeat_channel_dim, 1, 1)
            x[:, 1:, ...] = 0
        x = self.core(x)
        x = self.readout(x, data_key=data_key, **kwargs)
        x = self.non_linearity(x)
        return x

    def regularizer(self, data_key: Optional[str] = None) -> Union[int, torch.Tensor]:
        return self.core.regularizer() + self.readout.regularizer(data_key=data_key)
