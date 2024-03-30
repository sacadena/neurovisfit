from abc import abstractmethod

import torch
from torch import nn
from torch.nn.functional import elu


class NonLinearity(nn.Module):
    @abstractmethod
    def forward_implementation(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.forward_implementation(x)


class EluNonLinearity(NonLinearity):
    def __init__(self, offset: float = 0.0):
        super().__init__()
        self.offset = offset

    def forward_implementation(self, x: torch.Tensor) -> torch.Tensor:
        return elu(x + self.offset) + 1.0
