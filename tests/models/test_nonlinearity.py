import pytest
import torch
from torch.nn.functional import elu

from neurovisfit.models.nonlinearity import EluNonLinearity
from neurovisfit.models.nonlinearity import NonLinearity


@pytest.fixture
def sample_input():
    return torch.randn(3, 4, 5)


def test_non_linearity_raises_not_implement_ederror():
    non_linearity = NonLinearity()
    with pytest.raises(NotImplementedError):
        non_linearity(torch.randn(3, 4, 5))


def test_elu_non_linearity_forward(sample_input):
    offset = 0.5
    elu_non_linearity = EluNonLinearity(offset)
    expected_output = elu(sample_input + offset) + 1.0
    assert torch.allclose(elu_non_linearity(sample_input), expected_output)
