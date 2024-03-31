import pytest
import torch

from neurovisfit.models.core import TaskDrivenCore


@pytest.fixture(scope="module")
def sample_input():
    return torch.randn(1, 3, 224, 224)  # Example input tensor


@pytest.fixture(scope="module")
def task_core_layer3_resnet50():
    return TaskDrivenCore(
        input_channels=3,
        model_name="resnet50",
        layer_name="layer3.0",
        pretrained=False,
        bias=False,
        final_batchnorm=False,
        final_nonlinearity=False,
        momentum=0.1,
        fine_tune=False,
    )


def test_taskdrivencore_initialize(sample_input, task_core_layer3_resnet50):
    actual = task_core_layer3_resnet50(sample_input).shape
    expected = torch.Size([1, 1024, 14, 14])
    torch.testing.assert_close(actual, expected)
