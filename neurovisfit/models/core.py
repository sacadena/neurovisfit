import warnings
from typing import Any
from typing import Callable

import ptrnets
import torch
from ptrnets.utils.mlayer import clip_model
from ptrnets.utils.mlayer import hook_model_module
from torch import nn


class Core2d(nn.Module):
    def initialize(self, cuda: bool = False) -> None:
        self.apply(self.init_conv)
        self.put_to_cuda(cuda=cuda)

    def put_to_cuda(self, cuda: bool) -> None:
        if cuda:
            self.cuda()

    @staticmethod
    def init_conv(m: nn.Module) -> None:
        if isinstance(m, nn.Conv2d):
            nn.init.xavier_normal_(m.weight.data)
            if m.bias is not None:
                m.bias.data.fill_(0)


class TaskDrivenCore(Core2d):
    def __init__(
        self,
        input_channels: int,
        model_name: str,
        layer_name: str,
        pretrained: bool = True,
        bias: bool = False,
        final_batchnorm: bool = True,
        final_nonlinearity: bool = True,
        momentum: float = 0.1,
        fine_tune: bool = False,
        **kwargs: Any,
    ) -> None:
        """
        Core from pretrained networks on image tasks.

        Args:
            input_channels (int): Number of input channels. 1 if greyscale, 3 if RBG
            model_name (str): Name of the image recognition task model. Possible are all
                models in ptrnets and torchvision.models
            layer_name (str): Name of the layer at which to clip the model
            pretrained (boolean): Whether to use a randomly initialized or pretrained
                network (default: True)
            bias (boolean): Whether to keep bias weights in the output layer (default: False)
            final_batchnorm (boolean): Whether to add a batch norm after the final
                conv layer (default: True)
            final_nonlinearity (boolean): Whether to add a final nonlinearity
                (ReLU) (default: True)
            momentum (float): Momentum term for batch norm. Irrelevant if batch_norm=False
            fine_tune (boolean): Whether to freeze gradients of the core or to allow training
        """
        if kwargs:
            warnings.warn(
                f"Ignoring input {repr(kwargs)} when creating {self.__class__.__name__}",
                UserWarning,
            )
        super().__init__()

        self.input_channels = input_channels
        self.momentum = momentum
        self.use_probe = False
        self.layer_name = layer_name
        self.pretrained = pretrained

        # Download model and cut after specified layer
        self.model = getattr(ptrnets, model_name)(pretrained=pretrained)
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        self.model.to(self.device)

        # Decide whether to probe the model with a forward hook or to clip the model by
        # replicating architecture of the model up to layer :layer_name:

        x = torch.randn(1, 3, 224, 224).to(self.device)
        try:
            self.model.eval()
            model_clipped = clip_model(self.model, self.layer_name)
            clip_out = model_clipped(x)
        except (ValueError, RuntimeError):
            warnings.warn(f"Unable to clip model {model_name} at layer {self.layer_name}. Using a probe instead")
            self.use_probe = True

        self.model_probe = self.probe_model()

        if not self.use_probe:
            if not (torch.allclose(self.model_probe(x), clip_out)):
                warnings.warn("Unable to recover model outputs via a sequential modules. Using forward hook instead")
                self.use_probe = True

        # Remove the bias of the last conv layer if not :bias:
        if not bias and not self.use_probe:
            if "bias" in model_clipped[-1]._parameters:
                if model_clipped[-1].bias is not None:
                    zeros = torch.zeros_like(model_clipped[-1].bias)
                    model_clipped[-1].bias.data = zeros

        # Fix pretrained parameters during training
        if not fine_tune and not self.use_probe:
            for param in model_clipped.parameters():
                param.requires_grad = False

        # Stack model modules
        self.features = nn.Sequential()

        if not self.use_probe:
            self.features.add_module("TaskDriven", model_clipped)

        if final_batchnorm:
            self.features.add_module(
                "OutBatchNorm",
                nn.BatchNorm2d(self.outchannels, momentum=self.momentum),
            )
        if final_nonlinearity:
            self.features.add_module("OutNonlin", nn.ReLU(inplace=True))

        # Remove model module if not(self.use_probe):

        if not self.use_probe:
            del self.model

    def forward(self, input_: torch.Tensor) -> torch.Tensor:
        # If model is designed for RBG input but input is greyscale,
        # repeat the same input 3 times
        if self.input_channels == 1:
            input_ = input_.repeat(1, 3, 1, 1)

        if self.use_probe:
            input_ = self.model_probe(input_)

        input_ = self.features(input_)
        return input_

    def regularizer(self) -> int:
        return 0  # useful for final loss

    def probe_model(self) -> Callable:
        named_modules = [n for n, _ in self.model.named_modules()]
        if self.layer_name not in named_modules:
            raise ValueError(f"No module named {self.layer_name}")
        hook = hook_model_module(self.model, self.layer_name)

        def func(x: torch.Tensor) -> Any:
            try:
                self.model(x)
            except (ValueError, RuntimeError):
                pass
            return hook(self.layer_name)

        return func

    @property
    def outchannels(self) -> int:
        """
        Function which returns the number of channels in the output conv layer.
        If the output layer is not a conv layer, the last conv layer in the
        network is used.

        Returns: Number of output channels
        """
        x = torch.randn(1, 3, 224, 224).to(self.device)
        if self.use_probe:
            outch = self.model_probe(x).shape[1]
        else:
            task_driven = self.features.get_submodule("TaskDriven")
            outch = task_driven(x).shape[1]
        return outch

    def initialize(self, cuda: bool = False) -> None:
        # Overwrite parent class's initialize function
        if not self.pretrained:
            self.apply(self.init_conv)
        self.put_to_cuda(cuda=cuda)
