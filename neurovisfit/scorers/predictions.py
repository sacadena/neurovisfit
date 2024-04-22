from typing import Tuple

import numpy as np
import torch
from neuralpredictors.training import device_state
from torch.utils.data import DataLoader

from neurovisfit.models.model import Model


def get_model_predictions_and_targets(
    model: Model,
    dataloader: DataLoader,
    data_key: str,
    device: str = "cpu",
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Computes model predictions for a given dataloader and a model
    and returns a tuple of model predictions and targets
    """

    target, output = torch.empty(0), torch.empty(0)
    for batch in dataloader:
        images, responses = batch[:2] if not isinstance(batch, dict) else (batch["inputs"], batch["targets"])
        batch_kwargs = batch._asdict() if not isinstance(batch, dict) else batch

        with torch.no_grad():
            with device_state(model=model, device=device):
                output = torch.cat(
                    (
                        output,
                        (model(images.to(device), data_key=data_key, **batch_kwargs).detach().cpu()),
                    ),
                    dim=0,
                )
            target = torch.cat((target, responses.detach().cpu()), dim=0)

    return target.numpy(), output.numpy()
