from typing import List
from typing import Optional
from typing import Tuple

import numpy as np
import torch
from neuralpredictors.training import device_state
from neuralpredictors.training import eval_state
from torch.utils.data import DataLoader

from ..models.model import Model


def compute_model_predictions_for_repeated_input_loaders(
    model: Model,
    dataloader: DataLoader,
    data_key: str,
    device: str = "cuda",
    broadcast_to_target: bool = False,
    repeat_channel_dim: Optional[int] = None,
) -> Tuple[List[np.ndarray], np.ndarray]:
    """
    Computes model predictions for a dataloader that yields batches with identical inputs along the first dimension.
    Unique inputs will be forwarded only once through the model
    Returns:
        target: ground truth, i.e. neuronal firing rates of the neurons as a list: [num_images][num_reaps, num_neurons]
        output: responses as predicted by the network for the unique images. If broadcast_to_target, returns repeated
                outputs of shape [num_images][num_reaps, num_neurons] else (default) returns unique outputs of shape
                [num_images, num_neurons]
    """
    target, output = [], []
    unique_images = torch.empty(0).to(device)
    for batch in dataloader:
        images, responses = batch[:2]

        if len(images.shape) == 5:
            images = images.squeeze(dim=0)
            responses = responses.squeeze(dim=0)

        first_image = images[0, :1, ...]
        for im in images:
            if not torch.equal(im, first_image):
                raise ValueError("expected all images in the batch to be equal")

        unique_images = torch.cat(
            tensors=(unique_images, images[0:1, ...].to(device)),
            dim=0,
        )
        target.append(responses.detach().cpu().numpy())

        if images.shape[0] > 1:
            with eval_state(model):
                with device_state(model, device=device):
                    output.append(
                        model(
                            images.to(device),
                            data_key=data_key,
                            **batch._asdict(),
                            repeat_channel_dim=repeat_channel_dim,
                        )
                        .detach()
                        .cpu()
                        .numpy()
                    )

    # Forward unique images once
    if len(output) == 0:
        with eval_state(model):
            with device_state(model, device):
                output = (
                    model(unique_images.to(device), data_key=data_key, repeat_channel_dim=repeat_channel_dim)
                    .detach()
                    .cpu()
                    .numpy()
                )

    if broadcast_to_target:
        output = [np.broadcast_to(x, target[idx].shape) for idx, x in enumerate(output)]

    return target, output
