import warnings
from enum import Enum
from typing import Any
from typing import Callable
from typing import Dict
from typing import Optional
from typing import Union

import numpy as np
from neuralpredictors.measures.np_functions import corr
from torch.utils.data import DataLoader

from .predictions import get_model_predictions_and_targets
from neurovisfit.data.dataset import DataSplit
from neurovisfit.models.model import Model


class ScorerNames(Enum):
    CORRELATION = "get_correlations"
    POISSON_LOSS = "get_poisson_loss"

    @property
    def function(self) -> Callable:
        return globals()[self.value]


def get_correlations(
    model: Model,
    dataloaders: Dict[str, Any],
    data_split: Optional[DataSplit] = None,
    device: str = "cpu",
    as_dict: bool = False,
    per_neuron: bool = True,
    **kwargs: Dict[str, Any],
) -> Union[np.ndarray, Dict[str, Any], float]:
    """
    Computes single-trial correlation between model prediction and true responses

    Args:
        model (torch.nn.Module): Model used to predict responses.
        dataloaders (dict): dict of test set torch dataloaders.
        data_split(str): the data_split (train/test/val). If data_split is None, then it is
            assumed that the data_split key is not present.
        device (str, optional): device to compute on. Defaults to "cpu".
        as_dict (bool, optional): whether to return the results per data_key. Defaults to False.
        per_neuron (bool, optional): whether to return the results per neuron or averaged across
            neurons. Defaults to True.

    Returns:
        dict or np.ndarray: contains the correlation values.
    """
    correlations = {}
    named_dataloaders = dataloaders[data_split.value] if data_split is not None else dataloaders
    for session, loader in named_dataloaders.items():
        target, output = get_model_predictions_and_targets(
            dataloader=loader, model=model, data_key=session, device=device
        )
        correlations[session] = corr(target, output, axis=0)

        if np.any(np.isnan(correlations[session])):
            warnings.warn("{}% NaNs , NaNs will be set to Zero.".format(np.isnan(correlations[session]).mean() * 100))
        correlations[session][np.isnan(correlations[session])] = 0

    if not as_dict:
        correlations = (
            np.hstack([v for v in correlations.values()])
            if per_neuron
            else np.mean(np.hstack([v for v in correlations.values()]))
        )
    return correlations


def get_poisson_loss(
    model: Model,
    dataloaders: Dict[str, DataLoader],
    device: str = "cpu",
    as_dict: bool = False,
    avg: bool = False,
    per_neuron: bool = True,
    eps: float = 1e-12,
) -> Union[Dict[str, Any], np.ndarray, float]:
    poisson_loss = {}
    for session, loader in dataloaders.items():
        target, output = get_model_predictions_and_targets(
            dataloader=loader, model=model, data_key=session, device=device
        )
        loss = output - target * np.log(output + eps)
        poisson_loss[session] = np.mean(loss, axis=0) if avg else np.sum(loss, axis=0)
        print(f"Session: {session}")
    print("=== Computed Full loss once ===")
    if as_dict:
        return poisson_loss

    if per_neuron:
        return np.hstack([v for v in poisson_loss.values()])

    return (
        np.mean(np.hstack([v for v in poisson_loss.values()]))
        if avg
        else np.sum(np.hstack([v for v in poisson_loss.values()]))
    )
