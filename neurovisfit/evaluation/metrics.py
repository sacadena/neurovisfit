import warnings
from typing import Any
from typing import Dict
from typing import List
from typing import Optional
from typing import Tuple
from typing import Union

import numpy as np
import torch
from neuralpredictors.measures import corr

from ..models.model import Model
from .predictions import compute_model_predictions_for_repeated_input_loaders


def compute_feve_from_target_and_outputs(
    targets: Union[torch.Tensor, np.ndarray, List[np.ndarray]],
    outputs: Union[torch.Tensor, np.ndarray, List[np.ndarray]],
    return_fraction_explainable_var: bool = False,
) -> Union[Tuple[np.ndarray, np.ndarray], np.ndarray]:
    """

    Args:
        targets (list): Neuronal responses (ground truth) to image repeats.
            Dimensions: [num_images] np.array(num_reaps, num_neurons)
        outputs (list): Model predictions to the repeated images, with an identical shape as the targets
        return_fraction_explainable_var (bool): returns the fraction of explainable variance per neuron
            if set to True

    Returns:
        FEVe (np.array): the fraction of explainable variance explained per neuron
        --- optional: FEV (np.array): the fraction

    """
    image_variance = []
    predictions_variance = []

    for i, _ in enumerate(targets):
        predictions_variance.append((targets[i] - outputs[i]) ** 2)
        image_variance.append(np.var(targets[i], axis=0, ddof=1))

    predictions_variance = np.vstack(predictions_variance)
    image_variance = np.vstack(image_variance)

    total_variance = np.var(np.vstack(targets), axis=0, ddof=1)
    noise_variance = np.mean(image_variance, axis=0)
    fraction_explainable_variance = (total_variance - noise_variance) / total_variance

    predictions_variance_averaged = np.mean(predictions_variance, axis=0)
    fraction_explainable_variance_explained = 1 - (predictions_variance_averaged - noise_variance) / (
        total_variance - noise_variance
    )
    if return_fraction_explainable_var:
        return fraction_explainable_variance, fraction_explainable_variance_explained
    return fraction_explainable_variance_explained


def compute_correlation_to_average_signal(
    model: Model,
    dataloaders: Dict[str, Any],
    device: str = "cpu",
    as_dict: bool = False,
    per_neuron: bool = True,
    min_fraction_explainable_variance: Optional[float] = None,
) -> Union[Dict[str, Any], np.ndarray, float]:
    if "test" in dataloaders:
        dataloaders = dataloaders["test"]

    correlations = {}
    for data_key, dataloader in dataloaders.items():
        # Get targets and outputs
        targets, outputs = compute_model_predictions_for_repeated_input_loaders(
            dataloader=dataloader,
            model=model,
            data_key=data_key,
            device=device,
            broadcast_to_target=True,
        )

        # Get correlation to avg. targets
        outputs_avg = np.array([out.mean(axis=0) for out in outputs])
        targets_avg = np.array([t.mean(axis=0) for t in targets])
        raw_correlations = corr(targets_avg, outputs_avg, axis=0)

        if min_fraction_explainable_variance is None:
            correlations[data_key] = raw_correlations
        else:
            exp_var_frac, _ = compute_feve_from_target_and_outputs(
                targets=targets,
                outputs=outputs,
                return_fraction_explainable_var=True,
            )
            correlations[data_key] = raw_correlations[exp_var_frac > min_fraction_explainable_variance]

        # Check for nans
        if np.any(np.isnan(correlations[data_key])):
            warnings.warn("{}% NaNs , NaNs will be set to Zero.".format(np.isnan(correlations[data_key]).mean() * 100))
        correlations[data_key][np.isnan(correlations[data_key])] = 0

    if not as_dict:
        correlations_arr = np.hstack(list(correlations.values()))
        correlations = correlations_arr if per_neuron else np.mean(correlations_arr)

    return correlations


def compute_feve(
    model: Model,
    dataloaders: Dict[str, Any],
    as_dict: bool = False,
    device: str = "cuda",
    per_neuron: bool = True,
    min_fraction_explainable_variance: Optional[float] = None,
) -> Union[Dict[str, Any], np.ndarray, float]:
    dataloaders = dataloaders["test"] if "test" in dataloaders else dataloaders
    feve_sessions = {}
    for data_key, dataloader in dataloaders.items():
        targets, outputs = compute_model_predictions_for_repeated_input_loaders(
            model=model,
            dataloader=dataloader,
            data_key=data_key,
            device=device,
            broadcast_to_target=True,
        )
        if min_fraction_explainable_variance is None:
            feve_sessions[data_key] = compute_feve_from_target_and_outputs(targets=targets, outputs=outputs)
        else:
            fev, feve = compute_feve_from_target_and_outputs(
                targets=targets, outputs=outputs, return_fraction_explainable_var=True
            )
            feve_sessions[data_key] = feve[fev > min_fraction_explainable_variance]

    if not as_dict:
        feve_sessions_arr = np.hstack([v for v in feve_sessions.values()])
        feve_sessions = feve_sessions_arr if per_neuron else np.mean(feve_sessions_arr)
    return feve_sessions
