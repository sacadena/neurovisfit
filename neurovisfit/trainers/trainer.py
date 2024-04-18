import warnings
from functools import partial
from typing import Any
from typing import Dict
from typing import Optional
from typing import Tuple

import numpy as np
import torch
from neuralpredictors.training import early_stopping
from neuralpredictors.training import LongCycler
from neuralpredictors.training import MultipleObjectiveTracker
from torch.utils.data import DataLoader
from tqdm import tqdm

from ..scorers.scores import ScorerNames
from .params import TrainerParams
from neurovisfit.common.random import set_random_seed
from neurovisfit.data.dataset import DataSplit
from neurovisfit.models.model import Model


def train_and_evaluate(
    model: Model,
    dataloaders: Dict[str, Any],
    params: TrainerParams,
    seed: Optional[int] = None,
    device: str = "cuda",
) -> Tuple[float, Dict[str, Any], Dict[str, Any]]:
    trainer = Trainer(model, dataloaders, params, seed, device)
    trainer.train(model, dataloaders)
    return trainer.evaluate(model, dataloaders)


class Trainer:
    def __init__(
        self,
        model: Model,
        dataloaders: Dict[str, Any],
        params: TrainerParams,
        seed: Optional[int] = None,
        device: str = "cuda",
    ) -> None:
        if device == "cuda" and not torch.cuda.is_available():
            warnings.warn(f"{device} not available. Switching to cpu")
            device = "cpu"

        self.params = params
        self.device = device
        model = model

        # Set model to device
        model.to(self.device)

        # Set seed
        self.seed = seed
        if self.seed is not None:
            set_random_seed(self.seed)

        # Get criterion to optimize
        self.criterion = self.params.loss_function.measure(
            avg=self.params.avg_loss,
            per_neuron=False,
        )

        # Get stop condition
        self.stop_closure = partial(
            self.params.stop_function.function,
            dataloaders=dataloaders["validation"],
            device=self.device,
            per_neuron=False,
            avg=True,
        )

        # Get optimizer and scheduler
        self.n_iterations = len(LongCycler(dataloaders[DataSplit.TRAIN.value]))
        self.optimizer = torch.optim.Adam(model.parameters(), lr=self.params.lr_init)
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode="max" if self.params.maximize else "min",
            factor=self.params.lr_decay_factor,
            patience=self.params.patience,
            threshold=self.params.tolerance,
            min_lr=self.params.min_lr,
            verbose=self.params.verbose,
            threshold_mode="abs",
        )

        # set the number of iterations over which you would like to accumulate gradients
        self.optim_step_count = (
            len(dataloaders[DataSplit.TRAIN.value].keys())
            if self.params.loss_accum_batch_n is None
            else self.params.loss_accum_batch_n
        )

        # Tracker
        self.tracker = self.get_tracker(model, dataloaders["validation"])

    def train(
        self,
        model: Model,
        dataloaders: Dict[str, Any],
        start_epoch: Optional[int] = None,
    ) -> None:
        model.train()
        print("Start")
        for epoch, val_obj in early_stopping(
            model,
            self.stop_closure,
            interval=self.params.interval,
            patience=self.params.patience,
            start=start_epoch or self.params.epoch,
            max_iter=self.params.max_iter,
            maximize=self.params.maximize,
            tolerance=self.params.tolerance,
            restore_best=self.params.restore_best,
            tracker=self.tracker,
            scheduler=self.scheduler,
            lr_decay_steps=self.params.lr_decay_steps,
        ):
            print(f"Epoch: {epoch}")

            # print the quantities from tracker
            if self.params.verbose and self.tracker is not None:
                print("=======================================")
                for key in self.tracker.log.keys():
                    print(key, self.tracker.log[key][-1], flush=True)

            # executes callback function if passed in keyword args
            if self.params.call_back_function is not None:
                self.params.call_back_function()

            # train over batches
            self.optimizer.zero_grad()
            for batch_no, (data_key, data) in tqdm(
                enumerate(LongCycler(dataloaders[DataSplit.TRAIN.value])),
                total=self.n_iterations,
                desc="Epoch {}".format(epoch),
            ):
                print("here")

                loss = self.full_objective(
                    model=model,
                    dataset_len=len(dataloaders[DataSplit.TRAIN.value][data_key].dataset),
                    input_tensor=data[0],
                    targets=data[1],
                    data_key=str(data_key),
                )
                loss.backward()
                if (batch_no + 1) % self.optim_step_count == 0:
                    self.optimizer.step()
                    self.optimizer.zero_grad()
                if (batch_no % self.params.batch_ping == 0) and (self.params.call_back_function is not None):
                    self.params.call_back_function()

    def evaluate(
        self,
        model: Model,
        dataloaders: Dict[str, Any],
    ) -> Tuple[float, Dict[str, Any], Dict[str, Any]]:
        model.eval()
        self.tracker.finalize() if self.params.track_training else None

        # Compute avg validation and test correlation
        validation_correlation = ScorerNames.CORRELATION.function(
            model,
            dataloaders["validation"],
            device=self.device,
            as_dict=False,
            per_neuron=False,
        )
        if self.params.return_test_score:
            test_correlation = ScorerNames.CORRELATION.function(
                model,
                dataloaders[DataSplit.TEST.value],
                device=self.device,
                as_dict=False,
                per_neuron=False,
            )
            score = np.mean(test_correlation)
        else:
            score = np.mean(validation_correlation)
        # return the whole tracker output as a dict
        output = self.tracker.log if self.params.track_training else {}
        output["validation_corr"] = validation_correlation

        return score, output, model.state_dict()

    def get_tracker(self, model: Model, validation_dataloader: Dict[str, DataLoader]) -> MultipleObjectiveTracker:
        if self.params.track_training is None:
            return None
        tracker_scores = dict(
            correlation=partial(
                ScorerNames.CORRELATION.function,
                model=model,
                dataloaders=validation_dataloader,
                device=self.device,
                per_neuron=False,
            ),
            poisson_loss=partial(
                ScorerNames.POISSON_LOSS.function,
                model=model,
                dataloaders=validation_dataloader,
                device=self.device,
                per_neuron=False,
                avg=False,
            ),
        )
        if hasattr(model, "tracked_values"):
            tracker_scores.update(model.tracked_values)
        return MultipleObjectiveTracker(**tracker_scores)

    def full_objective(
        self,
        model: Model,
        dataset_len: int,
        input_tensor: torch.Tensor,
        targets: torch.Tensor,
        data_key: str,
    ) -> torch.Tensor:
        loss_scale = np.sqrt(dataset_len / input_tensor.shape[0]) if self.params.scale_loss else 1.0
        return model.regularizer(data_key) + loss_scale * self.criterion(
            model(input_tensor.to(self.device), data_key=data_key),
            targets.to(self.device),
        )
