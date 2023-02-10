from typing import Optional

import torch
import torch.utils.data
from running_mean_and_var import RunningMeanAndVar


class DeterministicDataloaderException(ValueError):
    """Raised when evaluate_classifier is called with a non-shuffled dataloader."""

    pass


@torch.no_grad()
def evaluate_classifier(
    model: torch.nn.Module,
    data_loader: torch.utils.data.DataLoader,
    *,
    loss_tolerance: Optional[float] = None,
    accuracy_tolerance: Optional[float] = None,
):
    """Evaluate a model on a data loader.

    Quits if the standard deviation of the empirical mean estimate is lower then `loss_tolerance`
    or `accuracy_tolerance`.

    The caller is responsible for putting the model into .eval() mode if this is desired."""
    device = next(model.parameters()).device

    if not isinstance(data_loader.sampler, torch.utils.data.RandomSampler):
        raise DeterministicDataloaderException(
            "`evaluate_classifier` requires randomized loaders."
        )

    count = 0
    loss_var = RunningMeanAndVar()
    acc_var = RunningMeanAndVar()

    for data, target in data_loader:
        data, target = data.to(device), target.to(device)
        output = model(data)

        loss_per_example = torch.nn.functional.cross_entropy(
            output, target, reduction="none"
        )
        predictions = output.argmax(dim=1, keepdim=True)
        is_correct_mask = predictions.eq(target.view_as(predictions)).float()

        count += len(target)
        loss_var.add_many(loss_per_example)
        acc_var.add_many(is_correct_mask)

        if (
            loss_tolerance is not None
            and loss_var.variance_of_mean() < loss_tolerance**2
        ):
            break
        if (
            accuracy_tolerance is not None
            and acc_var.variance_of_mean() < accuracy_tolerance**2
        ):
            break

    return {
        "cross_entropy": loss_var.empirical_mean_distribution,
        "accuracy": acc_var.empirical_mean_distribution,
        "num_examples_evaluated": count,
    }
