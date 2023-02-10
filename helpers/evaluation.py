import torch
import math
import torch.utils.data

from running_mean_and_var import RunningMeanAndVar


@torch.no_grad()
def evaluate_classifier(
    model: torch.nn.Module, data_loader: torch.utils.data.DataLoader
):
    """Evaluate a model on a data loader.

    The caller is responsible for putting the model into .eval() mode if this is desired."""
    device = next(model.parameters()).device

    loss_var = RunningMeanAndVar()
    acc_var = RunningMeanAndVar()

    for data, target in data_loader:
        data, target = data.to(device), target.to(device)
        output = model(data)

        losses = torch.nn.functional.cross_entropy(output, target, reduction="none")
        predictions = output.argmax(dim=1, keepdim=True)
        correct = predictions.eq(target.view_as(predictions)).float()

        loss_var.add_many(losses)
        acc_var.add_many(correct)

    return {
        "cross_entropy": loss_var.empirical_mean_distribution,
        "accuracy": acc_var.empirical_mean_distribution,
    }
