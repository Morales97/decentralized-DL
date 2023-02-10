import pytest
import torch
import torch.utils.data
import torchvision.datasets
import torchvision.models
import torchvision.transforms

import evaluation


def test_loading(model, test_loader):
    out = evaluation.evaluate_classifier(model, test_loader)
    assert "cross_entropy" in out
    assert "accuracy" in out
    assert isinstance(out["cross_entropy"], torch.distributions.Distribution)
    assert isinstance(out["accuracy"], torch.distributions.Distribution)


@pytest.fixture
def model() -> torch.nn.Module:
    return torchvision.models.mobilenet_v3_small(num_classes=10)


@pytest.fixture
def test_loader():
    data_mean = (0.4914, 0.4822, 0.4465)
    data_stddev = (0.2023, 0.1994, 0.2010)
    transform = torchvision.transforms.Compose(
        [
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(data_mean, data_stddev),
        ]
    )
    dataset = torchvision.datasets.CIFAR10(
        root="data",
        train=False,
        download=True,
        transform=transform,
    )
    return torch.utils.data.DataLoader(
        dataset,
        batch_size=1000,
        shuffle=True,
        drop_last=False,
        num_workers=2,
    )
