import torch
import math
from typing import Union


class RunningMeanAndVar:
    """Online mean and variance of a sequence of numbers, using Welford's method."""

    def __init__(self):
        self._count: int = 0
        self._mean: float = 0.0
        self._M2: float = 0.0

    def add(self, value: Union[float, torch.Tensor]):
        if isinstance(value, torch.Tensor):
            value = value.item()

            self._count += 1
            delta = value - self._mean
            self._mean += delta / self._count
            delta2 = value - self._mean
            self._M2 += delta * delta2

    def add_many(self, tensor: torch.Tensor):
        for value in tensor.view(-1):
            self.add(value)

    @property
    def mean(self):
        return self._mean

    def sample_variance(self, *, unbiased: bool = False) -> float:
        if self._count < 2:
            return float("nan")

        if unbiased:
            return self._M2 / (self._count - 1)
        else:
            return self._M2 / self._count

    def variance_of_mean(self, *, unbiased: bool = True) -> float:
        return self.sample_variance(unbiased=unbiased) / self._count

    @property
    def empirical_mean_distribution(self) -> torch.distributions.Normal:
        return torch.distributions.Normal(
            loc=self.mean, scale=math.sqrt(self.variance_of_mean())
        )
