"""
An efficient way to compute the uniform average of a sequence of tensors
over different time windows.
"""

from copy import deepcopy
import pathlib
from typing import Iterable, Optional, Union

import torch


class UniformAvgIndex:
    def __init__(
        self,
        checkpoint_dir: Union[pathlib.Path, str],
        *,
        checkpoint_period: Optional[int] = None,
    ):
        self._checkpoint_dir = pathlib.Path(checkpoint_dir)
        self._checkpoint_period = checkpoint_period
        self._current_avg: Optional[list[torch.Tensor]] = None
        self._counter: int = 0
        self._available_checkpoints: set[int] = set()

    @property
    def available_checkpoints(self) -> set[int]:
        return set(self._available_checkpoints)

    def add(self, tensors: Iterable[torch.Tensor]):
        """Add a new data point (e.g. model parameters) to the index."""
        if self._current_avg is None:
            self._counter = 1
            self._current_avg = [t.detach().clone() for t in tensors]
            return

        self._counter += 1
        new_weight = 1.0 / self._counter
        torch._foreach_mul_(self._current_avg, 1 - new_weight)
        torch._foreach_add_(self._current_avg, list(tensors), alpha=new_weight)

        if (
            self._checkpoint_period is not None
            and self._counter % self._checkpoint_period == 0
        ):
            self.store_checkpoint()

    def store_checkpoint(self):
        """Manually trigger the saving of a checkpoint."""
        self._available_checkpoints.add(self._counter)
        torch.save(
            self._current_avg,
            self._checkpoint_dir / f"avg_{self._counter}.pt",
        )

    def current_avg(self) -> list[torch.Tensor]:
        if self._current_avg is None:
            raise RuntimeError("No data added yet.")

        return [tensor.clone() for tensor in self._current_avg]

    def avg_from(
        self, start: int, *, until: Optional[int] = None
    ) -> list[torch.Tensor]:
        """Uniform average between `start` and `until`."""
        if until is None:
            until = self._counter

        if start == 0:
            return self._load_checkpoint(until)

        start_avg = self._load_checkpoint(start)
        until_avg = self._load_checkpoint(until)

        window_avg = start_avg
        torch._foreach_mul_(window_avg, -start / (until - start))
        torch._foreach_add_(window_avg, until_avg, alpha=until / (until - start))

        return window_avg

    def _load_checkpoint(self, step: int):
        if step == self._counter:
            return self.current_avg()
        else:
            assert step in self._available_checkpoints
            checkpoint = self._checkpoint_dir / f"avg_{step}.pt"
            return torch.load(checkpoint)


class ModelAvgIndex:
    def __init__(
        self,
        module: torch.nn.Module,
        index: UniformAvgIndex,
        *,
        include_buffers: bool = True,
    ):
        self._index = index
        self._module = module
        self._include_buffers = include_buffers

        self._tensors_to_avg: list[torch.Tensor] = list(module.parameters())
        self._num_params = len(self._tensors_to_avg)
        if include_buffers:
            self._tensors_to_avg.extend(list(module.buffers()))

    def record_step(self):
        self._index.add(tensor.float() for tensor in self._tensors_to_avg)

    def avg_from(self, start: int, *, until: Optional[int] = None) -> torch.nn.Module:
        avg_data = self._index.avg_from(start, until=until)

        original_device = next(self._module.parameters()).device
        avg_module = deepcopy(self._module).to(original_device)

        avg_params = avg_data[: self._num_params]
        for param, avg in zip(avg_module.parameters(), avg_params):
            param.data = avg

        if self._include_buffers:
            avg_buffers = avg_data[self._num_params :]
            for buffer, avg in zip(avg_module.buffers(), avg_buffers):
                buffer.data = avg

        return avg_module
