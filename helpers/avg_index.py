"""
An efficient way to compute the uniform average of a sequence of tensors
over different time windows.
"""

import pathlib
from copy import deepcopy
from dataclasses import dataclass
import uuid
from typing import Iterable, Optional, Union, Protocol

import torch
import pdb 
import os.path as osp

class AvgIndex(Protocol):
    available_checkpoints: set[int]

    def add(self, tensors: Iterable[torch.Tensor]) -> None:
        ...

    def store_checkpoint(self) -> None:
        ...

    def current_avg(self) -> list[torch.Tensor]:
        ...

    def avg_from(
        self, start: int, *, until: Optional[int] = None
    ) -> list[torch.Tensor]:
        ...

    def state_dict(self):
        ...

    def load_state_dict(
        self, state: Optional[dict] = None
    ):
        ...


class UniformAvgIndex(AvgIndex):
    def __init__(
        self,
        checkpoint_dir: Union[pathlib.Path, str],
        *,
        checkpoint_period: Optional[int] = None,
    ):
        self._available_checkpoints: set[int] = set()
        self._checkpoint_dir = pathlib.Path(checkpoint_dir)
        self._checkpoint_period = checkpoint_period
        self._counter: int = 0
        self._current_avg: Optional[list[torch.Tensor]] = None
        self._uuid: str = uuid.uuid4().hex

    @property
    def available_checkpoints(self) -> set[int]:
        checkpoints = set(self._available_checkpoints)
        checkpoints.add(self._counter)
        return checkpoints

    def add(self, tensors: Iterable[torch.Tensor]):
        """Add a new data point (e.g. model parameters) to the index."""
        if self._current_avg is None:
            self._counter = 1
            self._current_avg = _clone_tensors(tensors)
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
            self._checkpoint_dir / f"avg_{self._uuid}_{self._counter}.pt",
        )

    def current_avg(self) -> list[torch.Tensor]:
        if self._current_avg is None:
            raise RuntimeError("No data added yet.")

        return _clone_tensors(self._current_avg)

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

    def state_dict(self):
        return {
            "available_checkpoints": self._available_checkpoints,
            "checkpoint_dir": self._checkpoint_dir,
            "checkpoint_period": self._checkpoint_period,
            "counter": self._counter,
            "current_avg": self._current_avg,
            "uuid": self._uuid,
        }

    def load_state_dict(
        self, state: Optional[dict] = None
    ):
        self._available_checkpoints = state["available_checkpoints"]
        self._checkpoint_dir = state["checkpoint_dir"]
        self._checkpoint_period = state["checkpoint_period"]
        self._counter = state["counter"]
        self._current_avg = state["current_avg"]
        self._uuid = state["uuid"]

    def _load_checkpoint(self, step: int):
        assert step in self.available_checkpoints
        if step == self._counter:
            return self.current_avg()
        else:
            checkpoint = self._checkpoint_dir / f"avg_{self._uuid}_{step}.pt"
            return torch.load(checkpoint)


@dataclass(frozen=True)
class TAvgCheckpoint:
    uniform: list[torch.Tensor]
    t: list[torch.Tensor]


class TriangleAvgIndex(AvgIndex):
    def __init__(
        self,
        checkpoint_dir: Union[pathlib.Path, str],
        *,
        checkpoint_period: Optional[int] = None,
    ):
        self._available_checkpoints: set[int] = set()
        self._checkpoint_dir = pathlib.Path(checkpoint_dir)
        self._checkpoint_period = checkpoint_period
        self._counter: int = 0
        self._current_avg: Optional[list[torch.Tensor]] = None
        self._current_t_avg: Optional[list[torch.Tensor]] = None
        self._uuid: str = uuid.uuid4().hex

    @property
    def available_checkpoints(self) -> set[int]:
        checkpoints = set(self._available_checkpoints)
        checkpoints.add(self._counter)
        return checkpoints

    def add(self, tensors: Iterable[torch.Tensor]):
        """Add a new data point (e.g. model parameters) to the index."""
        if self._current_avg is None or self._current_t_avg is None:
            self._counter = 1
            self._current_avg = _clone_tensors(tensors)
            self._current_t_avg = _clone_tensors(tensors)
            return

        # Update uniform average
        self._counter += 1
        new_weight = 1.0 / self._counter
        torch._foreach_mul_(self._current_avg, 1 - new_weight)
        torch._foreach_add_(self._current_avg, list(tensors), alpha=new_weight)

        # Update t-average
        new_weight_t = 2.0 / (self._counter + 1)
        torch._foreach_mul_(self._current_t_avg, 1.0 - new_weight_t)
        torch._foreach_add_(self._current_t_avg, list(tensors), alpha=new_weight_t)
        if (
            self._checkpoint_period is not None
            and self._counter % self._checkpoint_period == 0
        ):
            self.store_checkpoint()

    def store_checkpoint(self):
        """Manually trigger the saving of a checkpoint."""
        self._available_checkpoints.add(self._counter)
        assert self._current_avg is not None
        assert self._current_t_avg is not None
        torch.save(
            TAvgCheckpoint(uniform=self._current_avg, t=self._current_t_avg),
            self._checkpoint_dir / f"avg_{self._uuid}_{self._counter}.pt",
        )

    def current_avg(self) -> list[torch.Tensor]:
        if self._current_t_avg is None:
            raise RuntimeError("No data added yet.")

        return _clone_tensors(self._current_t_avg)

    def avg_from(
        self, start: int, *, until: Optional[int] = None
    ) -> list[torch.Tensor]:
        """Uniform average between `start` and `until`."""
        if until is None:
            until = self._counter

        if start == 0:
            return self._load_checkpoint(until).t

        start_avg = self._load_checkpoint(start)
        until_avg = self._load_checkpoint(until)

        denom = (until - start) * (until - start + 1)

        window_avg = until_avg.t
        torch._foreach_mul_(window_avg, until * (until + 1) / denom)
        torch._foreach_add_(
            window_avg,
            start_avg.t,
            alpha=-start * (start + 1) / denom,
        )
        torch._foreach_add_(
            window_avg, until_avg.uniform, alpha=-start * until * 2 / denom
        )
        torch._foreach_add_(
            window_avg, start_avg.uniform, alpha=start * start * 2 / (denom)
        )

        return window_avg

    def state_dict(self):
        return {
            "available_checkpoints": self._available_checkpoints,
            "checkpoint_dir": self._checkpoint_dir,
            "checkpoint_period": self._checkpoint_period,
            "counter": self._counter,
            "current_t_avg": self._current_t_avg,
            "current_avg": self._current_avg,
            "uuid": self._uuid,
        }

    def load_state_dict(
        self, state: Optional[dict] = None
    ):
        self._available_checkpoints = state["available_checkpoints"]
        self._checkpoint_dir = state["checkpoint_dir"]
        self._checkpoint_period = state["checkpoint_period"]
        self._counter = state["counter"]
        self._current_avg = state["current_avg"]
        self._current_t_avg = state["current_t_avg"]
        self._uuid = state["uuid"]

    def _load_checkpoint(self, step: int) -> TAvgCheckpoint:
        assert step in self.available_checkpoints
        assert self._current_avg is not None
        assert self._current_t_avg is not None
        if step == self._counter:
            return TAvgCheckpoint(
                uniform=_clone_tensors(self._current_avg),
                t=_clone_tensors(self._current_t_avg),
            )
        else:
            checkpoint = self._checkpoint_dir / f"avg_{self._uuid}_{step}.pt"
            return torch.load(checkpoint)


def _clone_tensors(tensors: Iterable[torch.Tensor]) -> list[torch.Tensor]:
    return [tensor.detach().clone() for tensor in tensors]


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
