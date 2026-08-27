"""Temporal split helpers matching the official MiniOneRec data protocol."""

from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass
from typing import Generic, TypeVar


RecordT = TypeVar("RecordT")


@dataclass(frozen=True)
class TemporalSplits(Generic[RecordT]):
    """The official train/validation/test partitions."""

    train: tuple[RecordT, ...]
    valid: tuple[RecordT, ...]
    test: tuple[RecordT, ...]

    @property
    def total(self) -> int:
        return len(self.train) + len(self.valid) + len(self.test)


def global_target_time_split(
    records: Iterable[RecordT],
    *,
    target_timestamp: Callable[[RecordT], int],
) -> TemporalSplits[RecordT]:
    """Apply MiniOneRec's stable global target-time 80/10/10 split.

    This intentionally is not a per-user leave-one-out split. Python's sort is
    stable, so records with equal target timestamps retain their input order,
    matching the fixed upstream implementation.
    """

    ordered = tuple(sorted(records, key=lambda record: int(target_timestamp(record))))
    train_end = int(len(ordered) * 0.8)
    valid_end = int(len(ordered) * 0.9)
    return TemporalSplits(
        train=ordered[:train_end],
        valid=ordered[train_end:valid_end],
        test=ordered[valid_end:],
    )
