"""Data loading, training, and validation for the SASRec teacher."""

from __future__ import annotations

import ast
import csv
import json
import math
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from .sasrec import SASRec, SASRecConfig


@dataclass(frozen=True)
class SASRecTrainingConfig:
    """Inputs and optimization parameters for one teacher run."""

    train_file: Path
    valid_file: Path
    output_dir: Path
    num_items: int
    max_sequence_length: int = 10
    hidden_size: int = 32
    num_layers: int = 2
    num_heads: int = 2
    dropout: float = 0.3
    batch_size: int = 256
    eval_batch_size: int = 512
    num_epochs: int = 100
    learning_rate: float = 1e-3
    weight_decay: float = 0.0
    patience: int = 10
    seed: int = 42
    num_workers: int = 4
    device: str = "cuda:0"
    sample: int | None = None

    def validate(self) -> None:
        """Reject invalid and ambiguous run configurations."""

        if not self.train_file.is_file():
            raise FileNotFoundError(self.train_file)
        if not self.valid_file.is_file():
            raise FileNotFoundError(self.valid_file)
        if self.train_file.resolve() == self.valid_file.resolve():
            raise ValueError("train_file and valid_file must be different")
        if self.batch_size <= 0 or self.eval_batch_size <= 0:
            raise ValueError("batch sizes must be positive")
        if self.num_epochs <= 0:
            raise ValueError("num_epochs must be positive")
        if self.learning_rate <= 0.0:
            raise ValueError("learning_rate must be positive")
        if self.weight_decay < 0.0:
            raise ValueError("weight_decay cannot be negative")
        if self.patience <= 0:
            raise ValueError("patience must be positive")
        if self.num_workers < 0:
            raise ValueError("num_workers cannot be negative")
        if self.sample is not None and self.sample <= 0:
            raise ValueError("sample must be positive")


class _NextItemDataset(Dataset[tuple[torch.Tensor, torch.Tensor]]):
    """Fixed-width histories and next-item targets read from final CSV."""

    def __init__(
        self,
        path: Path,
        *,
        num_items: int,
        max_sequence_length: int,
        sample: int | None,
        seed: int,
    ) -> None:
        examples = _read_examples(
            path,
            num_items=num_items,
            max_sequence_length=max_sequence_length,
        )
        if sample is not None and sample < len(examples):
            generator = random.Random(seed)
            examples = generator.sample(examples, sample)
        self._histories = torch.stack([example[0] for example in examples])
        self._targets = torch.tensor(
            [example[1] for example in examples],
            dtype=torch.long,
        )

    def __len__(self) -> int:
        return self._targets.shape[0]

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self._histories[index], self._targets[index]


def _parse_history(value: str, *, row_number: int, path: Path) -> list[int]:
    try:
        parsed = ast.literal_eval(value)
    except (SyntaxError, ValueError) as error:
        raise ValueError(
            f"Invalid history_item_id at {path}:{row_number}"
        ) from error
    if not isinstance(parsed, list) or not parsed:
        raise ValueError(
            f"history_item_id must be a non-empty list at {path}:{row_number}"
        )
    if not all(isinstance(item_id, int) for item_id in parsed):
        raise ValueError(
            f"history_item_id must contain integers at {path}:{row_number}"
        )
    return parsed


def _read_examples(
    path: Path,
    *,
    num_items: int,
    max_sequence_length: int,
) -> list[tuple[torch.Tensor, int]]:
    examples: list[tuple[torch.Tensor, int]] = []
    with Path(path).open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        required = {"history_item_id", "item_id"}
        missing = required - set(reader.fieldnames or ())
        if missing:
            raise ValueError(f"Missing columns in {path}: {sorted(missing)}")
        for row_number, row in enumerate(reader, start=2):
            history = _parse_history(
                row["history_item_id"],
                row_number=row_number,
                path=path,
            )
            try:
                target = int(row["item_id"])
            except (TypeError, ValueError) as error:
                raise ValueError(
                    f"Invalid item_id at {path}:{row_number}"
                ) from error
            all_item_ids = history + [target]
            if min(all_item_ids) < 0 or max(all_item_ids) >= num_items:
                raise ValueError(
                    f"Item ID outside [0, {num_items - 1}] at "
                    f"{path}:{row_number}"
                )

            truncated = history[-max_sequence_length:]
            encoded = torch.zeros(max_sequence_length, dtype=torch.long)
            # Shift real zero-based item IDs by one; zero remains padding.
            encoded[: len(truncated)] = torch.tensor(truncated) + 1
            examples.append((encoded, target))
    if not examples:
        raise ValueError(f"No examples found in {path}")
    return examples


def _set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _ranking_metrics(
    logits: torch.Tensor,
    zero_based_targets: torch.Tensor,
    cutoffs: tuple[int, ...] = (1, 5, 10, 20),
) -> dict[str, float]:
    max_cutoff = min(max(cutoffs), logits.shape[1])
    top_indices = torch.topk(logits, k=max_cutoff, dim=1).indices
    matches = top_indices.eq(zero_based_targets.unsqueeze(1))
    result: dict[str, float] = {}
    for cutoff in cutoffs:
        effective_cutoff = min(cutoff, max_cutoff)
        selected = matches[:, :effective_cutoff]
        hit = selected.any(dim=1)
        result[f"HR@{cutoff}"] = hit.float().mean().item()

        positions = torch.arange(
            1,
            effective_cutoff + 1,
            device=logits.device,
            dtype=torch.float32,
        )
        discounts = 1.0 / torch.log2(positions + 1.0)
        ndcg = (selected.float() * discounts.unsqueeze(0)).sum(dim=1)
        result[f"NDCG@{cutoff}"] = ndcg.mean().item()
    return result


@torch.no_grad()
def _evaluate(
    model: SASRec,
    loader: DataLoader[tuple[torch.Tensor, torch.Tensor]],
    device: torch.device,
) -> dict[str, float]:
    model.eval()
    loss_function = nn.CrossEntropyLoss(reduction="sum")
    loss_sum = 0.0
    sample_count = 0
    metric_sums: dict[str, float] = {}
    for histories, targets in loader:
        histories = histories.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        logits = model(histories)
        loss_sum += loss_function(logits, targets).item()
        batch_size = targets.shape[0]
        sample_count += batch_size
        metrics = _ranking_metrics(logits, targets)
        for name, value in metrics.items():
            metric_sums[name] = metric_sums.get(name, 0.0) + value * batch_size
    return {
        "loss": loss_sum / sample_count,
        **{name: value / sample_count for name, value in metric_sums.items()},
    }


def _jsonable_config(config: SASRecTrainingConfig) -> dict[str, Any]:
    value = asdict(config)
    value["train_file"] = str(config.train_file)
    value["valid_file"] = str(config.valid_file)
    value["output_dir"] = str(config.output_dir)
    return value


def train_sasrec(config: SASRecTrainingConfig) -> dict[str, Any]:
    """Train one teacher and persist only its best validation checkpoint."""

    config.validate()
    model_file = config.output_dir / "sasrec_model.pth"
    statistics_file = config.output_dir / "training_stats.json"
    if model_file.exists() or statistics_file.exists():
        raise FileExistsError(
            f"Refusing to overwrite existing SASRec outputs in {config.output_dir}"
        )
    config.output_dir.mkdir(parents=True, exist_ok=True)
    _set_seed(config.seed)

    train_dataset = _NextItemDataset(
        config.train_file,
        num_items=config.num_items,
        max_sequence_length=config.max_sequence_length,
        sample=config.sample,
        seed=config.seed,
    )
    valid_dataset = _NextItemDataset(
        config.valid_file,
        num_items=config.num_items,
        max_sequence_length=config.max_sequence_length,
        sample=config.sample,
        seed=config.seed + 1,
    )
    pin_memory = config.device.startswith("cuda")
    generator = torch.Generator().manual_seed(config.seed)
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        pin_memory=pin_memory,
        generator=generator,
    )
    valid_loader = DataLoader(
        valid_dataset,
        batch_size=config.eval_batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=pin_memory,
    )

    device = torch.device(config.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    model_config = SASRecConfig(
        num_items=config.num_items,
        max_sequence_length=config.max_sequence_length,
        hidden_size=config.hidden_size,
        num_layers=config.num_layers,
        num_heads=config.num_heads,
        dropout=config.dropout,
    )
    model = SASRec(model_config).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    loss_function = nn.CrossEntropyLoss()

    best_ndcg = -math.inf
    best_epoch = 0
    best_metrics: dict[str, float] = {}
    epochs_without_improvement = 0
    records: list[dict[str, Any]] = []
    started_at = time.perf_counter()

    for epoch in range(1, config.num_epochs + 1):
        model.train()
        training_loss_sum = 0.0
        training_samples = 0
        for histories, targets in train_loader:
            histories = histories.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            logits = model(histories)
            loss = loss_function(logits, targets)
            loss.backward()
            optimizer.step()
            batch_size = targets.shape[0]
            training_loss_sum += loss.item() * batch_size
            training_samples += batch_size

        validation = _evaluate(model, valid_loader, device)
        record: dict[str, Any] = {
            "epoch": epoch,
            "train_loss": training_loss_sum / training_samples,
            "validation": validation,
        }
        records.append(record)
        print(json.dumps(record, ensure_ascii=False), flush=True)

        current_ndcg = validation["NDCG@10"]
        if current_ndcg > best_ndcg:
            best_ndcg = current_ndcg
            best_epoch = epoch
            best_metrics = validation
            epochs_without_improvement = 0
            torch.save(
                {
                    "model_config": asdict(model_config),
                    "model_state_dict": model.state_dict(),
                    "best_epoch": best_epoch,
                    "best_validation": best_metrics,
                    "item_id_offset": 1,
                },
                model_file,
            )
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= config.patience:
                break

    elapsed_seconds = time.perf_counter() - started_at
    statistics: dict[str, Any] = {
        "model_file": str(model_file),
        "statistics_file": str(statistics_file),
        "datasets": {
            "train": len(train_dataset),
            "validation": len(valid_dataset),
            "num_items": config.num_items,
        },
        "model": {
            **asdict(model_config),
            "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
            "item_id_offset": 1,
        },
        "training_parameters": _jsonable_config(config),
        "best": {
            "epoch": best_epoch,
            "validation": best_metrics,
        },
        "epochs_executed": len(records),
        "elapsed_seconds": round(elapsed_seconds, 3),
        "records": records,
    }
    if device.type == "cuda":
        statistics["cuda"] = {
            "logical_device": str(device),
            "name": torch.cuda.get_device_name(device),
            "peak_allocated_gib": round(
                torch.cuda.max_memory_allocated(device) / 1024**3,
                3,
            ),
        }
    with statistics_file.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(statistics, handle, indent=2, ensure_ascii=False)
    return statistics

