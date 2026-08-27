# SPDX-License-Identifier: Apache-2.0
"""Single-GPU training and compact artifact storage for MiniOneRec RQ-VAE."""

from __future__ import annotations

import json
import os
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np
import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from transformers import get_constant_schedule_with_warmup

from minionerec.semantic_id.rqvae import (
    OFFICIAL_CODEBOOK_SIZES,
    OFFICIAL_HIDDEN_DIMS,
    OFFICIAL_LATENT_DIM,
    RQVAE,
)


MODEL_FILE_NAME = "rqvae_model.pth"
STATISTICS_FILE_NAME = "rqvae_training_stats.json"


@dataclass(frozen=True)
class RQVAETrainingConfig:
    """The fixed official training settings, with server runtime controls."""

    epochs: int = 10_000
    batch_size: int = 20_480
    num_workers: int = 4
    eval_step: int = 50
    learning_rate: float = 1e-3
    weight_decay: float = 0.0
    warmup_epochs: int = 50
    device: str = "cuda:0"
    seed: int = 2024
    input_dim: int = 2560
    codebook_sizes: tuple[int, ...] = OFFICIAL_CODEBOOK_SIZES
    latent_dim: int = OFFICIAL_LATENT_DIM
    hidden_dims: tuple[int, ...] = OFFICIAL_HIDDEN_DIMS
    dropout: float = 0.0
    batch_norm: bool = False
    quant_loss_weight: float = 1.0
    beta: float = 0.25
    kmeans_init: bool = True
    kmeans_iters: int = 100
    sinkhorn_epsilons: tuple[float, ...] = (0.0, 0.0, 0.0)
    sinkhorn_iters: int = 50

    def validate(self) -> None:
        if self.epochs < 1:
            raise ValueError("epochs must be at least 1")
        if self.batch_size < 1:
            raise ValueError("batch_size must be at least 1")
        if self.num_workers < 0:
            raise ValueError("num_workers cannot be negative")
        if self.eval_step < 1:
            raise ValueError("eval_step must be at least 1")
        if self.learning_rate <= 0:
            raise ValueError("learning_rate must be positive")
        if self.warmup_epochs < 0:
            raise ValueError("warmup_epochs cannot be negative")
        if len(self.codebook_sizes) != len(self.sinkhorn_epsilons):
            raise ValueError("one Sinkhorn epsilon is required for each codebook")


class EmbeddingDataset(Dataset[torch.Tensor]):
    """A validated float32 item-embedding matrix."""

    def __init__(
        self,
        embedding_file: Path,
        *,
        expected_items: int | None,
        expected_dim: int,
    ) -> None:
        embedding_file = Path(embedding_file)
        if not embedding_file.is_file():
            raise FileNotFoundError(f"embedding file does not exist: {embedding_file}")
        matrix = np.load(embedding_file, allow_pickle=False)
        if matrix.ndim != 2:
            raise ValueError("embedding matrix must have shape [items, dimension]")
        if expected_items is not None and matrix.shape[0] != expected_items:
            raise ValueError(
                f"expected {expected_items} items, got {matrix.shape[0]}"
            )
        if matrix.shape[1] != expected_dim:
            raise ValueError(
                f"expected embedding dimension {expected_dim}, got {matrix.shape[1]}"
            )
        if not np.issubdtype(matrix.dtype, np.floating):
            raise ValueError("embedding matrix must use a floating-point dtype")
        if not np.isfinite(matrix).all():
            raise ValueError("embedding matrix contains NaN or Inf")
        if np.any(np.all(matrix == 0, axis=1)):
            raise ValueError("embedding matrix contains an all-zero item row")
        self.tensor = torch.from_numpy(np.asarray(matrix, dtype=np.float32))
        self.source_dtype = str(matrix.dtype)

    def __len__(self) -> int:
        return self.tensor.shape[0]

    def __getitem__(self, index: int) -> torch.Tensor:
        return self.tensor[index]


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def _state_dict_to_cpu(model: torch.nn.Module) -> dict[str, torch.Tensor]:
    return {
        name: tensor.detach().cpu().clone()
        for name, tensor in model.state_dict().items()
    }


def _atomic_torch_save(value: Any, path: Path) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(value, temporary, pickle_protocol=4)
    os.replace(temporary, path)


def _atomic_json_save(value: dict[str, Any], path: Path) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(value, handle, indent=2, ensure_ascii=False, allow_nan=False)
    os.replace(temporary, path)


def _build_model(config: RQVAETrainingConfig) -> RQVAE:
    return RQVAE(
        config.input_dim,
        codebook_sizes=config.codebook_sizes,
        latent_dim=config.latent_dim,
        hidden_dims=config.hidden_dims,
        dropout=config.dropout,
        batch_norm=config.batch_norm,
        quant_loss_weight=config.quant_loss_weight,
        beta=config.beta,
        kmeans_init=config.kmeans_init,
        kmeans_iters=config.kmeans_iters,
        sinkhorn_epsilons=config.sinkhorn_epsilons,
        sinkhorn_iters=config.sinkhorn_iters,
    )


def _train_epoch(
    model: RQVAE,
    data_loader: DataLoader[torch.Tensor],
    optimizer: AdamW,
    scheduler: Any,
    device: torch.device,
) -> dict[str, float]:
    model.train()
    totals = {"total": 0.0, "reconstruction": 0.0, "quantization": 0.0}
    sample_count = 0
    started = perf_counter()
    for batch in data_loader:
        batch = batch.to(device, non_blocking=True)
        optimizer.zero_grad()
        reconstruction, quantization_loss, _ = model(batch, use_sinkhorn=True)
        total_loss, reconstruction_loss = model.compute_loss(
            reconstruction,
            quantization_loss,
            batch,
        )
        if not torch.isfinite(total_loss):
            raise FloatingPointError("training loss became NaN or Inf")
        total_loss.backward()
        optimizer.step()
        scheduler.step()

        batch_size = batch.shape[0]
        sample_count += batch_size
        totals["total"] += total_loss.item() * batch_size
        totals["reconstruction"] += reconstruction_loss.item() * batch_size
        totals["quantization"] += quantization_loss.item() * batch_size
    if sample_count == 0:
        raise RuntimeError("training data loader produced no batches")
    return {
        "total_loss": totals["total"] / sample_count,
        "reconstruction_loss": totals["reconstruction"] / sample_count,
        "quantization_loss": totals["quantization"] / sample_count,
        "training_seconds": perf_counter() - started,
    }


@torch.no_grad()
def evaluate_codes(
    model: RQVAE,
    data_loader: DataLoader[torch.Tensor],
    device: torch.device,
    codebook_sizes: tuple[int, ...],
) -> dict[str, Any]:
    """Measure collisions and per-layer codebook utilization over all items."""

    model.eval()
    started = perf_counter()
    batches: list[torch.Tensor] = []
    for batch in data_loader:
        indices = model.get_indices(
            batch.to(device, non_blocking=True),
            use_sinkhorn=False,
        )
        batches.append(indices.cpu())
    if not batches:
        raise RuntimeError("evaluation data loader produced no batches")
    all_indices = torch.cat(batches, dim=0)
    if all_indices.ndim != 2 or all_indices.shape[1] != len(codebook_sizes):
        raise ValueError("RQ-VAE returned an unexpected Semantic ID shape")

    layer_statistics: list[dict[str, Any]] = []
    for layer, codebook_size in enumerate(codebook_sizes):
        layer_indices = all_indices[:, layer]
        if layer_indices.min().item() < 0 or layer_indices.max().item() >= codebook_size:
            raise ValueError(f"layer {layer + 1} returned an out-of-range code")
        used_codes = torch.unique(layer_indices).numel()
        layer_statistics.append(
            {
                "layer": layer + 1,
                "used_codes": used_codes,
                "total_codes": codebook_size,
                "utilization": used_codes / codebook_size,
            }
        )

    item_count = all_indices.shape[0]
    unique_sid_count = torch.unique(all_indices, dim=0).shape[0]
    collision_count = item_count - unique_sid_count
    return {
        "item_count": item_count,
        "unique_sid_count": unique_sid_count,
        "collision_count": collision_count,
        "collision_rate": collision_count / item_count,
        "codebook_layers": layer_statistics,
        "evaluation_seconds": perf_counter() - started,
    }


def _initial_statistics(
    *,
    embedding_file: Path,
    dataset: EmbeddingDataset,
    config: RQVAETrainingConfig,
) -> dict[str, Any]:
    return {
        "embedding_file": str(embedding_file),
        "embedding_shape": list(dataset.tensor.shape),
        "embedding_dtype": dataset.source_dtype,
        "config": asdict(config),
        "records": [],
        "best_loss": {"value": None, "epoch": None},
        "best_collision": {"value": None, "epoch": None},
    }


def _load_resume_state(
    *,
    model_file: Path,
    statistics_file: Path,
    model: RQVAE,
    optimizer: AdamW,
    scheduler: Any,
) -> tuple[int, dict[str, Any], dict[str, Any]]:
    if not model_file.is_file() or not statistics_file.is_file():
        raise FileNotFoundError("resume requires both model and statistics files")
    checkpoint = torch.load(model_file, map_location="cpu", weights_only=False)
    last = checkpoint.get("last")
    if not isinstance(last, dict):
        raise ValueError("model file does not contain a last training state")
    model.load_state_dict(last["model_state_dict"])
    model.rq.mark_codebooks_initialized()
    optimizer.load_state_dict(last["optimizer_state_dict"])
    scheduler.load_state_dict(last["scheduler_state_dict"])
    with statistics_file.open("r", encoding="utf-8") as handle:
        statistics = json.load(handle)
    return int(last["epoch"]), checkpoint, statistics


def train_rqvae(
    *,
    embedding_file: Path,
    output_dir: Path,
    config: RQVAETrainingConfig,
    expected_items: int | None = 3686,
    resume: bool = False,
) -> dict[str, Any]:
    """Train RQ-VAE and maintain exactly one model and one statistics file."""

    config.validate()
    embedding_file = Path(embedding_file)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    model_file = output_dir / MODEL_FILE_NAME
    statistics_file = output_dir / STATISTICS_FILE_NAME
    if not resume and (model_file.exists() or statistics_file.exists()):
        raise FileExistsError(
            "output already exists; choose a new directory or pass resume=True"
        )
    if config.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError(f"CUDA device requested but unavailable: {config.device}")

    set_seed(config.seed)
    device = torch.device(config.device)
    dataset = EmbeddingDataset(
        embedding_file,
        expected_items=expected_items,
        expected_dim=config.input_dim,
    )
    pin_memory = device.type == "cuda"
    train_loader = DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        pin_memory=pin_memory,
        drop_last=False,
    )
    evaluation_loader = DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=pin_memory,
        drop_last=False,
    )
    model = _build_model(config).to(device)
    optimizer = AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    scheduler = get_constant_schedule_with_warmup(
        optimizer,
        num_warmup_steps=config.warmup_epochs * len(train_loader),
    )

    start_epoch = 0
    checkpoint: dict[str, Any] = {
        "format_version": 1,
        "config": asdict(config),
        "best_collision": None,
        "last": None,
    }
    statistics = _initial_statistics(
        embedding_file=embedding_file,
        dataset=dataset,
        config=config,
    )
    if resume:
        start_epoch, checkpoint, statistics = _load_resume_state(
            model_file=model_file,
            statistics_file=statistics_file,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
        )
    if start_epoch >= config.epochs:
        raise ValueError(
            f"checkpoint is already at epoch {start_epoch}, target is {config.epochs}"
        )

    best_loss_value = statistics["best_loss"]["value"]
    best_loss = float("inf") if best_loss_value is None else float(best_loss_value)
    best_collision_value = statistics["best_collision"]["value"]
    best_collision = (
        float("inf") if best_collision_value is None else float(best_collision_value)
    )

    for epoch in range(start_epoch + 1, config.epochs + 1):
        losses = _train_epoch(model, train_loader, optimizer, scheduler, device)
        if losses["total_loss"] < best_loss:
            best_loss = losses["total_loss"]
            statistics["best_loss"] = {"value": best_loss, "epoch": epoch}

        if epoch % config.eval_step != 0:
            continue

        evaluation = evaluate_codes(
            model,
            evaluation_loader,
            device,
            config.codebook_sizes,
        )
        record = {"epoch": epoch, **losses, **evaluation}
        statistics["records"].append(record)

        if evaluation["collision_rate"] < best_collision:
            best_collision = evaluation["collision_rate"]
            statistics["best_collision"] = {
                "value": best_collision,
                "epoch": epoch,
            }
            checkpoint["best_collision"] = {
                "epoch": epoch,
                "collision_rate": best_collision,
                "model_state_dict": _state_dict_to_cpu(model),
            }

        checkpoint["last"] = {
            "epoch": epoch,
            "model_state_dict": _state_dict_to_cpu(model),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
        }
        _atomic_torch_save(checkpoint, model_file)
        _atomic_json_save(statistics, statistics_file)
        print(
            f"epoch={epoch} total_loss={losses['total_loss']:.6f} "
            f"recon_loss={losses['reconstruction_loss']:.6f} "
            f"collision_rate={evaluation['collision_rate']:.6f} "
            f"used_codes={[item['used_codes'] for item in evaluation['codebook_layers']]}"
        )

    if checkpoint["best_collision"] is None or checkpoint["last"] is None:
        raise RuntimeError(
            "no evaluation was run; epochs must reach at least one eval_step"
        )
    return {
        "model_file": str(model_file),
        "statistics_file": str(statistics_file),
        "best_loss": statistics["best_loss"],
        "best_collision": statistics["best_collision"],
        "records": len(statistics["records"]),
    }
