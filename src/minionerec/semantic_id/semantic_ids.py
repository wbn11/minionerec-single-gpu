# SPDX-License-Identifier: Apache-2.0
"""Generate MiniOneRec Semantic IDs and resolve collisions with Sinkhorn."""

from __future__ import annotations

import json
import os
from collections import Counter, defaultdict
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader

from minionerec.semantic_id.rqvae import RQVAE, VectorQuantizer
from minionerec.semantic_id.rqvae_training import EmbeddingDataset


OFFICIAL_TOKEN_PREFIXES = ("<a_{}>", "<b_{}>", "<c_{}>")
OFFICIAL_COLLISION_EPSILON = 0.003
OFFICIAL_MAX_COLLISION_ROUNDS = 20


def _load_best_collision_model(
    checkpoint_file: Path,
    device: torch.device,
) -> tuple[RQVAE, dict[str, Any], int, float]:
    checkpoint_file = Path(checkpoint_file)
    if not checkpoint_file.is_file():
        raise FileNotFoundError(f"checkpoint does not exist: {checkpoint_file}")
    checkpoint = torch.load(
        checkpoint_file,
        map_location="cpu",
        weights_only=False,
    )
    if not isinstance(checkpoint, dict):
        raise ValueError("checkpoint must contain a dictionary")
    config = checkpoint.get("config")
    best_collision = checkpoint.get("best_collision")
    if not isinstance(config, dict) or not isinstance(best_collision, dict):
        raise ValueError("checkpoint does not contain config and best_collision")
    model_state = best_collision.get("model_state_dict")
    if not isinstance(model_state, dict):
        raise ValueError("best_collision does not contain model weights")

    required_config = (
        "input_dim",
        "codebook_sizes",
        "latent_dim",
        "hidden_dims",
        "dropout",
        "batch_norm",
        "quant_loss_weight",
        "beta",
        "kmeans_init",
        "kmeans_iters",
        "sinkhorn_epsilons",
        "sinkhorn_iters",
    )
    missing = [name for name in required_config if name not in config]
    if missing:
        raise ValueError(f"checkpoint config is missing: {', '.join(missing)}")

    model = RQVAE(
        int(config["input_dim"]),
        codebook_sizes=tuple(config["codebook_sizes"]),
        latent_dim=int(config["latent_dim"]),
        hidden_dims=tuple(config["hidden_dims"]),
        dropout=float(config["dropout"]),
        batch_norm=bool(config["batch_norm"]),
        quant_loss_weight=float(config["quant_loss_weight"]),
        beta=float(config["beta"]),
        kmeans_init=bool(config["kmeans_init"]),
        kmeans_iters=int(config["kmeans_iters"]),
        sinkhorn_epsilons=tuple(config["sinkhorn_epsilons"]),
        sinkhorn_iters=int(config["sinkhorn_iters"]),
    )
    model.load_state_dict(model_state)
    model.rq.mark_codebooks_initialized()
    model.to(device)
    model.eval()
    epoch = int(best_collision["epoch"])
    collision_rate = float(best_collision["collision_rate"])
    return model, config, epoch, collision_rate


@torch.no_grad()
def _generate_initial_indices(
    model: RQVAE,
    data_loader: DataLoader[torch.Tensor],
    device: torch.device,
) -> torch.Tensor:
    batches: list[torch.Tensor] = []
    for batch in data_loader:
        indices = model.get_indices(
            batch.to(device, non_blocking=True),
            use_sinkhorn=False,
        )
        batches.append(indices.cpu())
    if not batches:
        raise RuntimeError("embedding data loader produced no batches")
    return torch.cat(batches, dim=0)


def _collision_groups(indices: torch.Tensor) -> list[list[int]]:
    index_groups: defaultdict[tuple[int, ...], list[int]] = defaultdict(list)
    for item_id, row in enumerate(indices.tolist()):
        index_groups[tuple(int(code) for code in row)].append(item_id)
    return [group for group in index_groups.values() if len(group) > 1]


def sid_statistics(
    indices: torch.Tensor,
    codebook_sizes: Sequence[int],
) -> dict[str, Any]:
    """Describe exact SID collisions and per-layer codebook use."""

    if indices.ndim != 2 or indices.shape[1] != len(codebook_sizes):
        raise ValueError("Semantic IDs have an unexpected shape")
    if indices.shape[0] == 0:
        raise ValueError("Semantic ID matrix is empty")

    rows = [tuple(int(code) for code in row) for row in indices.tolist()]
    counts = Counter(rows)
    collision_sizes = [count for count in counts.values() if count > 1]
    item_count = len(rows)
    unique_sid_count = len(counts)
    collision_count = item_count - unique_sid_count

    used_codes: list[int] = []
    for layer, codebook_size in enumerate(codebook_sizes):
        layer_indices = indices[:, layer]
        minimum = int(layer_indices.min().item())
        maximum = int(layer_indices.max().item())
        if minimum < 0 or maximum >= codebook_size:
            raise ValueError(f"layer {layer + 1} contains an out-of-range code")
        used_codes.append(int(torch.unique(layer_indices).numel()))

    return {
        "item_count": item_count,
        "unique_sid_count": unique_sid_count,
        "collision_count": collision_count,
        "collision_rate": collision_count / item_count,
        "collision_group_count": len(collision_sizes),
        "max_collision_group_size": max(counts.values()),
        "used_codes": used_codes,
    }


def _configure_official_collision_sinkhorn(
    model: RQVAE,
    epsilon: float,
) -> None:
    if len(model.rq.vq_layers) != 3:
        raise ValueError("official Semantic IDs require exactly three VQ layers")
    for layer_index, quantizer in enumerate(model.rq.vq_layers):
        if not isinstance(quantizer, VectorQuantizer):
            raise TypeError("vq_layers must contain VectorQuantizer instances")
        quantizer.sinkhorn_epsilon = epsilon if layer_index == 2 else 0.0


@torch.no_grad()
def resolve_collisions(
    *,
    model: RQVAE,
    embeddings: torch.Tensor,
    initial_indices: torch.Tensor,
    device: torch.device,
    codebook_sizes: tuple[int, ...],
    epsilon: float = OFFICIAL_COLLISION_EPSILON,
    max_rounds: int = OFFICIAL_MAX_COLLISION_ROUNDS,
) -> tuple[torch.Tensor, list[dict[str, Any]]]:
    """Reassign only colliding groups using third-level balanced assignment."""

    if epsilon <= 0:
        raise ValueError("Sinkhorn epsilon must be positive")
    if max_rounds < 1:
        raise ValueError("max_rounds must be at least 1")
    _configure_official_collision_sinkhorn(model, epsilon)
    indices = initial_indices.clone()
    rounds: list[dict[str, Any]] = []

    for round_number in range(1, max_rounds + 1):
        groups = _collision_groups(indices)
        if not groups:
            break
        for item_ids in groups:
            item_tensor = torch.tensor(item_ids, dtype=torch.long)
            group_embeddings = embeddings.index_select(0, item_tensor).to(device)
            reassigned = model.get_indices(
                group_embeddings,
                use_sinkhorn=True,
            )
            indices.index_copy_(0, item_tensor, reassigned.cpu())

        statistics = sid_statistics(indices, codebook_sizes)
        round_statistics = {"round": round_number, **statistics}
        rounds.append(round_statistics)
        print(
            f"round={round_number} "
            f"collision_rate={statistics['collision_rate']:.6f} "
            f"collision_groups={statistics['collision_group_count']} "
            f"max_group_size={statistics['max_collision_group_size']}"
        )
        if statistics["collision_count"] == 0:
            break
    return indices, rounds


def _semantic_id_catalog(indices: torch.Tensor) -> dict[str, list[str]]:
    if indices.ndim != 2 or indices.shape[1] != len(OFFICIAL_TOKEN_PREFIXES):
        raise ValueError("official Semantic IDs require exactly three codes")
    catalog: dict[str, list[str]] = {}
    for item_id, row in enumerate(indices.tolist()):
        catalog[str(item_id)] = [
            prefix.format(int(code))
            for prefix, code in zip(OFFICIAL_TOKEN_PREFIXES, row)
        ]
    return catalog


def _ensure_new_outputs(output_file: Path, statistics_file: Path) -> None:
    if output_file.resolve() == statistics_file.resolve():
        raise ValueError("index and statistics outputs must be different files")
    existing = [path for path in (output_file, statistics_file) if path.exists()]
    if existing:
        joined = ", ".join(str(path) for path in existing)
        raise FileExistsError(f"refusing to overwrite existing output: {joined}")
    for path in (output_file, statistics_file):
        if not path.parent.is_dir():
            raise FileNotFoundError(f"output directory does not exist: {path.parent}")


def _write_json(value: Any, path: Path) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(value, handle, indent=2, ensure_ascii=False, allow_nan=False)
    os.replace(temporary, path)


def generate_semantic_ids(
    *,
    embedding_file: Path,
    checkpoint_file: Path,
    output_file: Path,
    statistics_file: Path,
    expected_items: int = 3686,
    batch_size: int = 64,
    device: str = "cuda:0",
    epsilon: float = OFFICIAL_COLLISION_EPSILON,
    max_rounds: int = OFFICIAL_MAX_COLLISION_ROUNDS,
) -> dict[str, Any]:
    """Generate official three-token SIDs and save their collision statistics."""

    if batch_size < 1:
        raise ValueError("batch_size must be at least 1")
    if expected_items < 1:
        raise ValueError("expected_items must be at least 1")
    output_file = Path(output_file)
    if output_file.suffix != ".json":
        raise ValueError("output_file must end in .json")
    statistics_file = Path(statistics_file)
    _ensure_new_outputs(output_file, statistics_file)
    if device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError(f"CUDA device requested but unavailable: {device}")

    torch_device = torch.device(device)
    model, config, checkpoint_epoch, checkpoint_collision_rate = (
        _load_best_collision_model(Path(checkpoint_file), torch_device)
    )
    codebook_sizes = tuple(int(size) for size in config["codebook_sizes"])
    dataset = EmbeddingDataset(
        Path(embedding_file),
        expected_items=expected_items,
        expected_dim=int(config["input_dim"]),
    )
    data_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=torch_device.type == "cuda",
        drop_last=False,
    )
    initial_indices = _generate_initial_indices(model, data_loader, torch_device)
    before = sid_statistics(initial_indices, codebook_sizes)
    print(
        f"checkpoint_epoch={checkpoint_epoch} "
        f"initial_collision_rate={before['collision_rate']:.6f} "
        f"collision_groups={before['collision_group_count']} "
        f"max_group_size={before['max_collision_group_size']}"
    )

    final_indices, rounds = resolve_collisions(
        model=model,
        embeddings=dataset.tensor,
        initial_indices=initial_indices,
        device=torch_device,
        codebook_sizes=codebook_sizes,
        epsilon=epsilon,
        max_rounds=max_rounds,
    )
    after = sid_statistics(final_indices, codebook_sizes)
    catalog = _semantic_id_catalog(final_indices)
    if len(catalog) != expected_items:
        raise ValueError("Semantic ID catalog does not contain every expected item")

    statistics: dict[str, Any] = {
        "checkpoint_file": str(Path(checkpoint_file)),
        "checkpoint_epoch": checkpoint_epoch,
        "checkpoint_collision_rate": checkpoint_collision_rate,
        "item_count": expected_items,
        "sinkhorn_epsilon": epsilon,
        "sinkhorn_iterations": int(config["sinkhorn_iters"]),
        "max_rounds": max_rounds,
        "rounds_executed": len(rounds),
        "before": before,
        "rounds": rounds,
        "after": after,
        "index_file": str(output_file),
    }
    _write_json(catalog, output_file)
    _write_json(statistics, statistics_file)
    return {
        "index_file": str(output_file),
        "statistics_file": str(statistics_file),
        "checkpoint_epoch": checkpoint_epoch,
        "before": before,
        "after": after,
        "rounds_executed": len(rounds),
    }
