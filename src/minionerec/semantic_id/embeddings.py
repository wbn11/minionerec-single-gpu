# SPDX-License-Identifier: Apache-2.0
"""Official-compatible Amazon item text embeddings for MiniOneRec."""

from __future__ import annotations

import hashlib
import html
import json
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from tqdm.auto import tqdm
from transformers import AutoModel, AutoTokenizer


OFFICIAL_UPSTREAM_COMMIT = "0c64b955ecb8e3d7a9ae9f1fa88cf938f129b0ed"
OFFICIAL_MAX_LENGTH = 2048
OFFICIAL_FEATURES = ("title", "description")
QWEN3_EMBEDDING_4B_HIDDEN_SIZE = 2560


@dataclass(frozen=True)
class ItemText:
    """One item and the official title-plus-description encoder input."""

    item_id: int
    text: str


@dataclass(frozen=True)
class EmbeddingArtifactPaths:
    """The embedding matrix and its reproducibility manifest."""

    embeddings: Path
    manifest: Path

    @property
    def files(self) -> tuple[Path, Path]:
        return (self.embeddings, self.manifest)


def official_clean_text(raw_text: Any) -> str:
    """Match ``rq/text2emb/utils.py::clean_text`` at the fixed commit."""

    if isinstance(raw_text, list):
        parts: list[str] = []
        for raw in raw_text:
            cleaned = html.unescape(str(raw))
            cleaned = re.sub(r"</?\w+[^>]*>", "", cleaned)
            cleaned = re.sub(r'["\n\r]*', "", cleaned)
            parts.append(cleaned.strip())
        cleaned_text = " ".join(parts)
    else:
        if isinstance(raw_text, dict):
            cleaned_text = str(raw_text)[1:-1].strip()
        else:
            cleaned_text = str(raw_text).strip()
        cleaned_text = html.unescape(cleaned_text)
        cleaned_text = re.sub(r"</?\w+[^>]*>", "", cleaned_text)
        cleaned_text = re.sub(r'["\n\r]*', "", cleaned_text)

    index = -1
    while -index < len(cleaned_text) and cleaned_text[index] == ".":
        index -= 1
    index += 1
    if index == 0:
        cleaned_text = cleaned_text + "."
    else:
        cleaned_text = cleaned_text[:index] + "."
    if len(cleaned_text) >= 2000:
        cleaned_text = ""
    return cleaned_text


def build_item_text(item_id: int, features: Mapping[str, Any]) -> ItemText:
    """Join the fixed upstream title and description feature sequence."""

    parts: list[str] = []
    for feature_name in OFFICIAL_FEATURES:
        if feature_name in features:
            cleaned = official_clean_text(features[feature_name]).strip()
            if cleaned:
                parts.append(cleaned)
    return ItemText(item_id=item_id, text=" ".join(parts) or "unknown item")


def _load_json_object(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"JSON file does not exist: {path}")
    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, dict):
        raise ValueError(f"Expected a JSON object in {path}")
    return data


def load_item_mapping_ids(path: Path) -> tuple[int, ...]:
    """Read and validate the integer IDs in an official ``item2id`` file."""

    if not path.is_file():
        raise FileNotFoundError(f"item2id file does not exist: {path}")
    ids: list[int] = []
    original_items: set[str] = set()
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            fields = line.rstrip("\n\r").split("\t")
            if len(fields) != 2 or not fields[0]:
                raise ValueError(f"Invalid item2id row at {path}:{line_number}")
            original, raw_item_id = fields
            if original in original_items:
                raise ValueError(f"Duplicate original item at {path}:{line_number}")
            original_items.add(original)
            try:
                ids.append(int(raw_item_id))
            except ValueError as error:
                raise ValueError(
                    f"Non-integer item ID at {path}:{line_number}"
                ) from error
    _require_contiguous_ids(ids, source=str(path))
    return tuple(sorted(ids))


def _require_contiguous_ids(ids: Sequence[int], *, source: str) -> None:
    if len(set(ids)) != len(ids):
        raise ValueError(f"Duplicate item IDs in {source}")
    expected = list(range(len(ids)))
    if sorted(ids) != expected:
        raise ValueError(f"Item IDs in {source} must be contiguous from 0")


def load_item_texts(
    item_file: Path, item_mapping_file: Path | None = None
) -> tuple[ItemText, ...]:
    """Load item text in item-ID order and optionally cross-check ``item2id``."""

    item_features = _load_json_object(Path(item_file))
    parsed: dict[int, Mapping[str, Any]] = {}
    for raw_item_id, features in item_features.items():
        try:
            item_id = int(raw_item_id)
        except (TypeError, ValueError) as error:
            raise ValueError(f"Non-integer item key in {item_file}: {raw_item_id!r}") from error
        if item_id in parsed:
            raise ValueError(f"Duplicate integer item ID in {item_file}: {item_id}")
        if not isinstance(features, dict):
            raise ValueError(f"Item {item_id} in {item_file} is not a JSON object")
        parsed[item_id] = features

    _require_contiguous_ids(list(parsed), source=str(item_file))
    ordered_ids = tuple(range(len(parsed)))
    if item_mapping_file is not None:
        mapping_ids = load_item_mapping_ids(Path(item_mapping_file))
        if mapping_ids != ordered_ids:
            raise ValueError("item.json and item2id do not contain the same item IDs")
    return tuple(build_item_text(item_id, parsed[item_id]) for item_id in ordered_ids)


def masked_mean_pool(last_hidden_state: Any, attention_mask: Any) -> Any:
    """Apply the fixed upstream attention-mask-weighted mean pooling."""

    if last_hidden_state.ndim != 3:
        raise ValueError("last_hidden_state must have shape [batch, sequence, hidden]")
    if attention_mask.ndim != 2:
        raise ValueError("attention_mask must have shape [batch, sequence]")
    if tuple(last_hidden_state.shape[:2]) != tuple(attention_mask.shape):
        raise ValueError("last_hidden_state and attention_mask shapes do not align")
    expanded_mask = attention_mask.unsqueeze(-1).expand_as(last_hidden_state).float()
    summed = (last_hidden_state * expanded_mask).sum(dim=1)
    denominator = expanded_mask.sum(dim=1).clamp(min=1e-9)
    return summed / denominator


def artifact_paths(output_file: Path) -> EmbeddingArtifactPaths:
    output_file = Path(output_file)
    if output_file.suffix != ".npy":
        raise ValueError("output file must end in .npy")
    return EmbeddingArtifactPaths(
        embeddings=output_file,
        manifest=output_file.with_suffix(".manifest.json"),
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def ensure_new_artifacts(paths: EmbeddingArtifactPaths) -> None:
    if not paths.embeddings.parent.is_dir():
        raise FileNotFoundError(
            f"Output directory does not exist: {paths.embeddings.parent}"
        )
    existing = [path for path in paths.files if path.exists()]
    if existing:
        joined = ", ".join(str(path) for path in existing)
        raise FileExistsError(f"Refusing to overwrite existing output: {joined}")


def save_embedding_artifacts(
    embeddings: Any,
    item_ids: Sequence[int],
    paths: EmbeddingArtifactPaths,
    *,
    item_file: Path,
    item_mapping_file: Path,
    model_path: Path,
    max_length: int,
    batch_size: int,
    torch_dtype: str,
) -> dict[str, Any]:
    """Validate and save the matrix plus item-order and run manifests."""

    ensure_new_artifacts(paths)
    if embeddings.ndim != 2:
        raise ValueError("embeddings must be a two-dimensional matrix")
    if embeddings.shape[0] != len(item_ids):
        raise ValueError("embedding row count does not match item ID count")
    _require_contiguous_ids(list(item_ids), source="embedding item IDs")
    if not np.isfinite(embeddings).all():
        raise ValueError("embeddings contain NaN or Inf")
    if np.any(np.all(embeddings == 0, axis=1)):
        raise ValueError("at least one item embedding is all zero")

    matrix = np.asarray(embeddings, dtype=np.float32)
    manifest: dict[str, Any] = {
        "upstream_commit": OFFICIAL_UPSTREAM_COMMIT,
        "item_file": str(Path(item_file)),
        "item_file_sha256": _sha256(Path(item_file)),
        "item_mapping_file": str(Path(item_mapping_file)),
        "item_mapping_file_sha256": _sha256(Path(item_mapping_file)),
        "model_path": str(Path(model_path)),
        "features": list(OFFICIAL_FEATURES),
        "text_joiner": "single space",
        "pooling": "attention-mask weighted mean of last_hidden_state",
        "normalized": False,
        "max_length": max_length,
        "batch_size": batch_size,
        "model_torch_dtype": torch_dtype,
        "saved_numpy_dtype": str(matrix.dtype),
        "shape": list(matrix.shape),
        "embedding_file": str(paths.embeddings),
    }

    np.save(paths.embeddings, matrix, allow_pickle=False)
    with paths.manifest.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(manifest, handle, indent=2, ensure_ascii=False)
    return manifest


def generate_embeddings(
    *,
    item_file: Path,
    item_mapping_file: Path,
    model_path: Path,
    output_file: Path,
    batch_size: int = 8,
    max_length: int = OFFICIAL_MAX_LENGTH,
    device: str = "cuda:0",
    torch_dtype: str = "float16",
) -> dict[str, Any]:
    """Run single-device Qwen encoding and save official-compatible embeddings."""

    if batch_size < 1:
        raise ValueError("batch_size must be at least 1")
    if max_length < 1:
        raise ValueError("max_length must be at least 1")
    if torch_dtype not in {"bfloat16", "float16", "float32"}:
        raise ValueError("torch_dtype must be bfloat16, float16, or float32")

    item_file = Path(item_file)
    item_mapping_file = Path(item_mapping_file)
    model_path = Path(model_path)
    if not model_path.is_dir():
        raise FileNotFoundError(f"Model directory does not exist: {model_path}")
    paths = artifact_paths(Path(output_file))
    ensure_new_artifacts(paths)
    items = load_item_texts(item_file, item_mapping_file)
    if not items:
        raise ValueError("item file contains no items")

    if device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError(f"CUDA device requested but CUDA is unavailable: {device}")
    dtype = getattr(torch, torch_dtype)
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    tokenizer.padding_side = "right"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModel.from_pretrained(
        model_path,
        trust_remote_code=True,
        torch_dtype=dtype,
        low_cpu_mem_usage=True,
    )
    hidden_size = getattr(model.config, "hidden_size", None)
    if hidden_size != QWEN3_EMBEDDING_4B_HIDDEN_SIZE:
        raise ValueError(
            "Expected Qwen3-Embedding-4B hidden_size "
            f"{QWEN3_EMBEDDING_4B_HIDDEN_SIZE}, got {hidden_size}"
        )
    model.config.use_cache = False
    model.requires_grad_(False)
    model.to(device)
    model.eval()

    batches: list[np.ndarray] = []
    texts = [item.text for item in items]
    with torch.inference_mode():
        for start in tqdm(range(0, len(texts), batch_size), desc="Embedding items"):
            encoded = tokenizer(
                texts[start : start + batch_size],
                max_length=max_length,
                truncation=True,
                return_tensors="pt",
                padding=True,
            ).to(device)
            outputs = model(
                input_ids=encoded.input_ids,
                attention_mask=encoded.attention_mask,
                use_cache=False,
            )
            pooled = masked_mean_pool(outputs.last_hidden_state, encoded.attention_mask)
            batches.append(pooled.float().cpu().numpy())

    embeddings = np.concatenate(batches, axis=0)
    if embeddings.shape != (len(items), QWEN3_EMBEDDING_4B_HIDDEN_SIZE):
        raise ValueError(
            "Unexpected embedding shape: "
            f"expected {(len(items), QWEN3_EMBEDDING_4B_HIDDEN_SIZE)}, "
            f"got {embeddings.shape}"
        )
    return save_embedding_artifacts(
        embeddings,
        [item.item_id for item in items],
        paths,
        item_file=item_file,
        item_mapping_file=item_mapping_file,
        model_path=model_path,
        max_length=max_length,
        batch_size=batch_size,
        torch_dtype=torch_dtype,
    )
