"""Small SASRec teacher used to score real catalog items."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch import nn


@dataclass(frozen=True)
class SASRecConfig:
    """Architecture parameters for the collaborative teacher."""

    num_items: int
    max_sequence_length: int = 10
    hidden_size: int = 32
    num_layers: int = 2
    num_heads: int = 2
    dropout: float = 0.3

    def validate(self) -> None:
        """Validate values before allocating model parameters."""

        if self.num_items <= 1:
            raise ValueError("num_items must be greater than one")
        if self.max_sequence_length <= 0:
            raise ValueError("max_sequence_length must be positive")
        if self.hidden_size <= 0:
            raise ValueError("hidden_size must be positive")
        if self.num_layers <= 0:
            raise ValueError("num_layers must be positive")
        if self.num_heads <= 0 or self.hidden_size % self.num_heads:
            raise ValueError("num_heads must evenly divide hidden_size")
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError("dropout must be in [0, 1)")


class SASRec(nn.Module):
    """Causal Transformer that predicts the next real item ID."""

    def __init__(self, config: SASRecConfig) -> None:
        super().__init__()
        config.validate()
        self.config = config

        # CSV item IDs are zero based. They are shifted by one before entering
        # the model so internal ID zero can be reserved for padding.
        self.item_embedding = nn.Embedding(
            config.num_items + 1,
            config.hidden_size,
            padding_idx=0,
        )
        self.position_embedding = nn.Embedding(
            config.max_sequence_length,
            config.hidden_size,
        )
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config.hidden_size,
            nhead=config.num_heads,
            dim_feedforward=config.hidden_size * 4,
            dropout=config.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=config.num_layers,
            enable_nested_tensor=False,
        )
        self.input_dropout = nn.Dropout(config.dropout)
        self.output_norm = nn.LayerNorm(config.hidden_size)
        self._reset_parameters()

    def _reset_parameters(self) -> None:
        nn.init.normal_(self.item_embedding.weight, mean=0.0, std=0.02)
        nn.init.normal_(self.position_embedding.weight, mean=0.0, std=0.02)
        with torch.no_grad():
            self.item_embedding.weight[0].zero_()

    def sequence_features(self, history_ids: torch.Tensor) -> torch.Tensor:
        """Encode right-padded internal item IDs into one vector per history."""

        if history_ids.ndim != 2:
            raise ValueError("history_ids must have shape [batch, sequence]")
        if history_ids.shape[1] != self.config.max_sequence_length:
            raise ValueError(
                "history sequence width must equal max_sequence_length"
            )
        padding_mask = history_ids.eq(0)
        lengths = history_ids.ne(0).sum(dim=1)
        if torch.any(lengths == 0):
            raise ValueError("every history must contain at least one item")

        positions = torch.arange(
            history_ids.shape[1],
            device=history_ids.device,
        ).unsqueeze(0)
        hidden = self.item_embedding(history_ids)
        hidden = hidden + self.position_embedding(positions)
        hidden = self.input_dropout(hidden)

        sequence_width = history_ids.shape[1]
        causal_mask = torch.triu(
            torch.ones(
                sequence_width,
                sequence_width,
                dtype=torch.bool,
                device=history_ids.device,
            ),
            diagonal=1,
        )
        hidden = self.encoder(
            hidden,
            mask=causal_mask,
            src_key_padding_mask=padding_mask,
        )
        final_positions = lengths - 1
        batch_indices = torch.arange(history_ids.shape[0], device=history_ids.device)
        return self.output_norm(hidden[batch_indices, final_positions])

    def forward(self, history_ids: torch.Tensor) -> torch.Tensor:
        """Return full-catalog logits in original item-ID column order."""

        features = self.sequence_features(history_ids)
        catalog_embeddings = self.item_embedding.weight[1:]
        return features @ catalog_embeddings.transpose(0, 1)

    def score_items(
        self,
        history_ids: torch.Tensor,
        original_item_ids: torch.Tensor,
    ) -> torch.Tensor:
        """Score selected zero-based catalog item IDs for each history."""

        if original_item_ids.ndim != 2:
            raise ValueError("original_item_ids must have shape [batch, candidates]")
        if original_item_ids.shape[0] != history_ids.shape[0]:
            raise ValueError("history and candidate batch sizes must match")
        if torch.any(original_item_ids < 0) or torch.any(
            original_item_ids >= self.config.num_items
        ):
            raise ValueError("candidate item ID is outside the catalog")
        features = self.sequence_features(history_ids)
        candidate_embeddings = self.item_embedding(original_item_ids + 1)
        return torch.einsum("bd,bkd->bk", features, candidate_embeddings)


def load_sasrec_checkpoint(
    checkpoint_file: Path,
    *,
    device: str | torch.device,
) -> tuple[SASRec, dict[str, Any]]:
    """Load one validated teacher checkpoint and freeze its parameters."""

    checkpoint_file = Path(checkpoint_file)
    if not checkpoint_file.is_file():
        raise FileNotFoundError(checkpoint_file)
    payload = torch.load(
        checkpoint_file,
        map_location=device,
        weights_only=True,
    )
    if not isinstance(payload, dict):
        raise TypeError("SASRec checkpoint must contain a dictionary")
    raw_config = payload.get("model_config")
    model_state = payload.get("model_state_dict")
    if not isinstance(raw_config, dict) or not isinstance(model_state, dict):
        raise ValueError("SASRec checkpoint is missing model configuration or weights")

    model = SASRec(SASRecConfig(**raw_config))
    model.load_state_dict(model_state, strict=True)
    model.to(device)
    model.eval()
    model.requires_grad_(False)
    return model, payload
