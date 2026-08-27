"""Official sequence-normalized GRPO loss used by MiniOneRec."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
import torch.nn.functional as functional


OFFICIAL_BETA = 1e-3


@dataclass(frozen=True)
class GRPOLossOutput:
    """Scalar loss and the metrics logged by the official trainer."""

    loss: torch.Tensor
    mean_kl: torch.Tensor
    mean_completion_length: torch.Tensor


def selective_log_softmax(
    logits: torch.Tensor,
    token_ids: torch.Tensor,
) -> torch.Tensor:
    """Return log-probabilities only for selected token IDs."""

    if logits.ndim != 3:
        raise ValueError("logits must have shape [batch, sequence, vocabulary]")
    if token_ids.ndim != 2:
        raise ValueError("token_ids must have shape [batch, sequence]")
    if logits.shape[:2] != token_ids.shape:
        raise ValueError("logits and token_ids sequence shapes must match")

    if logits.dtype in (torch.float32, torch.float64):
        selected_logits = logits.gather(
            dim=-1,
            index=token_ids.unsqueeze(-1),
        ).squeeze(-1)
        log_normalizers = torch.stack(
            [torch.logsumexp(row_logits, dim=-1) for row_logits in logits]
        )
        return selected_logits - log_normalizers

    per_row_log_probs: list[torch.Tensor] = []
    for row_logits, row_token_ids in zip(logits, token_ids):
        row_log_probs = functional.log_softmax(row_logits, dim=-1)
        selected = row_log_probs.gather(
            dim=-1,
            index=row_token_ids.unsqueeze(-1),
        ).squeeze(-1)
        per_row_log_probs.append(selected)
    return torch.stack(per_row_log_probs)


def completion_token_log_probs(
    *,
    model: Any,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    completion_length: int,
) -> torch.Tensor:
    """Compute causal log-probabilities for completion tokens only."""

    if input_ids.ndim != 2 or attention_mask.ndim != 2:
        raise ValueError("input_ids and attention_mask must be two-dimensional")
    if input_ids.shape != attention_mask.shape:
        raise ValueError("input_ids and attention_mask shapes must match")
    if completion_length < 1 or completion_length >= input_ids.size(1):
        raise ValueError("completion_length must fit inside the input sequence")

    outputs = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        logits_to_keep=completion_length + 1,
    )
    logits = outputs.logits
    expected_positions = completion_length + 1
    if logits.size(1) < expected_positions:
        raise RuntimeError(
            f"Expected at least {expected_positions} logit positions, got "
            f"{logits.size(1)}"
        )
    logits = logits[:, -expected_positions:-1, :]
    completion_ids = input_ids[:, -completion_length:]
    return selective_log_softmax(logits, completion_ids)


def official_grpo_loss(
    *,
    policy_log_probs: torch.Tensor,
    reference_log_probs: torch.Tensor,
    advantages: torch.Tensor,
    completion_mask: torch.Tensor,
    beta: float = OFFICIAL_BETA,
) -> GRPOLossOutput:
    """Compute the fixed commit's non-DAPO, non-GSPO GRPO objective."""

    if policy_log_probs.ndim != 2:
        raise ValueError("policy_log_probs must be two-dimensional")
    if reference_log_probs.shape != policy_log_probs.shape:
        raise ValueError("reference and policy log-probability shapes must match")
    if completion_mask.shape != policy_log_probs.shape:
        raise ValueError("completion_mask and log-probability shapes must match")
    if advantages.ndim != 1 or advantages.size(0) != policy_log_probs.size(0):
        raise ValueError("advantages must contain one value per candidate")
    if beta < 0:
        raise ValueError("beta cannot be negative")

    mask = completion_mask.to(dtype=policy_log_probs.dtype)
    completion_lengths = mask.sum(dim=1)
    if torch.any(completion_lengths == 0):
        raise ValueError("every candidate must contain a supervised token")

    reference_log_probs = reference_log_probs.detach()
    log_ratio = reference_log_probs - policy_log_probs
    per_token_kl = torch.exp(log_ratio) - log_ratio - 1

    policy_ratio = torch.exp(policy_log_probs - policy_log_probs.detach())
    per_token_loss = -(
        policy_ratio * advantages.unsqueeze(1) - beta * per_token_kl
    )
    per_candidate_loss = (
        (per_token_loss * mask).sum(dim=1) / completion_lengths
    )
    loss = per_candidate_loss.mean()
    mean_kl = (
        (per_token_kl * mask).sum(dim=1) / completion_lengths
    ).mean()
    mean_completion_length = completion_lengths.float().mean()
    return GRPOLossOutput(
        loss=loss,
        mean_kl=mean_kl,
        mean_completion_length=mean_completion_length,
    )
