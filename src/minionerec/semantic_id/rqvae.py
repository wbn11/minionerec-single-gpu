# SPDX-License-Identifier: Apache-2.0
"""Official-compatible residual-quantized VAE used by MiniOneRec."""

from __future__ import annotations

from collections.abc import Sequence

import torch
from sklearn.cluster import KMeans
from torch import nn
from torch.nn import functional as F
from torch.nn.init import xavier_normal_


OFFICIAL_CODEBOOK_SIZES = (256, 256, 256)
OFFICIAL_LATENT_DIM = 32
OFFICIAL_HIDDEN_DIMS = (2048, 1024, 512, 256, 128, 64)


class MLPLayers(nn.Module):
    """The MLP stack used by the fixed upstream RQ-VAE implementation."""

    def __init__(
        self,
        layers: Sequence[int],
        *,
        dropout: float = 0.0,
        batch_norm: bool = False,
    ) -> None:
        super().__init__()
        if len(layers) < 2:
            raise ValueError("MLP needs at least an input and output dimension")

        modules: list[nn.Module] = []
        for index, (input_size, output_size) in enumerate(
            zip(layers[:-1], layers[1:])
        ):
            modules.append(nn.Dropout(p=dropout))
            modules.append(nn.Linear(input_size, output_size))
            if batch_norm:
                modules.append(nn.BatchNorm1d(output_size))
            if index != len(layers) - 2:
                modules.append(nn.ReLU())
        self.mlp_layers = nn.Sequential(*modules)
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            xavier_normal_(module.weight.data)
            if module.bias is not None:
                module.bias.data.zero_()

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.mlp_layers(inputs)


def kmeans(
    samples: torch.Tensor,
    num_clusters: int,
    num_iters: int,
) -> torch.Tensor:
    """Run the same scikit-learn KMeans initialization as upstream."""

    if samples.ndim != 2:
        raise ValueError("KMeans samples must have shape [samples, dimension]")
    if samples.shape[0] < num_clusters:
        raise ValueError(
            f"KMeans needs at least {num_clusters} samples, got {samples.shape[0]}"
        )
    fitted = KMeans(n_clusters=num_clusters, max_iter=num_iters).fit(
        samples.detach().cpu().numpy()
    )
    return torch.from_numpy(fitted.cluster_centers_).to(
        device=samples.device,
        dtype=samples.dtype,
    )


@torch.no_grad()
def sinkhorn_algorithm(
    distances: torch.Tensor,
    epsilon: float,
    iterations: int,
) -> torch.Tensor:
    """Balanced assignment used by the upstream collision post-processing."""

    assignments = torch.exp(-distances / epsilon)
    batch_size, codebook_size = assignments.shape
    assignments /= assignments.sum(dim=1, keepdim=True).sum(dim=0, keepdim=True)
    for _ in range(iterations):
        assignments /= assignments.sum(dim=1, keepdim=True)
        assignments /= batch_size
        assignments /= assignments.sum(dim=0, keepdim=True)
        assignments /= codebook_size
    assignments *= batch_size
    return assignments


class VectorQuantizer(nn.Module):
    """One learnable codebook with straight-through vector quantization."""

    def __init__(
        self,
        codebook_size: int,
        embedding_dim: int,
        *,
        beta: float = 0.25,
        kmeans_init: bool = False,
        kmeans_iters: int = 100,
        sinkhorn_epsilon: float = 0.0,
        sinkhorn_iters: int = 50,
    ) -> None:
        super().__init__()
        self.codebook_size = codebook_size
        self.embedding_dim = embedding_dim
        self.beta = beta
        self.kmeans_init = kmeans_init
        self.kmeans_iters = kmeans_iters
        self.sinkhorn_epsilon = sinkhorn_epsilon
        self.sinkhorn_iters = sinkhorn_iters
        self.embedding = nn.Embedding(codebook_size, embedding_dim)
        if kmeans_init:
            self.initted = False
            self.embedding.weight.data.zero_()
        else:
            self.initted = True
            self.embedding.weight.data.uniform_(
                -1.0 / codebook_size,
                1.0 / codebook_size,
            )

    def initialize_codebook(self, samples: torch.Tensor) -> None:
        centers = kmeans(samples, self.codebook_size, self.kmeans_iters)
        self.embedding.weight.data.copy_(centers)
        self.initted = True

    @staticmethod
    def _center_distances(distances: torch.Tensor) -> torch.Tensor:
        maximum = distances.max()
        minimum = distances.min()
        middle = (maximum + minimum) / 2
        amplitude = maximum - middle + 1e-5
        return (distances - middle) / amplitude

    def forward(
        self,
        inputs: torch.Tensor,
        *,
        use_sinkhorn: bool = True,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        latent = inputs.reshape(-1, self.embedding_dim)
        if not self.initted and self.training:
            self.initialize_codebook(latent)

        distances = (
            latent.square().sum(dim=1, keepdim=True)
            + self.embedding.weight.square().sum(dim=1, keepdim=True).t()
            - 2 * latent @ self.embedding.weight.t()
        )
        if not use_sinkhorn or self.sinkhorn_epsilon <= 0:
            indices = torch.argmin(distances, dim=-1)
        else:
            centered = self._center_distances(distances).double()
            assignments = sinkhorn_algorithm(
                centered,
                self.sinkhorn_epsilon,
                self.sinkhorn_iters,
            )
            finite = assignments[torch.isfinite(assignments)]
            if finite.numel() == 0:
                raise FloatingPointError("Sinkhorn returned no finite assignments")
            assignments = torch.nan_to_num(assignments, nan=finite.min().item())
            if not torch.isfinite(assignments).all():
                raise FloatingPointError("Sinkhorn returned NaN or Inf")
            indices = torch.argmax(assignments, dim=-1)

        quantized = self.embedding(indices).view_as(inputs)
        commitment_loss = F.mse_loss(quantized.detach(), inputs)
        codebook_loss = F.mse_loss(quantized, inputs.detach())
        loss = codebook_loss + self.beta * commitment_loss
        quantized = inputs + (quantized - inputs).detach()
        return quantized, loss, indices.view(inputs.shape[:-1])


class ResidualVectorQuantizer(nn.Module):
    """Apply several codebooks successively to the remaining residual."""

    def __init__(
        self,
        codebook_sizes: Sequence[int],
        embedding_dim: int,
        *,
        beta: float = 0.25,
        kmeans_init: bool = False,
        kmeans_iters: int = 100,
        sinkhorn_epsilons: Sequence[float] | None = None,
        sinkhorn_iters: int = 50,
    ) -> None:
        super().__init__()
        epsilons = tuple(sinkhorn_epsilons or [0.0] * len(codebook_sizes))
        if len(epsilons) != len(codebook_sizes):
            raise ValueError("one Sinkhorn epsilon is required for each codebook")
        self.vq_layers = nn.ModuleList(
            [
                VectorQuantizer(
                    codebook_size,
                    embedding_dim,
                    beta=beta,
                    kmeans_init=kmeans_init,
                    kmeans_iters=kmeans_iters,
                    sinkhorn_epsilon=epsilon,
                    sinkhorn_iters=sinkhorn_iters,
                )
                for codebook_size, epsilon in zip(codebook_sizes, epsilons)
            ]
        )

    def mark_codebooks_initialized(self) -> None:
        """Prevent KMeans from running again after loading trained weights."""

        for quantizer in self.vq_layers:
            if not isinstance(quantizer, VectorQuantizer):
                raise TypeError("vq_layers must contain VectorQuantizer instances")
            quantizer.initted = True

    def forward(
        self,
        inputs: torch.Tensor,
        *,
        use_sinkhorn: bool = True,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        losses: list[torch.Tensor] = []
        indices: list[torch.Tensor] = []
        quantized_sum = torch.zeros_like(inputs)
        residual = inputs
        for quantizer in self.vq_layers:
            quantized, loss, layer_indices = quantizer(
                residual,
                use_sinkhorn=use_sinkhorn,
            )
            residual = residual - quantized
            quantized_sum = quantized_sum + quantized
            losses.append(loss)
            indices.append(layer_indices)
        return (
            quantized_sum,
            torch.stack(losses).mean(),
            torch.stack(indices, dim=-1),
        )


class RQVAE(nn.Module):
    """The three-level RQ-VAE used to turn item embeddings into SIDs."""

    def __init__(
        self,
        input_dim: int,
        *,
        codebook_sizes: Sequence[int] = OFFICIAL_CODEBOOK_SIZES,
        latent_dim: int = OFFICIAL_LATENT_DIM,
        hidden_dims: Sequence[int] = OFFICIAL_HIDDEN_DIMS,
        dropout: float = 0.0,
        batch_norm: bool = False,
        quant_loss_weight: float = 1.0,
        beta: float = 0.25,
        kmeans_init: bool = True,
        kmeans_iters: int = 100,
        sinkhorn_epsilons: Sequence[float] | None = None,
        sinkhorn_iters: int = 50,
    ) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.codebook_sizes = tuple(codebook_sizes)
        self.latent_dim = latent_dim
        self.hidden_dims = tuple(hidden_dims)
        self.quant_loss_weight = quant_loss_weight

        encoder_dims = (input_dim, *self.hidden_dims, latent_dim)
        self.encoder = MLPLayers(
            encoder_dims,
            dropout=dropout,
            batch_norm=batch_norm,
        )
        self.rq = ResidualVectorQuantizer(
            self.codebook_sizes,
            latent_dim,
            beta=beta,
            kmeans_init=kmeans_init,
            kmeans_iters=kmeans_iters,
            sinkhorn_epsilons=sinkhorn_epsilons,
            sinkhorn_iters=sinkhorn_iters,
        )
        self.decoder = MLPLayers(
            tuple(reversed(encoder_dims)),
            dropout=dropout,
            batch_norm=batch_norm,
        )

    def forward(
        self,
        inputs: torch.Tensor,
        *,
        use_sinkhorn: bool = True,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        encoded = self.encoder(inputs)
        quantized, quantization_loss, indices = self.rq(
            encoded,
            use_sinkhorn=use_sinkhorn,
        )
        return self.decoder(quantized), quantization_loss, indices

    @torch.no_grad()
    def get_indices(
        self,
        inputs: torch.Tensor,
        *,
        use_sinkhorn: bool = False,
    ) -> torch.Tensor:
        encoded = self.encoder(inputs)
        _, _, indices = self.rq(encoded, use_sinkhorn=use_sinkhorn)
        return indices

    def compute_loss(
        self,
        reconstruction: torch.Tensor,
        quantization_loss: torch.Tensor,
        inputs: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        reconstruction_loss = F.mse_loss(reconstruction, inputs, reduction="mean")
        total_loss = reconstruction_loss + self.quant_loss_weight * quantization_loss
        return total_loss, reconstruction_loss
