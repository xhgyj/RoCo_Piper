"""A proprioception-only Action Chunking Transformer.

LeRobot 0.4.4 requires an image or ``observation.environment_state`` for its
ACT implementation.  This local variant keeps the ACT conditional VAE and
action-query decoder while accepting only ``observation.state``.
"""

from __future__ import annotations

from collections import deque
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as functional


@dataclass(frozen=True)
class ProprioACTConfig:
    """Architecture and inference configuration for proprioceptive ACT."""

    state_dim: int = 230
    action_dim: int = 6
    chunk_size: int = 10
    dim_model: int = 256
    n_heads: int = 8
    dim_feedforward: int = 1024
    n_encoder_layers: int = 4
    n_decoder_layers: int = 2
    latent_dim: int = 32
    dropout: float = 0.1
    kl_weight: float = 10.0
    temporal_ensemble_coeff: float = 0.1


@dataclass(frozen=True)
class ACTLoss:
    """Named loss values returned during policy training."""

    loss: torch.Tensor
    reconstruction: torch.Tensor
    kl: torch.Tensor


class FeatureNormalizer:
    """Mean/std normalization stored alongside a checkpoint."""

    def __init__(
        self,
        state_mean: np.ndarray,
        state_std: np.ndarray,
        action_mean: np.ndarray,
        action_std: np.ndarray,
    ):
        """Store state and action statistics with safe standard deviations."""
        self.state_mean = np.asarray(state_mean, dtype=np.float32)
        self.state_std = np.maximum(np.asarray(state_std, dtype=np.float32), 1e-6)
        self.action_mean = np.asarray(action_mean, dtype=np.float32)
        self.action_std = np.maximum(np.asarray(action_std, dtype=np.float32), 1e-6)

    @classmethod
    def from_samples(
        cls, states: np.ndarray, actions: np.ndarray
    ) -> FeatureNormalizer:
        """Fit normalization statistics from training samples only.

        Returns:
            Fitted normalizer.
        """
        state_values = np.asarray(states, dtype=np.float32)
        action_values = np.asarray(actions, dtype=np.float32)
        if state_values.ndim != 2 or action_values.ndim != 2:
            raise ValueError("states and actions must both be rank-two arrays")
        return cls(
            state_values.mean(axis=0),
            state_values.std(axis=0),
            action_values.mean(axis=0),
            action_values.std(axis=0),
        )

    def normalize_state(self, value: np.ndarray) -> np.ndarray:
        """Normalize a state vector or batch.

        Returns:
            Normalized state values.
        """
        return (np.asarray(value, dtype=np.float32) - self.state_mean) / self.state_std

    def normalize_action(self, value: np.ndarray) -> np.ndarray:
        """Normalize an action vector or batch.

        Returns:
            Normalized action values.
        """
        value_array = np.asarray(value, dtype=np.float32)
        return (value_array - self.action_mean) / self.action_std

    def denormalize_action(self, value: np.ndarray) -> np.ndarray:
        """Convert normalized model actions to physical units.

        Returns:
            Action values in physical units.
        """
        return np.asarray(value, dtype=np.float32) * self.action_std + self.action_mean

    def state_dict(self) -> dict[str, np.ndarray]:
        """Return serializable normalization arrays."""
        return {
            "state_mean": self.state_mean,
            "state_std": self.state_std,
            "action_mean": self.action_mean,
            "action_std": self.action_std,
        }


class ProprioACT(nn.Module):
    """Conditional VAE transformer that predicts a Cartesian action chunk."""

    def __init__(self, config: ProprioACTConfig):
        """Construct the conditional VAE and action decoder."""
        super().__init__()
        self.config = config
        d_model = config.dim_model
        self.state_projection = nn.Linear(config.state_dim, d_model)
        self.action_projection = nn.Linear(config.action_dim, d_model)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
        self.vae_position = nn.Parameter(
            torch.zeros(1, config.chunk_size + 2, d_model)
        )
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=config.n_heads,
            dim_feedforward=config.dim_feedforward,
            dropout=config.dropout,
            batch_first=True,
            norm_first=True,
        )
        self.vae_encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=config.n_encoder_layers
        )
        self.latent_projection = nn.Linear(d_model, 2 * config.latent_dim)
        self.latent_input = nn.Linear(config.latent_dim, d_model)
        self.action_queries = nn.Parameter(
            torch.zeros(1, config.chunk_size, d_model)
        )
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=config.n_heads,
            dim_feedforward=config.dim_feedforward,
            dropout=config.dropout,
            batch_first=True,
            norm_first=True,
        )
        self.decoder = nn.TransformerDecoder(
            decoder_layer, num_layers=config.n_decoder_layers
        )
        self.action_head = nn.Linear(d_model, config.action_dim)
        self._reset_parameters()

    def _reset_parameters(self) -> None:
        nn.init.normal_(self.cls_token, std=0.02)
        nn.init.normal_(self.vae_position, std=0.02)
        nn.init.normal_(self.action_queries, std=0.02)

    def encode_actions(
        self,
        states: torch.Tensor,
        actions: torch.Tensor,
        action_is_pad: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Encode a demonstrated action chunk into the ACT latent.

        Returns:
            Sampled latent, mean, and log variance.
        """
        batch_size = states.shape[0]
        state_token = self.state_projection(states).unsqueeze(1)
        action_tokens = self.action_projection(actions)
        tokens = torch.cat(
            (self.cls_token.expand(batch_size, -1, -1), state_token, action_tokens),
            dim=1,
        )
        tokens = tokens + self.vae_position
        padding_mask = None
        if action_is_pad is not None:
            prefix = torch.zeros(
                (batch_size, 2), dtype=torch.bool, device=states.device
            )
            padding_mask = torch.cat((prefix, action_is_pad.bool()), dim=1)
        encoded = self.vae_encoder(tokens, src_key_padding_mask=padding_mask)
        parameters = self.latent_projection(encoded[:, 0])
        mean, log_variance = parameters.chunk(2, dim=-1)
        latent = mean + torch.exp(0.5 * log_variance) * torch.randn_like(mean)
        return latent, mean, log_variance

    def predict_chunk(
        self, states: torch.Tensor, latent: torch.Tensor | None = None
    ) -> torch.Tensor:
        """Predict a normalized action chunk for normalized states.

        Returns:
            Predicted action chunk.
        """
        batch_size = states.shape[0]
        if latent is None:
            latent = torch.zeros(
                (batch_size, self.config.latent_dim),
                dtype=states.dtype,
                device=states.device,
            )
        memory = torch.stack(
            (self.state_projection(states), self.latent_input(latent)), dim=1
        )
        queries = self.action_queries.expand(batch_size, -1, -1)
        decoded = self.decoder(queries, memory)
        return self.action_head(decoded)

    def forward(
        self,
        states: torch.Tensor,
        actions: torch.Tensor,
        action_is_pad: torch.Tensor | None = None,
    ) -> ACTLoss:
        """Return the masked ACT reconstruction and KL training loss."""
        latent, mean, log_variance = self.encode_actions(
            states, actions, action_is_pad
        )
        predicted = self.predict_chunk(states, latent)
        reconstruction_values = functional.l1_loss(
            predicted, actions, reduction="none"
        )
        if action_is_pad is not None:
            valid = (~action_is_pad.bool()).unsqueeze(-1)
            denominator = valid.sum().clamp_min(1) * actions.shape[-1]
            reconstruction = (reconstruction_values * valid).sum() / denominator
        else:
            reconstruction = reconstruction_values.mean()
        kl = -0.5 * (1 + log_variance - mean.square() - log_variance.exp()).mean()
        loss = reconstruction + self.config.kl_weight * kl
        return ACTLoss(loss, reconstruction, kl)


class ProprioACTPolicy:
    """Normalized, temporally ensembled inference wrapper."""

    def __init__(
        self,
        model: ProprioACT,
        normalizer: FeatureNormalizer,
        device: torch.device | str,
    ):
        """Bind a trained model, normalizer, and inference device."""
        self.model = model.to(device).eval()
        self.normalizer = normalizer
        self.device = torch.device(device)
        self._chunks: deque[np.ndarray] = deque(maxlen=model.config.chunk_size)

    def reset(self) -> None:
        """Discard predictions from the preceding episode."""
        self._chunks.clear()

    @torch.inference_mode()
    def select_action(self, state: np.ndarray) -> np.ndarray:
        """Predict and ensemble the current physical 6D action.

        Returns:
            Current action in physical units.
        """
        normalized = self.normalizer.normalize_state(state)
        tensor = torch.as_tensor(normalized, device=self.device).unsqueeze(0)
        chunk = self.model.predict_chunk(tensor).squeeze(0).cpu().numpy()
        chunk = self.normalizer.denormalize_action(chunk)
        self._chunks.appendleft(chunk)
        candidates = []
        weights = []
        coefficient = self.model.config.temporal_ensemble_coeff
        for age, prior_chunk in enumerate(self._chunks):
            if age >= prior_chunk.shape[0]:
                continue
            candidates.append(prior_chunk[age])
            weights.append(np.exp(-coefficient * age))
        weight_array = np.asarray(weights, dtype=np.float64)
        values = np.stack(candidates)
        return np.average(values, axis=0, weights=weight_array).astype(np.float32)


def save_checkpoint(
    path: str | Path,
    model: ProprioACT,
    normalizer: FeatureNormalizer,
    optimizer: torch.optim.Optimizer | None = None,
    step: int = 0,
) -> None:
    """Save model, normalization, optimizer, and configuration together."""
    payload: dict[str, object] = {
        "config": asdict(model.config),
        "model": model.state_dict(),
        "normalizer": normalizer.state_dict(),
        "step": step,
    }
    if optimizer is not None:
        payload["optimizer"] = optimizer.state_dict()
    torch.save(payload, Path(path))


def load_policy(
    path: str | Path, device: torch.device | str = "cpu"
) -> ProprioACTPolicy:
    """Load a complete inference policy from a local checkpoint.

    Returns:
        Ready-to-run normalized ACT policy.
    """
    payload = torch.load(Path(path), map_location=device, weights_only=False)
    config = ProprioACTConfig(**payload["config"])
    model = ProprioACT(config)
    model.load_state_dict(payload["model"])
    normalizer = FeatureNormalizer(**payload["normalizer"])
    return ProprioACTPolicy(model, normalizer, device)
