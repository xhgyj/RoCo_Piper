"""Dataset utilities for proprioceptive Mate-down ACT training."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import torch
from torch.utils.data import Dataset

from rocobrick.policy.proprio_act import FeatureNormalizer


class MateDownSequenceDataset(Dataset):
    """Create episode-bounded future action chunks from LeRobot frames."""

    def __init__(self, frames: Sequence[dict[str, object]], chunk_size: int):
        """Store frame access and requested chunk length."""
        if chunk_size <= 0:
            raise ValueError("chunk_size must be positive")
        self.frames = frames
        self.chunk_size = chunk_size

    def __len__(self) -> int:
        """Return the number of starting frames."""
        return len(self.frames)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        """Return one state and its padded future action chunk."""
        current = self.frames[index]
        episode = _scalar_int(current["episode_index"])
        state = _tensor(current["observation.state"])
        action_template = _tensor(current["action"])
        actions = torch.zeros(
            (self.chunk_size, action_template.numel()), dtype=torch.float32
        )
        action_is_pad = torch.ones(self.chunk_size, dtype=torch.bool)
        for offset in range(self.chunk_size):
            frame_index = index + offset
            if frame_index >= len(self.frames):
                break
            frame = self.frames[frame_index]
            if _scalar_int(frame["episode_index"]) != episode:
                break
            actions[offset] = _tensor(frame["action"])
            action_is_pad[offset] = False
        return {
            "observation.state": state,
            "action": actions,
            "action_is_pad": action_is_pad,
        }


def fit_normalizer(frames: Sequence[dict[str, object]]) -> FeatureNormalizer:
    """Fit state/action normalization using the provided training frames.

    Returns:
        Fitted feature normalizer.
    """
    if not frames:
        raise ValueError("cannot fit normalizer on an empty dataset")
    states = np.stack(
        [np.asarray(frame["observation.state"], dtype=np.float32) for frame in frames]
    )
    actions = np.stack(
        [np.asarray(frame["action"], dtype=np.float32) for frame in frames]
    )
    return FeatureNormalizer.from_samples(states, actions)


class NormalizedMateDownDataset(Dataset):
    """Apply fixed training statistics to state and action chunks."""

    def __init__(
        self,
        dataset: MateDownSequenceDataset,
        normalizer: FeatureNormalizer,
    ):
        """Wrap a sequence dataset with fixed statistics."""
        self.dataset = dataset
        self.normalizer = normalizer

    def __len__(self) -> int:
        """Return the wrapped dataset length."""
        return len(self.dataset)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        """Return a normalized sequence item."""
        item = self.dataset[index]
        state = self.normalizer.normalize_state(item["observation.state"].numpy())
        action = self.normalizer.normalize_action(item["action"].numpy())
        return {
            "observation.state": torch.from_numpy(state),
            "action": torch.from_numpy(action),
            "action_is_pad": item["action_is_pad"],
        }


def episode_split(
    frames: Sequence[dict[str, object]], validation_fraction: float = 0.1
) -> tuple[list[int], list[int]]:
    """Return deterministic train/validation frame indices split by episode."""
    if not 0.0 < validation_fraction < 1.0:
        raise ValueError("validation_fraction must be in (0, 1)")
    episodes = sorted({_scalar_int(frame["episode_index"]) for frame in frames})
    validation_count = max(1, round(len(episodes) * validation_fraction))
    validation_episodes = set(episodes[-validation_count:])
    train = []
    validation = []
    for index, frame in enumerate(frames):
        destination = (
            validation
            if _scalar_int(frame["episode_index"]) in validation_episodes
            else train
        )
        destination.append(index)
    if not train:
        raise ValueError("at least two episodes are required for an episode split")
    return train, validation


def _scalar_int(value: object) -> int:
    if isinstance(value, torch.Tensor):
        return int(value.item())
    return int(np.asarray(value).item())


def _tensor(value: object) -> torch.Tensor:
    if isinstance(value, torch.Tensor):
        return value.detach().to(dtype=torch.float32).flatten()
    return torch.as_tensor(np.asarray(value), dtype=torch.float32).flatten()
