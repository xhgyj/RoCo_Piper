"""Unit tests for the proprioception-only ACT model and sequence data."""

from pathlib import Path

import numpy as np
import torch

from rocobrick.policy.mate_down_data import (
    MateDownSequenceDataset,
    episode_split,
    fit_normalizer,
)
from rocobrick.policy.proprio_act import (
    FeatureNormalizer,
    ProprioACT,
    ProprioACTConfig,
    load_policy,
    save_checkpoint,
)


def _frames():
    return [
        {
            "episode_index": episode,
            "observation.state": np.full(4, episode * 10 + step, dtype=np.float32),
            "action": np.full(2, step, dtype=np.float32),
        }
        for episode in range(2)
        for step in range(3)
    ]


def _config():
    return ProprioACTConfig(
        state_dim=4,
        action_dim=2,
        chunk_size=3,
        dim_model=16,
        n_heads=4,
        dim_feedforward=32,
        n_encoder_layers=1,
        n_decoder_layers=1,
        latent_dim=4,
    )


def test_sequence_chunks_do_not_cross_episode_boundaries():
    """Future action targets are padded at an episode boundary."""
    dataset = MateDownSequenceDataset(_frames(), chunk_size=3)
    item = dataset[2]
    np.testing.assert_array_equal(item["action"][0], [2.0, 2.0])
    assert item["action_is_pad"].tolist() == [False, True, True]


def test_episode_split_keeps_whole_episodes_together():
    """Validation selection never leaks frames from its episode into training."""
    frames = _frames()
    train, validation = episode_split(frames, validation_fraction=0.5)
    assert train == [0, 1, 2]
    assert validation == [3, 4, 5]


def test_proprio_act_forward_returns_finite_masked_loss():
    """Conditional VAE accepts only state and padded action chunks."""
    model = ProprioACT(_config())
    states = torch.randn(2, 4)
    actions = torch.randn(2, 3, 2)
    padding = torch.tensor([[False, False, False], [False, True, True]])
    result = model(states, actions, padding)
    assert result.loss.ndim == 0
    assert torch.isfinite(result.loss)
    assert model.predict_chunk(states).shape == (2, 3, 2)


def test_checkpoint_round_trip_preserves_inference_contract(tmp_path: Path):
    """Checkpoint includes architecture and normalization needed by evaluation."""
    model = ProprioACT(_config())
    normalizer = FeatureNormalizer(
        np.zeros(4), np.ones(4), np.zeros(2), np.ones(2)
    )
    checkpoint = tmp_path / "policy.pt"
    save_checkpoint(checkpoint, model, normalizer)
    policy = load_policy(checkpoint)
    action = policy.select_action(np.zeros(4, dtype=np.float32))
    assert action.shape == (2,)
    assert np.isfinite(action).all()


def test_normalizer_is_fit_from_robot_state_and_actions_only():
    """Normalization consumes exactly the two learning features."""
    normalizer = fit_normalizer(_frames())
    assert normalizer.state_mean.shape == (4,)
    assert normalizer.action_mean.shape == (2,)
