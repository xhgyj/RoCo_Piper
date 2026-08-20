#!/usr/bin/env python3
"""Train proprioceptive ACT on successful Mate-down demonstrations."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from torch.utils.data import DataLoader

from rocobrick.policy.mate_down import FRAME_STATE_DIM
from rocobrick.policy.mate_down_data import (
    MateDownSequenceDataset,
    NormalizedMateDownDataset,
    episode_split,
    fit_normalizer,
)
from rocobrick.policy.mate_down_runtime import load_runtime_config
from rocobrick.policy.proprio_act import (
    ProprioACT,
    ProprioACTConfig,
    save_checkpoint,
)


def _arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--repo-id", default="local/roco-mate-down-successes")
    parser.add_argument("--output", default="outputs/mate_down_act.pt")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu"
    )
    return parser.parse_args()


def main():
    """Train and save a proprioceptive ACT checkpoint."""
    args = _arguments()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    root = Path(__file__).resolve().parents[1]
    runtime_config, _, act_values = load_runtime_config(
        root / "config/mate_down_config.json"
    )
    source = LeRobotDataset(args.repo_id, root=Path(args.dataset))
    frames = [source[index] for index in range(len(source))]
    train_indices, validation_indices = episode_split(frames)
    train_frames = [frames[index] for index in train_indices]
    validation_frames = [frames[index] for index in validation_indices]
    normalizer = fit_normalizer(train_frames)
    chunk_size = int(act_values["chunk_size"])
    train_data = NormalizedMateDownDataset(
        MateDownSequenceDataset(train_frames, chunk_size), normalizer
    )
    validation_data = NormalizedMateDownDataset(
        MateDownSequenceDataset(validation_frames, chunk_size), normalizer
    )
    train_loader = DataLoader(
        train_data, batch_size=args.batch_size, shuffle=True, num_workers=0
    )
    validation_loader = DataLoader(
        validation_data, batch_size=args.batch_size, shuffle=False, num_workers=0
    )
    config = ProprioACTConfig(
        state_dim=runtime_config.history_steps * FRAME_STATE_DIM,
        action_dim=6,
        **act_values,
    )
    model = ProprioACT(config).to(args.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    best_validation = float("inf")
    global_step = 0
    for epoch in range(1, args.epochs + 1):
        model.train()
        train_loss = 0.0
        for batch in train_loader:
            states = batch["observation.state"].to(args.device)
            actions = batch["action"].to(args.device)
            padding = batch["action_is_pad"].to(args.device)
            result = model(states, actions, padding)
            optimizer.zero_grad(set_to_none=True)
            result.loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_loss += float(result.loss.detach())
            global_step += 1
        validation_loss = _validate(model, validation_loader, args.device)
        mean_train = train_loss / max(1, len(train_loader))
        print(
            f"epoch={epoch:03d} train={mean_train:.6f} "
            f"validation={validation_loss:.6f}"
        )
        if validation_loss < best_validation:
            best_validation = validation_loss
            save_checkpoint(output, model, normalizer, optimizer, global_step)
    print(f"best checkpoint: {output} (validation={best_validation:.6f})")


@torch.inference_mode()
def _validate(model, loader, device):
    model.eval()
    total = 0.0
    for batch in loader:
        result = model(
            batch["observation.state"].to(device),
            batch["action"].to(device),
            batch["action_is_pad"].to(device),
        )
        total += float(result.loss)
    return total / max(1, len(loader))


if __name__ == "__main__":
    main()
