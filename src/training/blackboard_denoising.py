"""
Training for Direction 2 (BERT-style denoising)
Same setting but with denoising objective

Settings:
  - local  (Setting 2): model may only "fix" last-step cells + current step cells
  - global (Setting 3): model may "fix" any editable cell (carry/result rows)

Usage (example):
  python -m src.training.blackboard_denoising --setting local --denoise-rate 0.15 --p-revert 0.25
"""

from dataclasses import asdict
import argparse
import os

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from src.data.addition_algo import BoardConfig, VOCAB_SIZE
from src.data.problems import generate_diversified_problems
from src.data.board_dataset import BlackboardAdditionDenoisingStepDataset
from src.models.positional_encodings import *
from src.models.pe_factory import make_pes
from src.models.transformers import BlackboardTransformer
from src.training.configs import ModelConfig, TrainConfig
from src.training.trainers import BlackboardTrainer


def masked_cross_entropy(logits, target_ids, mask):
    """
    logits: (B, L, V)
    target_ids: (B, L)
    mask: (B, L) bool, True where we train

    Returns CE averaged over masked positions only.
    """
    vocab_size = logits.size(-1)

    logits_flat = logits.reshape(-1, vocab_size)   # (B*L, V)
    targets_flat = target_ids.reshape(-1)          # (B*L,)
    mask_flat = mask.reshape(-1)                   # (B*L,)

    logits_sel = logits_flat[mask_flat]
    targets_sel = targets_flat[mask_flat]

    return F.cross_entropy(logits_sel, targets_sel)


def accuracy_with_splits(
    logits: torch.Tensor,
    target_ids: torch.Tensor,
    mask: torch.Tensor,
    H: int,
    W: int,
    carry_row: int = 0,
    digit_row: int | None = None,
):
    if digit_row is None:
        digit_row = H - 1

    B, L, V = logits.shape
    device = logits.device

    preds = logits.argmax(dim=-1)           # (B, L)
    correct = (preds == target_ids) & mask  # (B, L)

    total_tokens = mask.sum().item()
    total_correct = correct.sum().item()

    positions = torch.arange(L, device=device)
    row_idx = positions // W

    carry_pos = (row_idx == carry_row).unsqueeze(0).expand_as(mask)
    digit_pos = (row_idx == digit_row).unsqueeze(0).expand_as(mask)

    carry_tokens = (mask & carry_pos).sum().item()
    digit_tokens = (mask & digit_pos).sum().item()

    carry_correct = (correct & carry_pos).sum().item()
    digit_correct = (correct & digit_pos).sum().item()

    return (
        total_correct,
        total_tokens,
        carry_correct,
        carry_tokens,
        digit_correct,
        digit_tokens,
    )


def model_cfg_name(cfg: ModelConfig) -> str:
    return (
        f"d{cfg.d_model}"
        f"_h{cfg.nhead}"
        f"_L{cfg.num_layers}"
        f"_ff{cfg.dim_feedforward}"
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--setting", choices=["local", "global"], default="local")
    parser.add_argument("--denoise-rate", type=float, default=0.15)

    # Mode B (explicit revert/backtracking)
    parser.add_argument(
        "--p-revert",
        type=float,
        default=0.25,
        help="Probability of revert samples (Mode B). 0.0 => pure denoising (Mode A).",
    )
    parser.add_argument(
        "--keep-stepwrite-on-revert",
        action="store_true",
        help="If set, still supervise stepwrite positions on revert samples (default: revert-only).",
    )

    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out-dir", type=str, default="models")
    parser.add_argument("--n-train", type=int, default=40_000)
    parser.add_argument("--n-val", type=int, default=10_000)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    board_cfg = BoardConfig(H=4, W=5, n_digits=3)
    max_len = board_cfg.H * board_cfg.W

    train_cfg = TrainConfig(
        batch_size=64,
        num_epochs=10,
        lr=3e-4,
        log_interval=0.1,
        enable_docs=True,
        save_model=True,
        seed=args.seed,
        out_dir=args.out_dir,
        exp_name="",
    )

    model_cfg = ModelConfig(
        d_model=128,
        nhead=4,
        num_layers=4,
        dim_feedforward=512,
        dropout=0.1,
        max_len=200,
    )

    # Data
    train_problems = generate_diversified_problems(board_cfg, args.n_train, seed=0)
    val_problems = generate_diversified_problems(board_cfg, args.n_val, seed=1)

    train_ds = BlackboardAdditionDenoisingStepDataset(
        train_problems,
        setting=args.setting,
        denoise_rate=args.denoise_rate,
        p_revert=args.p_revert,
        disable_stepwrite_on_revert=(not args.keep_stepwrite_on_revert),
        seed=args.seed,
    )
    val_ds = BlackboardAdditionDenoisingStepDataset(
        val_problems,
        setting=args.setting,
        denoise_rate=args.denoise_rate,
        p_revert=args.p_revert,
        disable_stepwrite_on_revert=(not args.keep_stepwrite_on_revert),
        seed=args.seed + 1,
    )

    train_loader = DataLoader(train_ds, batch_size=train_cfg.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=train_cfg.batch_size, shuffle=False)

    # Positional encodings
    pes = make_pes(model_cfg, board_cfg)

    trainer = BlackboardTrainer(
        model=None,
        optimizer=None,
        device=device,
        train_loader=train_loader,
        val_loader=val_loader,
        train_cfg=None,
        board_cfg=board_cfg,
        accuracy_with_splits=accuracy_with_splits,
        masked_cross_entropy_fn=masked_cross_entropy,
    )

    for pe_name, pe in pes:
        cfg_name = model_cfg_name(model_cfg)
        train_cfg.exp_name = os.path.join(
            "direction2_denoising",
            args.setting,
            f"rate{args.denoise_rate}",
            f"prevert{args.p_revert}",
            pe_name,
            cfg_name,
        )

        print(
            f"\n===== Direction2 | {args.setting} | denoise={args.denoise_rate} | p_revert={args.p_revert} | {pe_name} | {cfg_name} ====="
        )

        model = BlackboardTransformer(
            vocab_size=VOCAB_SIZE,
            pos_enc=pe,
            **asdict(model_cfg),
        ).to(device)

        optimizer = torch.optim.Adam(model.parameters(), lr=train_cfg.lr)

        trainer.model = model
        trainer.optimizer = optimizer
        trainer.train_cfg = train_cfg

        trainer.fit()


if __name__ == "__main__":
    main()
