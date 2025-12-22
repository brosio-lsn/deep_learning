from src.training.configs import ModelConfig, TrainConfig
import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F
from tqdm.auto import tqdm  

from src.training.trainers import BlackboardTrainer
from src.data.addition_algo import BoardConfig
from src.data.problems import generate_problems, generate_diversified_problems
from src.models.positional_encodings import *
from src.data.board_dataset import BlackboardAdditionStepDataset
from src.models.transformers import BlackboardTransformer


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
    """
    Compute overall accuracy, plus accuracies restricted to:
      - carry row (smallest row index, by default 0)
      - 'next digit' row (by default last row, H-1)

    logits: (B, L, V)
    target_ids: (B, L)
    mask: (B, L) bool
    """
    if digit_row is None:
        digit_row = H - 1

    B, L, V = logits.shape
    device = logits.device

    preds = logits.argmax(dim=-1)          # (B, L)
    correct = (preds == target_ids) & mask  # (B, L) bool

    total_tokens = mask.sum().item()
    total_correct = correct.sum().item()

    # Row indices for each sequence position (same for all batch elements)
    positions = torch.arange(L, device=device)
    row_idx = positions // W  # (L,)

    # Masks for carry row (row 0) and 'next digit' row (row H-1 by default)
    carry_pos = (row_idx == carry_row).unsqueeze(0).expand_as(mask)   # (B, L)
    digit_pos = (row_idx == digit_row).unsqueeze(0).expand_as(mask)   # (B, L)

    # Effective masks (still intersected with the training mask via 'correct')
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


if __name__ == "__main__":

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)


    board_cfg = BoardConfig(H=4, W=5, n_digits=3)
    vocab_size = 12
    max_len = board_cfg.H * board_cfg.W

    n_train = 500_000
    n_val = 2_000

    model_cfg = ModelConfig(
        d_model = 128,
        nhead = 4,
        num_layers = 3,
        dim_feedforward = 512,
        dropout = 0.1,
        max_len = 200
    )

    train_cfg = TrainConfig(
        batch_size=64,
        num_epochs=5,
        lr=3e-4,
        log_interval=0.1,
        enable_docs=True,
        enable_plots=True,
        save_model=True,
        seed=0,
        out_dir="models",
        exp_name="blackboard_addition",
    )


    train_problems = generate_diversified_problems(board_cfg, n_train, seed=0)
    val_problems   = generate_diversified_problems(board_cfg, n_val,   seed=1)

    train_ds = BlackboardAdditionStepDataset(train_problems)
    val_ds   = BlackboardAdditionStepDataset(val_problems)

    train_loader = DataLoader(
        train_ds,
        batch_size=train_cfg.batch_size,
        shuffle=True,
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=train_cfg.batch_size,
        shuffle=False,
    )


    pes = [
        ("relative_pe", RelativePositionBias2D(model_cfg.nhead, board_cfg.H, board_cfg.W)),
        ("sinusoidal_pe", SinusoidalPositionalEncoding(model_cfg.d_model, model_cfg.max_len)),
        ("absolute_pe", AbsolutePositionalEncoding2D(model_cfg.d_model, board_cfg.H, board_cfg.W)),
    ]


    for pe_name, pe in pes:
        print(f"\n===== Starting experiment: {pe_name} =====")

        train_cfg.exp_name = f"blackboard_addition_{pe_name}"

        model = BlackboardTransformer(
            vocab_size=12,
            pos_enc=pe,
            **asdict(model_cfg)
        ).to(device)

        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"Trainable params: {trainable_params/1e6:.3f}M")
        print(model_cfg)

        optimizer = torch.optim.Adam(model.parameters(), lr=train_cfg.lr)

        trainer = BlackboardTrainer(
            model=model,
            optimizer=optimizer,
            device=device,
            train_loader=train_loader,
            val_loader=val_loader,
            train_cfg=train_cfg,
            board_cfg=board_cfg,
            accuracy_with_splits=accuracy_with_splits,
            masked_cross_entropy_fn=masked_cross_entropy
        )

        trainer.fit()