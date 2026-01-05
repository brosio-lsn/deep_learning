# src/training/training_board_phase_0.py
import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F
from tqdm.auto import tqdm  # <-- NEW

from src.data.addition_algo import BoardConfig
from src.data.problems import generate_problems, generate_diversified_problems
from src.data.board_dataset import BlackboardAdditionStepDataset
from src.models.transformers import BlackboardTransformer

from src.data.cot_dataset import CoTAdditionDataset, COT_VOCAB_TOKENS, ID2TOK
from src.data.sampler import BucketBatchSampler
from src.models.transformers import COTTransformer
from src.models.positional_encodings import *


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

def evaluate_board_model(model, data_loader, cfg, device, desc):
    model.eval()
    val_loss = 0.0
    val_tokens = 0
    val_correct = 0
    val_carry_correct = 0
    val_carry_tokens = 0
    val_digit_correct = 0
    val_digit_tokens = 0

    pbar_val = tqdm(data_loader, desc=desc)
    with torch.no_grad():
        for batch in pbar_val:
            input_ids  = batch["input_ids"].to(device)
            target_ids = batch["target_ids"].to(device)
            mask       = batch["mask"].to(device)

            logits, _ = model(input_ids)

            loss = masked_cross_entropy(logits, target_ids, mask)

            batch_tokens = mask.sum().item()
            batch_loss = loss.item()

            (
                b_total_correct,
                b_total_tokens,
                b_carry_correct,
                b_carry_tokens,
                b_digit_correct,
                b_digit_tokens,
            ) = accuracy_with_splits(
                logits, target_ids, mask, H=cfg.H, W=cfg.W
            )

            val_loss += batch_loss * batch_tokens
            val_tokens += batch_tokens
            val_correct += b_total_correct
            val_carry_correct += b_carry_correct
            val_carry_tokens += b_carry_tokens
            val_digit_correct += b_digit_correct
            val_digit_tokens += b_digit_tokens

            batch_acc = b_total_correct / max(b_total_tokens, 1)
            pbar_val.set_postfix(loss=batch_loss, acc=batch_acc)

    val_avg_loss = val_loss / max(val_tokens, 1)
    val_avg_acc  = val_correct / max(val_tokens, 1)
    val_carry_acc = (
        val_carry_correct / val_carry_tokens
        if val_carry_tokens > 0 else 0.0
    )
    val_digit_acc = (
        val_digit_correct / val_digit_tokens
        if val_digit_tokens > 0 else 0.0
    )

    return val_avg_loss, val_avg_acc, val_carry_acc, val_digit_acc


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    max_eval_digits = 9
    cfg = BoardConfig(H=4, W=max_eval_digits + 2, n_digits=3)
    vocab_size = 12

    n_train_problems = 500_000
    n_val_problems = 2000
    batch_size = 64
    num_epochs = 3
    lr = 3e-4

    train_problems = generate_diversified_problems(cfg, n_train_problems, seed=0)
    val_problems   = generate_diversified_problems(cfg, n_val_problems,   seed=1)

    train_ds = BlackboardAdditionStepDataset(train_problems)
    val_ds   = BlackboardAdditionStepDataset(val_problems)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False)

    d_model = 128
    max_len = cfg.H * cfg.W
    n_heads = 4

    pes = [
        ("Relative PE", RelativePositionBias2D(n_heads, cfg.H, cfg.W)),
        ("Sinusoidal PE", SinusoidalPositionalEncoding(d_model, max_len=max_len)),
        ("Absolute PE", AbsolutePositionalEncoding2D(d_model, cfg.H, cfg.W)),
        ("Abs+Rel PE",Abs2DPlusRelBias2D(
                abs_pe=AbsolutePositionalEncoding2D(d_model, cfg.H, cfg.W),
                rel_bias=RelativePositionBias2D(n_heads, cfg.H, cfg.W),
            ))
    ]

    for pe in pes:
        print(f"Starting {pe[0]} ...")

        model = BlackboardTransformer(
            vocab_size=vocab_size,
            d_model=d_model,
            nhead=n_heads,
            num_layers=3,
            dim_feedforward=512,
            max_len=max_len,
            dropout=0.1,
            pos_enc=pe[1],
        ).to(device)

        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"Model parameters: {total_params:,} total | {trainable_params:,} trainable")
        print(f"≈ {trainable_params/1e6:.3f}M trainable parameters\n")

        print("Precise Overview of Trainable Parameters:\n")
        for name, param in model.named_parameters():
            if param.requires_grad:
                print(name, param.shape)

        optimizer = torch.optim.Adam(model.parameters(), lr=lr)

        history = {
            "train_loss": [],
            "train_acc": [],
            "train_carry_acc": [],
            "train_digit_acc": [],
            "val_loss": [],
            "val_acc": [],
            "val_carry_acc": [],
            "val_digit_acc": [],
        }

        model.train()

        for epoch in range(1, num_epochs + 1):
            model.train()
            total_loss = 0.0
            total_tokens = 0
            total_correct = 0
            total_carry_correct = 0
            total_carry_tokens = 0
            total_digit_correct = 0
            total_digit_tokens = 0

            pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{num_epochs} [train]")
            for batch in pbar:
                input_ids  = batch["input_ids"].to(device)
                target_ids = batch["target_ids"].to(device)
                mask       = batch["mask"].to(device)

                optimizer.zero_grad()
                logits, _ = model(input_ids)

                loss = masked_cross_entropy(logits, target_ids, mask)
                loss.backward()
                optimizer.step()

                batch_tokens = mask.sum().item()
                batch_loss = loss.item()

                (
                    b_total_correct,
                    b_total_tokens,
                    b_carry_correct,
                    b_carry_tokens,
                    b_digit_correct,
                    b_digit_tokens,
                ) = accuracy_with_splits(
                    logits, target_ids, mask, H=cfg.H, W=cfg.W
                )

                total_loss += batch_loss * batch_tokens
                total_tokens += batch_tokens
                total_correct += b_total_correct
                total_carry_correct += b_carry_correct
                total_carry_tokens += b_carry_tokens
                total_digit_correct += b_digit_correct
                total_digit_tokens += b_digit_tokens

                batch_acc = b_total_correct / max(b_total_tokens, 1)
                pbar.set_postfix(loss=batch_loss, acc=batch_acc)

            avg_loss = total_loss / max(total_tokens, 1)
            avg_acc  = total_correct / max(total_tokens, 1)
            carry_acc = (
                total_carry_correct / total_carry_tokens
                if total_carry_tokens > 0 else 0.0
            )
            digit_acc = (
                total_digit_correct / total_digit_tokens
                if total_digit_tokens > 0 else 0.0
            )

            history["train_loss"].append(avg_loss)
            history["train_acc"].append(avg_acc)
            history["train_carry_acc"].append(carry_acc)
            history["train_digit_acc"].append(digit_acc)

            val_avg_loss, val_avg_acc, val_carry_acc, val_digit_acc = evaluate_board_model(
                model,
                val_loader,
                cfg,
                device,
                desc=f"Epoch {epoch}/{num_epochs} [val]",
            )

            history["val_loss"].append(val_avg_loss)
            history["val_acc"].append(val_avg_acc)
            history["val_carry_acc"].append(val_carry_acc)
            history["val_digit_acc"].append(val_digit_acc)

            print(
                f"\nEpoch {epoch}/{num_epochs} "
                f"| train loss/token: {avg_loss:.4f} "
                f"| train acc(masked): {avg_acc:.4f} "
                f"| train carry acc: {carry_acc:.4f} "
                f"| train digit acc: {digit_acc:.4f}"
            )
            print(
                f"Epoch {epoch}/{num_epochs} "
                f"| val   loss/token: {val_avg_loss:.4f} "
                f"| val   acc(masked): {val_avg_acc:.4f} "
                f"| val   carry acc: {val_carry_acc:.4f} "
                f"| val   digit acc: {val_digit_acc:.4f}"
            )
            print("-" * 80)

        print("Training history:")
        for e in range(num_epochs):
            print(
                f"Epoch {e+1}: "
                f"train_loss={history['train_loss'][e]:.4f}, "
                f"train_acc={history['train_acc'][e]:.4f}, "
                f"train_carry_acc={history['train_carry_acc'][e]:.4f}, "
                f"train_digit_acc={history['train_digit_acc'][e]:.4f}, "
                f"val_loss={history['val_loss'][e]:.4f}, "
                f"val_acc={history['val_acc'][e]:.4f}, "
                f"val_carry_acc={history['val_carry_acc'][e]:.4f}, "
                f"val_digit_acc={history['val_digit_acc'][e]:.4f}"
            )

        digit_cfgs = {
            3: BoardConfig(H=cfg.H, W=cfg.W, n_digits=3),
            5: BoardConfig(H=cfg.H, W=cfg.W, n_digits=5),
            7: BoardConfig(H=cfg.H, W=cfg.W, n_digits=7),
        }
        gen_results = {}
        for n_digits, cfg_eval in digit_cfgs.items():
            if n_digits == 3:
                eval_loader = val_loader
            else:
                eval_problems = generate_diversified_problems(
                    cfg_eval, n_val_problems, seed=10 + n_digits
                )
                eval_ds = BlackboardAdditionStepDataset(eval_problems)
                eval_loader = DataLoader(eval_ds, batch_size=batch_size, shuffle=False)

            eval_loss, eval_acc, eval_carry_acc, eval_digit_acc = evaluate_board_model(
                model,
                eval_loader,
                cfg_eval,
                device,
                desc=f"{pe[0]} {n_digits}-digit eval",
            )
            gen_results[n_digits] = {
                "loss": eval_loss,
                "acc": eval_acc,
                "carry_acc": eval_carry_acc,
                "digit_acc": eval_digit_acc,
            }

        print("Generalization results:")
        for n_digits in sorted(gen_results.keys()):
            r = gen_results[n_digits]
            print(
                f"{n_digits}-digit | loss: {r['loss']:.4f} "
                f"| acc: {r['acc']:.4f} "
                f"| carry acc: {r['carry_acc']:.4f} "
                f"| digit acc: {r['digit_acc']:.4f}"
            )

        digits_list = sorted(gen_results.keys())
        accs = [gen_results[d]["acc"] for d in digits_list]
        carry_accs = [gen_results[d]["carry_acc"] for d in digits_list]
        digit_accs = [gen_results[d]["digit_acc"] for d in digits_list]

        plt.figure()
        plt.plot(digits_list, accs, marker="o", label="overall")
        plt.plot(digits_list, carry_accs, marker="o", label="carry row")
        plt.plot(digits_list, digit_accs, marker="o", label="digit row")
        plt.xlabel("Number of digits")
        plt.ylabel("Accuracy")
        plt.ylim(0.0, 1.05)
        plt.title(f"{pe[0]} - length generalization")
        plt.legend()
        plt.grid(True)
        filename = pe[0].lower().replace(" ", "_")
        plt.savefig(f"length_generalization_{filename}.png")
        plt.close()


if __name__ == "__main__":
    main()


