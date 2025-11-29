# training_local_generalization.py

import math
import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F
from tqdm.auto import tqdm
import matplotlib.pyplot as plt

from src.data.addition_algo import BoardConfig
from src.data.problems import generate_diversified_problems
from src.data.board_dataset import BlackboardAdditionStepDataset
from src.models.blackboard_transformer import BlackboardTransformer
from src.models.positional_encodings import (
    RelativePositionBias2D,
    SinusoidalPositionalEncoding,
    AbsolutePositionalEncoding2D,
)


def masked_cross_entropy(logits, target_ids, mask):
    vocab_size = logits.size(-1)
    logits_flat = logits.reshape(-1, vocab_size)
    targets_flat = target_ids.reshape(-1)
    mask_flat = mask.reshape(-1)

    logits_sel = logits_flat[mask_flat]
    targets_sel = targets_flat[mask_flat]

    return F.cross_entropy(logits_sel, targets_sel)


def accuracy_with_splits(
    logits: torch.Tensor,
    target_ids: torch.Tensor,
    mask: torch.Tensor,
    H: int,
    W: int,
    carry_row: int,
    digit_row: int,
):
    if digit_row is None:
        digit_row = H - 1

    B, L, V = logits.shape
    device = logits.device

    preds = logits.argmax(dim=-1)
    correct = (preds == target_ids) & mask

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
            input_ids = batch["input_ids"].to(device)
            target_ids = batch["target_ids"].to(device)
            mask = batch["mask"].to(device)

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
                logits,
                target_ids,
                mask,
                H=cfg.H,
                W=cfg.W,
                carry_row=cfg.carry_row,
                digit_row=cfg.result_row,
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
    val_avg_acc = val_correct / max(val_tokens, 1)
    val_carry_acc = (
        val_carry_correct / val_carry_tokens if val_carry_tokens > 0 else 0.0
    )
    val_digit_acc = (
        val_digit_correct / val_digit_tokens if val_digit_tokens > 0 else 0.0
    )

    return val_avg_loss, val_avg_acc, val_carry_acc, val_digit_acc


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    H_total = 8
    W = 5
    n_digits = 3
    train_offset = 1

    cfg_train = BoardConfig(
        H=H_total,
        W=W,
        n_digits=n_digits,
        carry_row=train_offset,
        top_row=train_offset + 1,
        bottom_row=train_offset + 2,
        result_row=train_offset + 3,
    )

    vocab_size = 12

    n_train_problems = 500_000
    n_val_problems = 2_000
    batch_size = 64
    num_epochs = 2
    lr = 3e-4

    train_problems = generate_diversified_problems(
        cfg_train, n_train_problems, seed=0
    )
    val_problems = generate_diversified_problems(cfg_train, n_val_problems, seed=1)

    train_ds = BlackboardAdditionStepDataset(train_problems)
    val_ds = BlackboardAdditionStepDataset(val_problems)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    d_model = 128
    max_len = cfg_train.H * cfg_train.W
    n_heads = 4

    pos_encs = [
        ("Relative PE", RelativePositionBias2D(n_heads, cfg_train.H, cfg_train.W)),
        # (
        #     "Sinusoidal PE",
        #     SinusoidalPositionalEncoding(d_model, max_len=max_len),
        # ),
        # ("Absolute PE", AbsolutePositionalEncoding2D(d_model, cfg_train.H, cfg_train.W)),
    ]

    for pe_name, pe_module in pos_encs:
        print(f"Starting location experiment for {pe_name} ...")

        model = BlackboardTransformer(
            vocab_size=vocab_size,
            d_model=d_model,
            nhead=n_heads,
            num_layers=3,
            dim_feedforward=512,
            max_len=max_len,
            dropout=0.1,
            pos_enc=pe_module,
        ).to(device)

        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"Model parameters: {total_params:,} total | {trainable_params:,} trainable")
        print(f"≈ {trainable_params/1e6:.3f}M trainable parameters\n")

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

        for epoch in range(1, num_epochs + 1):
            model.train()
            total_loss = 0.0
            total_tokens = 0
            total_correct = 0
            total_carry_correct = 0
            total_carry_tokens = 0
            total_digit_correct = 0
            total_digit_tokens = 0

            pbar = tqdm(train_loader, desc=f"{pe_name} Epoch {epoch}/{num_epochs} [train]")
            for batch in pbar:
                input_ids = batch["input_ids"].to(device)
                target_ids = batch["target_ids"].to(device)
                mask = batch["mask"].to(device)

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
                    logits,
                    target_ids,
                    mask,
                    H=cfg_train.H,
                    W=cfg_train.W,
                    carry_row=cfg_train.carry_row,
                    digit_row=cfg_train.result_row,
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
            avg_acc = total_correct / max(total_tokens, 1)
            carry_acc = (
                total_carry_correct / total_carry_tokens
                if total_carry_tokens > 0
                else 0.0
            )
            digit_acc = (
                total_digit_correct / total_digit_tokens
                if total_digit_tokens > 0
                else 0.0
            )

            history["train_loss"].append(avg_loss)
            history["train_acc"].append(avg_acc)
            history["train_carry_acc"].append(carry_acc)
            history["train_digit_acc"].append(digit_acc)

            (
                val_avg_loss,
                val_avg_acc,
                val_carry_acc,
                val_digit_acc,
            ) = evaluate_board_model(
                model,
                val_loader,
                cfg_train,
                device,
                desc=f"{pe_name} Epoch {epoch}/{num_epochs} [val]",
            )

            history["val_loss"].append(val_avg_loss)
            history["val_acc"].append(val_avg_acc)
            history["val_carry_acc"].append(val_carry_acc)
            history["val_digit_acc"].append(val_digit_acc)

            print(
                f"\n{pe_name} Epoch {epoch}/{num_epochs} "
                f"| train loss/token: {avg_loss:.4f} "
                f"| train acc(masked): {avg_acc:.4f} "
                f"| train carry acc: {carry_acc:.4f} "
                f"| train digit acc: {digit_acc:.4f}"
            )
            print(
                f"{pe_name} Epoch {epoch}/{num_epochs} "
                f"| val   loss/token: {val_avg_loss:.4f} "
                f"| val   acc(masked): {val_avg_acc:.4f} "
                f"| val   carry acc: {val_carry_acc:.4f} "
                f"| val   digit acc: {val_digit_acc:.4f}"
            )
            print("-" * 80)

        print("Training history:")
        for e in range(num_epochs):
            print(
                f"{pe_name} Epoch {e+1}: "
                f"train_loss={history['train_loss'][e]:.4f}, "
                f"train_acc={history['train_acc'][e]:.4f}, "
                f"train_carry_acc={history['train_carry_acc'][e]:.4f}, "
                f"train_digit_acc={history['train_digit_acc'][e]:.4f}, "
                f"val_loss={history['val_loss'][e]:.4f}, "
                f"val_acc={history['val_acc'][e]:.4f}, "
                f"val_carry_acc={history['val_carry_acc'][e]:.4f}, "
                f"val_digit_acc={history['val_digit_acc'][e]:.4f}"
            )

        max_offset = H_total - 4
        offsets = list(range(0, max_offset + 1))

        global_results = {}
        for offset in offsets:
            cfg_eval = BoardConfig(
                H=H_total,
                W=W,
                n_digits=n_digits,
                carry_row=offset,
                top_row=offset + 1,
                bottom_row=offset + 2,
                result_row=offset + 3,
            )
            eval_problems = generate_diversified_problems(
                cfg_eval, n_val_problems, seed=100 + offset
            )
            eval_ds = BlackboardAdditionStepDataset(eval_problems)
            eval_loader = DataLoader(
                eval_ds, batch_size=batch_size, shuffle=False
            )

            eval_loss, eval_acc, eval_carry_acc, eval_digit_acc = evaluate_board_model(
                model,
                eval_loader,
                cfg_eval,
                device,
                desc=f"{pe_name} global shift offset={offset}",
            )
            global_results[offset] = {
                "loss": eval_loss,
                "acc": eval_acc,
                "carry_acc": eval_carry_acc,
                "digit_acc": eval_digit_acc,
            }

        print(f"\nGlobal shift results for {pe_name}:")
        for offset in offsets:
            r = global_results[offset]
            print(
                f"offset {offset} | loss: {r['loss']:.4f} "
                f"| acc: {r['acc']:.4f} "
                f"| carry acc: {r['carry_acc']:.4f} "
                f"| digit acc: {r['digit_acc']:.4f}"
            )

        rel_configs = {
            "bottom_down_1": BoardConfig(
                H=H_total,
                W=W,
                n_digits=n_digits,
                carry_row=train_offset,
                top_row=train_offset + 1,
                bottom_row=train_offset + 3,
                result_row=train_offset + 4,
            ),
        }

        print(f"\nRelative shift results for {pe_name}:")
        for name, cfg_rel in rel_configs.items():
            rel_problems = generate_diversified_problems(
                cfg_rel, n_val_problems, seed=200
            )
            rel_ds = BlackboardAdditionStepDataset(rel_problems)
            rel_loader = DataLoader(
                rel_ds, batch_size=batch_size, shuffle=False
            )
            (
                rel_loss,
                rel_acc,
                rel_carry_acc,
                rel_digit_acc,
            ) = evaluate_board_model(
                model,
                rel_loader,
                cfg_rel,
                device,
                desc=f"{pe_name} relative shift {name}",
            )
            print(
                f"{name} | loss: {rel_loss:.4f} "
                f"| acc: {rel_acc:.4f} "
                f"| carry acc: {rel_carry_acc:.4f} "
                f"| digit acc: {rel_digit_acc:.4f}"
            )

        pos_list = sorted(global_results.keys())
        accs = [global_results[o]["acc"] for o in pos_list]
        carry_accs = [global_results[o]["carry_acc"] for o in pos_list]
        digit_accs = [global_results[o]["digit_acc"] for o in pos_list]

        plt.figure()
        plt.plot(pos_list, accs, marker="o", label="overall")
        plt.plot(pos_list, carry_accs, marker="o", label="carry row")
        plt.plot(pos_list, digit_accs, marker="o", label="digit row")
        plt.xlabel("Global block vertical offset (carry row index)")
        plt.ylabel("Accuracy")
        plt.ylim(0.0, 1.05)
        plt.title(f"{pe_name} - location generalization")
        plt.legend()
        plt.grid(True)
        filename = pe_name.lower().replace(" ", "_")
        plt.savefig(f"local_generalization_{filename}.png")
        plt.close()


if __name__ == "__main__":
    main()
