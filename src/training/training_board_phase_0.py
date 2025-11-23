# src/training/training_board_phase_0.py

import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F
from tqdm.auto import tqdm  # <-- NEW

from src.data.addition_algo import BoardConfig
from src.data.problems import generate_problems, generate_diversified_problems
from src.data.board_dataset import BlackboardAdditionStepDataset
from src.models.blackboard_transformer import BlackboardTransformer

from src.data.cot_dataset import CoTAdditionDataset, COT_VOCAB_TOKENS
from src.data.sampler import BucketBatchSampler
from src.models.blackboard_transformer import COTTransformer


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


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    # ----- Configs -----
    cfg = BoardConfig(H=4, W=5, n_digits=3)
    vocab_size = 12   # 0-9, PLUS, BLANK

    n_train_problems = 500_000
    n_val_problems = 2000
    batch_size = 64
    num_epochs = 5
    lr = 3e-4

    # ----- Data -----
    train_problems = generate_diversified_problems(cfg, n_train_problems, seed=0)
    val_problems   = generate_diversified_problems(cfg, n_val_problems,   seed=1)

    train_ds = BlackboardAdditionStepDataset(train_problems)
    val_ds   = BlackboardAdditionStepDataset(val_problems)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False)

    # ----- Model -----
    model = BlackboardTransformer(
        vocab_size=vocab_size,
        d_model=128,
        nhead=4,
        num_layers=3,
        dim_feedforward=512,
        max_len=cfg.H * cfg.W,
        dropout=0.1,
    ).to(device)

    # >>> print number of parameters <<<
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model parameters: {total_params:,} total | {trainable_params:,} trainable")
    print(f"≈ {trainable_params/1e6:.3f}M trainable parameters\n")

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # history for later inspection / plotting
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

    # ----- Epoch 0: evaluate random (untrained) model -----
    model.eval()

    # --- train set baseline ---
    init_train_loss = 0.0
    init_train_tokens = 0
    init_train_correct = 0
    init_train_carry_correct = 0
    init_train_carry_tokens = 0
    init_train_digit_correct = 0
    init_train_digit_tokens = 0

    pbar_init_train = tqdm(train_loader, desc="Epoch 0 [train, random]")
    with torch.no_grad():
        for batch in pbar_init_train:
            input_ids  = batch["input_ids"].to(device)
            target_ids = batch["target_ids"].to(device)
            mask       = batch["mask"].to(device)

            logits, _ = model(input_ids)

            loss = masked_cross_entropy(logits, target_ids, mask)
            batch_tokens = mask.sum().item()
            batch_loss = loss.item()

            (b_total_correct,
             b_total_tokens,
             b_carry_correct,
             b_carry_tokens,
             b_digit_correct,
             b_digit_tokens) = accuracy_with_splits(
                logits, target_ids, mask, H=cfg.H, W=cfg.W
            )

            init_train_loss += batch_loss * batch_tokens
            init_train_tokens += batch_tokens
            init_train_correct += b_total_correct
            init_train_carry_correct += b_carry_correct
            init_train_carry_tokens += b_carry_tokens
            init_train_digit_correct += b_digit_correct
            init_train_digit_tokens += b_digit_tokens

            batch_acc = b_total_correct / max(b_total_tokens, 1)
            pbar_init_train.set_postfix(loss=batch_loss, acc=batch_acc)

    init_train_avg_loss = init_train_loss / max(init_train_tokens, 1)
    init_train_avg_acc  = init_train_correct / max(init_train_tokens, 1)
    init_train_carry_acc = (
        init_train_carry_correct / init_train_carry_tokens
        if init_train_carry_tokens > 0 else 0.0
    )
    init_train_digit_acc = (
        init_train_digit_correct / init_train_digit_tokens
        if init_train_digit_tokens > 0 else 0.0
    )

    # --- val set baseline ---
    init_val_loss = 0.0
    init_val_tokens = 0
    init_val_correct = 0
    init_val_carry_correct = 0
    init_val_carry_tokens = 0
    init_val_digit_correct = 0
    init_val_digit_tokens = 0

    pbar_init_val = tqdm(val_loader, desc="Epoch 0 [val, random]")
    with torch.no_grad():
        for batch in pbar_init_val:
            input_ids  = batch["input_ids"].to(device)
            target_ids = batch["target_ids"].to(device)
            mask       = batch["mask"].to(device)

            logits, _ = model(input_ids)

            loss = masked_cross_entropy(logits, target_ids, mask)
            batch_tokens = mask.sum().item()
            batch_loss = loss.item()

            (b_total_correct,
             b_total_tokens,
             b_carry_correct,
             b_carry_tokens,
             b_digit_correct,
             b_digit_tokens) = accuracy_with_splits(
                logits, target_ids, mask, H=cfg.H, W=cfg.W
            )

            init_val_loss += batch_loss * batch_tokens
            init_val_tokens += batch_tokens
            init_val_correct += b_total_correct
            init_val_carry_correct += b_carry_correct
            init_val_carry_tokens += b_carry_tokens
            init_val_digit_correct += b_digit_correct
            init_val_digit_tokens += b_digit_tokens

            batch_acc = b_total_correct / max(b_total_tokens, 1)
            pbar_init_val.set_postfix(loss=batch_loss, acc=batch_acc)

    init_val_avg_loss = init_val_loss / max(init_val_tokens, 1)
    init_val_avg_acc  = init_val_correct / max(init_val_tokens, 1)
    init_val_carry_acc = (
        init_val_carry_correct / init_val_carry_tokens
        if init_val_carry_tokens > 0 else 0.0
    )
    init_val_digit_acc = (
        init_val_digit_correct / init_val_digit_tokens
        if init_val_digit_tokens > 0 else 0.0
    )

    print(
        f"\nEpoch 0 (random model) "
        f"| train loss/token: {init_train_avg_loss:.4f} "
        f"| train acc(masked): {init_train_avg_acc:.4f} "
        f"| train carry acc: {init_train_carry_acc:.4f} "
        f"| train digit acc: {init_train_digit_acc:.4f}"
    )
    print(
        f"Epoch 0 (random model) "
        f"| val   loss/token: {init_val_avg_loss:.4f} "
        f"| val   acc(masked): {init_val_avg_acc:.4f} "
        f"| val   carry acc: {init_val_carry_acc:.4f} "
        f"| val   digit acc: {init_val_digit_acc:.4f}"
    )
    print("=" * 80)

    # ----- Training loop -----
    for epoch in range(1, num_epochs + 1):
        model.train()
        total_loss = 0.0
        total_tokens = 0
        total_correct = 0
        total_carry_correct = 0
        total_carry_tokens = 0
        total_digit_correct = 0
        total_digit_tokens = 0

        # tqdm over training loader
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{num_epochs} [train]")
        for batch in pbar:
            input_ids  = batch["input_ids"].to(device)   # (B, L)
            target_ids = batch["target_ids"].to(device)  # (B, L)
            mask       = batch["mask"].to(device)        # (B, L)

            optimizer.zero_grad()
            logits, _ = model(input_ids)                 # (B, L, V)

            loss = masked_cross_entropy(logits, target_ids, mask)
            loss.backward()
            optimizer.step()

            batch_tokens = mask.sum().item()
            batch_loss = loss.item()

            (b_total_correct,
             b_total_tokens,
             b_carry_correct,
             b_carry_tokens,
             b_digit_correct,
             b_digit_tokens) = accuracy_with_splits(
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

        # ----- Validation -----
        model.eval()
        val_loss = 0.0
        val_tokens = 0
        val_correct = 0
        val_carry_correct = 0
        val_carry_tokens = 0
        val_digit_correct = 0
        val_digit_tokens = 0

        pbar_val = tqdm(val_loader, desc=f"Epoch {epoch}/{num_epochs} [val] ")
        with torch.no_grad():
            for batch in pbar_val:
                input_ids  = batch["input_ids"].to(device)
                target_ids = batch["target_ids"].to(device)
                mask       = batch["mask"].to(device)

                logits, _ = model(input_ids)

                loss = masked_cross_entropy(logits, target_ids, mask)

                batch_tokens = mask.sum().item()
                batch_loss = loss.item()

                (b_total_correct,
                 b_total_tokens,
                 b_carry_correct,
                 b_carry_tokens,
                 b_digit_correct,
                 b_digit_tokens) = accuracy_with_splits(
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

    # optional: print the whole history at the end
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



def main2():

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    # ----- Configs -----
    cfg = BoardConfig(H=4, W=5, n_digits=3)

    n_train_problems = 40000
    n_val_problems = 2000
    batch_size = 64
    num_epochs = 5
    lr = 3e-4

    # ----- Data -----
    train_problems = generate_diversified_problems(cfg, n_train_problems, seed=0)
    val_problems   = generate_diversified_problems(cfg, n_val_problems,   seed=1)

    train_ds = CoTAdditionDataset(train_problems)
    val_ds   = CoTAdditionDataset(val_problems)

    train_loader = DataLoader(train_ds, batch_sampler=BucketBatchSampler(train_ds, batch_size=batch_size, shuffle=True))
    val_loader   = DataLoader(val_ds,  batch_sampler=BucketBatchSampler(train_ds, batch_size=batch_size, shuffle=False))

    # ----- Model -----
    model = COTTransformer(
        vocab_size= len(COT_VOCAB_TOKENS),
        d_model=128,
        nhead=4,
        num_layers=3,
        dim_feedforward=512,
        max_len=200, #unsure what to set,
        dropout=0.1,
    ).to(device)

    # >>> print number of parameters <<<
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

        # tqdm over training loader
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{num_epochs} [train]")

        for batch in pbar:
            
            input_ids  = batch["input_ids"].to(device)   # (B,  L_prefix + L_step, ) 
            target_ids = batch["label_ids"].to(device)  # (B, L_prefix + L_step,)
            loss_mask = batch["loss_mask"].to(device)   # (B, L_prefix + L_step)
            attn_mask = batch["attn_mask"][0].to(device) # one is enough grouping samples in a batch that correspond to same step


            optimizer.zero_grad()
            logits, _ = model(input_ids, src_mask=attn_mask) # (B, L, V)

            loss = masked_cross_entropy(logits, target_ids, loss_mask)
            loss.backward()
            optimizer.step()

            batch_tokens = mask.sum().item()
            batch_loss = loss.item()

            pred_ids = logits.argmax(dim=-1)         # (B, L)
            correct = (pred_ids == target_ids)       # (B, L) bool

            b_total_correct = (correct & mask).sum().item()

            indices = mask[0].nonzero(as_tuple=True)[0]

            digit_pos = indices[0] 
            carry_pos = indices[1]

            digit_mask = torch.zeros_like(mask).bool()
            digit_mask[:, digit_pos] = True

            carry_mask = torch.zeros_like(mask).bool()
            carry_mask[:, carry_pos] = True

            b_digit_correct = (correct & digit_mask).sum().item()
            b_digit_tokens  = digit_mask.sum().item()

            b_carry_correct = (correct & carry_mask).sum().item()
            b_carry_tokens  = carry_mask.sum().item()


            total_loss += batch_loss * batch_tokens
            total_tokens += batch_tokens
            total_correct += b_total_correct
            total_carry_correct += b_carry_correct
            total_carry_tokens += b_carry_tokens
            total_digit_correct += b_digit_correct
            total_digit_tokens += b_digit_tokens

            batch_acc = b_total_correct / max(batch_tokens, 1)
            pbar.set_postfix(loss=batch_loss, acc=batch_accc)
        
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

        # Evaluation

        model.eval()
        val_loss = 0.0
        val_tokens = 0
        val_correct = 0
        val_carry_correct = 0
        val_carry_tokens = 0
        val_digit_correct = 0
        val_digit_tokens = 0

        pbar_val = tqdm(val_loader, desc=f"Epoch {epoch}/{num_epochs} [val] ")
        with torch.no_grad():
            for batch in pbar_val:
                input_ids  = batch["input_ids"].to(device)   # (B,  L_prefix + L_step, ) 
                target_ids = batch["label_ids"].to(device)  # (B, L_prefix + L_step,)
                loss_mask = batch["loss_mask"].to(device)   # (B, L_prefix + L_step)
                attn_mask = batch["attn_mask"][0].to(device) # one is enough grouping samples in a batch that correspond to same step


                optimizer.zero_grad()
                logits, _ = model(input_ids, src_mask=attn_mask) # (B, L, V)

                loss = masked_cross_entropy(logits, target_ids, loss_mask)
                loss.backward()
                optimizer.step()

                batch_tokens = mask.sum().item()
                batch_loss = loss.item()

                pred_ids = logits.argmax(dim=-1)         # (B, L)
                correct = (pred_ids == target_ids)       # (B, L) bool

                b_total_correct = (correct & mask).sum().item()

                indices = mask[0].nonzero(as_tuple=True)[0]

                digit_pos = indices[0] 
                carry_pos = indices[1]

                digit_mask = torch.zeros_like(mask).bool()
                digit_mask[:, digit_pos] = True

                carry_mask = torch.zeros_like(mask).bool()
                carry_mask[:, carry_pos] = True

                b_digit_correct = (correct & digit_mask).sum().item()
                b_digit_tokens  = digit_mask.sum().item()

                b_carry_correct = (correct & carry_mask).sum().item()
                b_carry_tokens  = carry_mask.sum().item()


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

    # optional: print the whole history at the end
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



if __name__ == "__main__":
    main()
   #main2()
