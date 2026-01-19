# src/training/training_sudoku.py
import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F
from tqdm.auto import tqdm
import os

from src.data.sudoku_hf import load_hf_sudoku_problems
from src.data.board_dataset_sudoku import BlackboardSudokuStepDataset
from src.models.transformers import BlackboardTransformer
from src.models.positional_encodings import RelativePositionBias2D


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

N_HEADS=8

def masked_cross_entropy(logits, target_ids, mask):
    vocab_size = logits.size(-1)
    logits_flat = logits.reshape(-1, vocab_size)
    targets_flat = target_ids.reshape(-1)
    mask_flat = mask.reshape(-1)

    logits_sel = logits_flat[mask_flat]
    targets_sel = targets_flat[mask_flat]

    return F.cross_entropy(logits_sel, targets_sel)


def accuracy_masked(logits, target_ids, mask):
    preds = logits.argmax(dim=-1)
    correct = (preds == target_ids) & mask
    total = mask.sum().item()
    return correct.sum().item(), total


def build_sudoku_model(n_heads: int = N_HEADS) -> BlackboardTransformer:
    d_model = 128
    num_layers = 4
    dim_feedforward = 512
    H, W = 9, 9
    max_len = H * W
    vocab_size = 10

    pos_enc = RelativePositionBias2D(n_heads, H, W)

    model = BlackboardTransformer(
        vocab_size=vocab_size,
        d_model=d_model,
        nhead=n_heads,
        num_layers=num_layers,
        dim_feedforward=dim_feedforward,
        max_len=max_len,
        dropout=0.1,
        pos_enc=pos_enc,
    ).to(DEVICE)

    return model


def main():
    n_train = 100000 #10_000
    n_val = 1000 #1_000
    batch_size = 64
    num_epochs = 5
    lr = 3e-4



    train_problems = load_hf_sudoku_problems(split="train", n=n_train, seed=0)
    val_problems   = load_hf_sudoku_problems(split="validation", n=n_val, seed=1)


    train_ds = BlackboardSudokuStepDataset(train_problems)
    val_ds = BlackboardSudokuStepDataset(val_problems)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    model = build_sudoku_model(n_heads=N_HEADS)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    for epoch in range(1, num_epochs + 1):
        model.train()
        total_loss = 0.0
        total_correct = 0
        total_tokens = 0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{num_epochs} [train]")
        for batch in pbar:
            input_ids = batch["input_ids"].to(DEVICE)
            target_ids = batch["target_ids"].to(DEVICE)
            mask = batch["mask"].to(DEVICE)

            optimizer.zero_grad()
            logits, _ = model(input_ids)
            loss = masked_cross_entropy(logits, target_ids, mask)
            loss.backward()
            optimizer.step()

            correct, tokens = accuracy_masked(logits, target_ids, mask)
            total_loss += loss.item() * tokens
            total_correct += correct
            total_tokens += tokens

            pbar.set_postfix(
                loss=loss.item(),
                acc=correct / max(tokens, 1)
            )

        train_loss = total_loss / max(total_tokens, 1)
        train_acc = total_correct / max(total_tokens, 1)

        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_tokens = 0
        with torch.no_grad():
            pbar_val = tqdm(val_loader, desc=f"Epoch {epoch}/{num_epochs} [val]")
            for batch in pbar_val:
                input_ids = batch["input_ids"].to(DEVICE)
                target_ids = batch["target_ids"].to(DEVICE)
                mask = batch["mask"].to(DEVICE)

                logits, _ = model(input_ids)
                loss = masked_cross_entropy(logits, target_ids, mask)

                correct, tokens = accuracy_masked(logits, target_ids, mask)
                val_loss += loss.item() * tokens
                val_correct += correct
                val_tokens += tokens

                pbar_val.set_postfix(
                    loss=loss.item(),
                    acc=correct / max(tokens, 1)
                )

        val_loss /= max(val_tokens, 1)
        val_acc = val_correct / max(val_tokens, 1)

        print(
            f"Epoch {epoch}/{num_epochs} "
            f"| train loss: {train_loss:.4f}, train acc: {train_acc:.4f} "
            f"| val loss: {val_loss:.4f}, val acc: {val_acc:.4f}"
        )


        CHECKPOINT_DIR = "src/training/trained_weights"
        os.makedirs(CHECKPOINT_DIR, exist_ok=True)
        ckpt_path = os.path.join(CHECKPOINT_DIR, "sudoku_relative_pe_8heads.pt")
        torch.save(model.state_dict(), ckpt_path)
        print(f"Saved checkpoint to {ckpt_path}")

if __name__ == "__main__":
    main()
