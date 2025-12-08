# src/training/sample_efficiency_trainer.py

import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from typing import List, Tuple, Dict

from src.data.addition_algo import BoardConfig
from src.data.board_dataset import BlackboardAdditionStepDataset
from src.models.blackboard_transformer import BlackboardTransformer
from src.models.positional_encodings import (
    SinusoidalPositionalEncoding,
    AbsolutePositionalEncoding2D,
    RelativePositionBias2D,
)

from src.data.sample_efficiency import (
    Triplet,
    generate_setting1_random_fraction,
    generate_setting2_position_split,
    generate_setting3_order_constraint,
    generate_setting4_triplet_holdout,
    extract_triplets,
)


# ---------------------------------------------------------------------------
# Loss + accuracy utilities (same spirit as training_board_phase_0)
# ---------------------------------------------------------------------------

def masked_cross_entropy(logits, target_ids, mask):
    """
    logits: (B, L, V)
    target_ids: (B, L)
    mask: (B, L) bool

    Cross-entropy averaged over masked positions.
    """
    vocab_size = logits.size(-1)

    logits_flat = logits.reshape(-1, vocab_size)   # (B*L, V)
    targets_flat = target_ids.reshape(-1)          # (B*L,)
    mask_flat = mask.reshape(-1)                   # (B*L,)

    logits_sel = logits_flat[mask_flat]
    targets_sel = targets_flat[mask_flat]

    return F.cross_entropy(logits_sel, targets_sel)


def compute_accuracy(logits, target_ids, mask):
    """
    Overall masked accuracy.

    logits: (B, L, V)
    target_ids: (B, L)
    mask: (B, L) bool
    """
    preds = logits.argmax(dim=-1)          # (B, L)
    correct = (preds == target_ids) & mask  # (B, L)
    total = mask.sum().item()
    if total == 0:
        return 0.0
    return correct.sum().item() / total


def evaluate_model(model, data_loader, device):
    model.eval()
    total_loss = 0.0
    total_tokens = 0
    total_correct = 0

    with torch.no_grad():
        for batch in data_loader:
            input_ids  = batch["input_ids"].to(device)
            target_ids = batch["target_ids"].to(device)
            mask       = batch["mask"].to(device)

            logits, _ = model(input_ids)

            loss = masked_cross_entropy(logits, target_ids, mask)
            batch_tokens = mask.sum().item()
            total_loss += loss.item() * batch_tokens
            total_tokens += batch_tokens

            preds = logits.argmax(dim=-1)
            correct = ((preds == target_ids) & mask).sum().item()
            total_correct += correct

    avg_loss = total_loss / max(total_tokens, 1)
    avg_acc  = total_correct / max(total_tokens, 1)
    return avg_loss, avg_acc


def train_one_model(
    cfg: BoardConfig,
    train_problems,
    test_problems,
    pos_enc_name: str,
    pos_enc_module,
    device,
    d_model: int = 128,
    n_heads: int = 4,
    num_layers: int = 3,
    dim_feedforward: int = 512,
    dropout: float = 0.1,
    batch_size: int = 128,
    num_epochs: int = 3,
    lr: float = 3e-4,
):
    """
    Train a BlackboardTransformer with the given positional encoding on
    train_problems, evaluate on test_problems, and return test accuracy.
    """
    # Datasets / loaders
    train_ds = BlackboardAdditionStepDataset(train_problems)
    test_ds  = BlackboardAdditionStepDataset(test_problems)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    test_loader  = DataLoader(test_ds,  batch_size=batch_size, shuffle=False)

    vocab_size = 12
    max_len = cfg.H * cfg.W

    # Some PEs are absolute (no args), some depend on H,W. We assume the
    # module passed is already instantiated with correct shape if needed.
    # BlackboardTransformer takes a pos_enc module as argument.
    model = BlackboardTransformer(
        vocab_size=vocab_size,
        d_model=d_model,
        nhead=n_heads,
        num_layers=num_layers,
        dim_feedforward=dim_feedforward,
        max_len=max_len,
        dropout=dropout,
        pos_enc=pos_enc_module,
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    for epoch in range(1, num_epochs + 1):
        model.train()
        total_loss = 0.0
        total_tokens = 0
        total_correct = 0

        pbar = tqdm(train_loader, desc=f"{pos_enc_name} | epoch {epoch}/{num_epochs}")
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
            total_loss += loss.item() * batch_tokens
            total_tokens += batch_tokens

            preds = logits.argmax(dim=-1)
            correct = ((preds == target_ids) & mask).sum().item()
            total_correct += correct

            batch_acc = correct / max(batch_tokens, 1)
            pbar.set_postfix(loss=loss.item(), acc=batch_acc)

        avg_train_loss = total_loss / max(total_tokens, 1)
        avg_train_acc  = total_correct / max(total_tokens, 1)
        print(
            f"[{pos_enc_name}] Epoch {epoch}/{num_epochs} "
            f"| train loss/token: {avg_train_loss:.4f} "
            f"| train acc(masked): {avg_train_acc:.4f}"
        )

    test_loss, test_acc = evaluate_model(model, test_loader, device)
    print(
        f"[{pos_enc_name}] Final test loss/token: {test_loss:.4f} "
        f"| test acc(masked): {test_acc:.4f}"
    )
    return test_acc


# ---------------------------------------------------------------------------
# Main experiment: 4 settings
# ---------------------------------------------------------------------------

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    # -------------------------------
    # Shared model hyperparameters
    # -------------------------------
    d_model = 128
    n_heads = 4
    num_layers = 3
    dim_feedforward = 512
    dropout = 0.1
    batch_size = 128
    num_epochs = 10
    lr = 3e-4

    # -------------------------------
    # Base config: 3-digit addition
    # used for settings 1, 3, 4
    # -------------------------------
    cfg_3 = BoardConfig(H=4, W=3 + 2, n_digits=3)
    max_len_3 = cfg_3.H * cfg_3.W

    # Positional encodings for 3-digit experiments
    pes_3 = [
        ("Relative PE",   RelativePositionBias2D(n_heads, cfg_3.H, cfg_3.W)),
        ("Sinusoidal PE", SinusoidalPositionalEncoding(d_model, max_len=max_len_3)),
        ("Absolute PE",   AbsolutePositionalEncoding2D(d_model, cfg_3.H, cfg_3.W)),
    ]

    # ----------------------------------------------------------------------
    # Setting 1: random subset size sweep (fraction of a fixed max_train)
    # ----------------------------------------------------------------------
    print("\n==================== SETTING 1: Random fraction sweep ====================")

    max_train_setting1 = 500000
    n_test_setting1 = 100000
    seed_base = 0

    frac_values = [i / 10.0 for i in range(1, 10)]  # 0.1, 0.2, ..., 0.9

    setting1_results: Dict[str, List[float]] = {name: [] for name, _ in pes_3}

    for frac in frac_values:
        n_train = max(1, int(frac * max_train_setting1))
        print(f"\n[Setting 1] frac={frac:.1f}, n_train={n_train}, n_test={n_test_setting1}")

        train_problems, test_problems = generate_setting1_random_fraction(
            cfg_3,
            n_train=n_train,
            n_test=n_test_setting1,
            seed=seed_base + int(frac * 100),
        )

        for name, pe_module in pes_3:
            print(f"\n--- Setting 1 | frac={frac:.1f} | PE={name} ---")
            acc = train_one_model(
                cfg_3,
                train_problems,
                test_problems,
                pos_enc_name=name,
                pos_enc_module=pe_module,
                device=device,
                d_model=d_model,
                n_heads=n_heads,
                num_layers=num_layers,
                dim_feedforward=dim_feedforward,
                dropout=dropout,
                batch_size=batch_size,
                num_epochs=num_epochs,
                lr=lr,
            )
            setting1_results[name].append(acc)

    # Plot Setting 1: accuracy vs fraction
    plt.figure()
    for name in setting1_results:
        plt.plot(frac_values, setting1_results[name], marker="o", label=name)
    plt.xlabel("Training fraction (of max_train_setting1)")
    plt.ylabel("Test masked accuracy")
    plt.ylim(0.0, 1.05)
    plt.title("Setting 1: sample size vs accuracy (3-digit)")
    plt.grid(True)
    plt.legend()
    plt.savefig("/cluster/project/infk/krause/wnanadavies/deep_learning/plots/setting1_sample_efficiency.png")
    plt.close()

    # ----------------------------------------------------------------------
    # Shared pool of 10 triplets (used in settings 2, 3, 4)
    # Triplets: (carry_in, digit_top, digit_bottom), with top != bottom
    # ----------------------------------------------------------------------
    TRIPLETS_POOL: List[Triplet] = [
        (0, 0, 1),
        (0, 0, 2),
        (0, 1, 3),
        (0, 2, 5),
        (0, 3, 4),
        (0, 3, 7),
        (0, 4, 6),
        (0, 5, 8),
        (0, 6, 9),
        (0, 7, 9),
    ]

    # ----------------------------------------------------------------------
    # Setting 2: position split with triplets_of_interest; 10-digit addition
    # ----------------------------------------------------------------------
    print("\n==================== SETTING 2: Position split (10 digits) ====================")

    # 10-digit board config
    cfg_10 = BoardConfig(H=4, W=10 + 2, n_digits=10)
    max_len_10 = cfg_10.H * cfg_10.W

    # PEs instantiated for the 10-digit board
    pes_10 = [
        ("Relative PE",   RelativePositionBias2D(n_heads, cfg_10.H, cfg_10.W)),
        ("Sinusoidal PE", SinusoidalPositionalEncoding(d_model, max_len=max_len_10)),
        ("Absolute PE",   AbsolutePositionalEncoding2D(d_model, cfg_10.H, cfg_10.W)),
    ]

    n_train_setting2 = 500000
    n_test_setting2 = 100000
    seed_setting2 = 42

    frac_pos_values = [i / 10.0 for i in range(1, 10)]  # 0.1..0.9

    setting2_results: Dict[str, List[float]] = {name: [] for name, _ in pes_10}

    for frac_pos in frac_pos_values:
        print(f"\n[Setting 2] frac_positions={frac_pos:.1f}, n_train={n_train_setting2}, n_test={n_test_setting2}")

        train_problems, test_problems, allowed_cols = generate_setting2_position_split(
            cfg_10,
            n_train=n_train_setting2,
            n_test=n_test_setting2,
            seed=seed_setting2 + int(frac_pos * 100),
            triplets_of_interest=TRIPLETS_POOL,
            frac_positions=frac_pos,
        )
        print(f"Allowed training columns per triplet: {allowed_cols}")

        for name, pe_module in pes_10:
            print(f"\n--- Setting 2 | frac_positions={frac_pos:.1f} | PE={name} ---")
            acc = train_one_model(
                cfg_10,
                train_problems,
                test_problems,
                pos_enc_name=name,
                pos_enc_module=pe_module,
                device=device,
                d_model=d_model,
                n_heads=n_heads,
                num_layers=num_layers,
                dim_feedforward=dim_feedforward,
                dropout=dropout,
                batch_size=batch_size,
                num_epochs=num_epochs,
                lr=lr,
            )
            setting2_results[name].append(acc)

    # Plot Setting 2: accuracy vs frac_positions (10-digit)
    plt.figure()
    for name in setting2_results:
        plt.plot(frac_pos_values, setting2_results[name], marker="o", label=name)
    plt.xlabel("frac_positions (train columns allowed for each triplet)")
    plt.ylabel("Test masked accuracy")
    plt.ylim(0.0, 1.05)
    plt.title("Setting 2: position split vs accuracy (10-digit)")
    plt.grid(True)
    plt.legend()
    plt.savefig("/cluster/project/infk/krause/wnanadavies/deep_learning/plots/setting2_position_split_10digit.png")
    plt.close()

        # ----------------------------------------------------------------------
    # Setting 3: order constraint (pattern_train vs pattern_test, 3 digits)
    # ----------------------------------------------------------------------
    print("\n==================== SETTING 3: Order constraint (3 digits) ====================")

    pattern_train: List[Triplet] = TRIPLETS_POOL[:3]
    pattern_test: List[Triplet] = [(cin, b, a) for (cin, a, b) in pattern_train]

    n_train_setting3 = 500000
    n_test_setting3 = 100000
    seed_setting3 = 123

    train3, test3 = generate_setting3_order_constraint(
        cfg_3,
        n_train=n_train_setting3,
        n_test=n_test_setting3,
        seed=seed_setting3,
        pattern_train=pattern_train,
        pattern_test=pattern_test,
    )

    print("Setting 3 patterns:")
    print("  pattern_train:", pattern_train)
    print("  pattern_test :", pattern_test)

    # collect results to save
    setting3_results: Dict[str, float] = {}

    for name, pe_module in pes_3:
        print(f"\n--- Setting 3 | PE={name} ---")
        acc = train_one_model(
            cfg_3,
            train3,
            test3,
            pos_enc_name=name,
            pos_enc_module=pe_module,
            device=device,
            d_model=d_model,
            n_heads=n_heads,
            num_layers=num_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_size=batch_size,
            num_epochs=num_epochs,
            lr=lr,
        )
        print(f"[Setting 3] {name} test accuracy: {acc:.4f}")
        setting3_results[name] = acc

    # save Setting 3 results to a text file
    with open("setting3_results.txt", "w") as f:
        f.write("=== Setting 3: Order constraint (3 digits) ===\n")
        f.write(f"pattern_train: {pattern_train}\n")
        f.write(f"pattern_test : {pattern_test}\n\n")
        for name, acc in setting3_results.items():
            f.write(f"{name}: test_accuracy = {acc:.6f}\n")


    # ----------------------------------------------------------------------
    # Setting 4: triplet hold-out (10 forbidden triplets, 3 digits)
    # ----------------------------------------------------------------------
    print("\n==================== SETTING 4: Triplet hold-out (3 digits) ====================")

    forbidden_triplets: List[Triplet] = TRIPLETS_POOL  # all 10 are forbidden

    n_train_setting4 = 500000
    n_test_setting4 = 100000
    seed_setting4 = 999

    train4, test4 = generate_setting4_triplet_holdout(
        cfg_3,
        n_train=n_train_setting4,
        n_test=n_test_setting4,
        seed=seed_setting4,
        forbidden_triplets=forbidden_triplets,
    )

    print("Setting 4 forbidden triplets:", forbidden_triplets)

    # collect results to save
    setting4_results: Dict[str, float] = {}

    for name, pe_module in pes_3:
        print(f"\n--- Setting 4 | PE={name} ---")
        acc = train_one_model(
            cfg_3,
            train4,
            test4,
            pos_enc_name=name,
            pos_enc_module=pe_module,
            device=device,
            d_model=d_model,
            n_heads=n_heads,
            num_layers=num_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_size=batch_size,
            num_epochs=num_epochs,
            lr=lr,
        )
        print(f"[Setting 4] {name} test accuracy: {acc:.4f}")
        setting4_results[name] = acc

    # save Setting 4 results to a text file
    with open("setting4_results.txt", "w") as f:
        f.write("=== Setting 4: Triplet hold-out (3 digits) ===\n")
        f.write(f"forbidden_triplets: {forbidden_triplets}\n\n")
        for name, acc in setting4_results.items():
            f.write(f"{name}: test_accuracy = {acc:.6f}\n")



if __name__ == "__main__":
    main()


#TODO change nb of epochs and nb train/test