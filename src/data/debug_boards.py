# src/data/debug_boards.py
import numpy as np
from src.data.addition_algo import BoardConfig, BLANK_TOKEN, PLUS_TOKEN
from src.data.addition_algo import BoardConfig, generate_trajectory
from src.data.problems import generate_problems
from src.data.cot_dataset import CoTAdditionDataset, ID2TOK


from torch.utils.data import DataLoader
from src.data.board_dataset import BlackboardAdditionStepDataset
import torch

def tokens_to_str(token_ids: torch.Tensor) -> str:
    """
    Convert a 1D tensor of token ids into a human-readable CoT string.
    """
    toks = [ID2TOK[int(t)] for t in token_ids]
    return " ".join(toks)


def board_to_str(board: np.ndarray, cfg: BoardConfig) -> str:
    """
    Convert a (H, W) board of token ids into a human-readable string.
    Digits 0-9 are shown as themselves, BLANK as '.', '+' as '+'.
    """
    chars = []
    H, W = board.shape
    for i in range(H):
        row_chars = []
        for j in range(W):
            tok = board[i, j]
            if tok == BLANK_TOKEN:
                row_chars.append(".")
            elif tok == PLUS_TOKEN:
                row_chars.append("+")
            elif 0 <= tok <= 9:
                row_chars.append(str(int(tok)))
            else:
                row_chars.append("?")
        chars.append(" ".join(row_chars))
    return "\n".join(chars)

def print_board(board: np.ndarray, cfg: BoardConfig, title: str = "") -> None:
    if title:
        print(title)
    print(board_to_str(board, cfg))
    print("-" * (2 * cfg.W))

def main_vis():
    # Simple config: 4 rows, 5 columns, 3-digit addition
    cfg = BoardConfig(H=4, W=5, n_digits=3)

    # Generate 2 problems with a fixed seed
    problems = generate_problems(cfg, n=2, seed=123)

    for idx, problem in enumerate(problems):
        xs = problem.operands
        print(f"=== Problem {idx} ===")
        print(f"Operands: {xs[0]} + {xs[1]}")
        S_seq, M_seq = generate_trajectory(cfg, xs)

        # Print all steps S_t
        for t, S_t in enumerate(S_seq):
            print_board(S_t, cfg, title=f"S_{t}")

def main():
    cfg = BoardConfig(H=4, W=5, n_digits=3)

    problems = generate_problems(cfg, n=2, seed=123)
    dataset = BlackboardAdditionStepDataset(problems)
    loader = DataLoader(dataset, batch_size=1, shuffle=False)

    # Look at the first few (S_t, S_{t+1}, M_t)
    for i, batch in enumerate(loader):
        if i >= 6:  # for 2 problems * 3 steps = 6 samples
            break

        inp = batch["input_ids"][0]   # shape (L,)
        tgt = batch["target_ids"][0]  # shape (L,)
        mask = batch["mask"][0]       # shape (L,)

        S_t   = inp.view(cfg.H, cfg.W).numpy()
        S_tp1 = tgt.view(cfg.H, cfg.W).numpy()
        M_t   = mask.view(cfg.H, cfg.W).numpy()

        print(f"=== Dataset sample {i} ===")
        print_board(S_t,   cfg, title="S_t")
        print_board(S_tp1, cfg, title="S_{t+1}")
        print("Mask (True where we train):")
        print(M_t.astype(int))
        print()

def main_cot():
    """
    Debug CoT step-wise dataset: print a few samples with masks highlighting
    which tokens are supervised (result digits, carry digits).
    Mirrors the blackboard step-wise debug.
    """
    cfg = BoardConfig(H=4, W=5, n_digits=3)

    # Use the same underlying problems as for the board
    problems = generate_problems(cfg, n=2, seed=123)
    dataset = CoTAdditionDataset(problems)
    loader = DataLoader(dataset, batch_size=1, shuffle=False)

    for i, batch in enumerate(loader):
        if i >= 6:  # 2 problems * 3 steps = 6 samples
            break

        input_ids = batch["input_ids"][0]    # (L_prefix,)
        target_ids = batch["target_ids"][0]  # (L_step-1,)
        mask = batch["mask"][0]              # (L_step-1,)

        problem_idx = i // cfg.n_digits
        step_idx = i % cfg.n_digits          # 0-based
        xs = problems[problem_idx].operands

        print(f"=== CoT dataset sample {i} ===")
        print(f"Problem {problem_idx}, step {step_idx + 1}")
        print(f"Operands: {xs[0]} + {xs[1]}")

        input_str = tokens_to_str(input_ids)
        target_toks = [ID2TOK[int(t)] for t in target_ids]

        pretty_target = []
        for tok, m in zip(target_toks, mask):
            if m:
                pretty_target.append(f"[{tok}]")
            else:
                pretty_target.append(tok)
        pretty_target_str = " ".join(pretty_target)

        print("Input sequence:")
        print(input_str)
        print("Target sequence (brackets = supervised tokens):")
        print(pretty_target_str)
        print("Mask (1 = supervised, 0 = ignored):")
        print(mask.to(torch.int).tolist())
        print("-" * 80)




if __name__ == "__main__":
    main_cot()