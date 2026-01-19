#test_specific_sudoku.py

import os
from typing import List

import numpy as np
import torch

from src.data.sudoku_algo import BLANK_TOKEN
from src.models.transformers import BlackboardTransformer
from src.models.positional_encodings import RelativePositionBias2D
from src.visualization.sudoku_video import save_sudoku_video

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

CHECKPOINT_DIR = "src/training/trained_weights"
CKPT_NAME = "sudoku_relative_pe_4heads.pt"


def build_sudoku_model(n_heads: int = 4) -> BlackboardTransformer:
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


def board_to_tensor(board: np.ndarray) -> torch.Tensor:
    x = torch.from_numpy(board.astype(np.int64).reshape(-1))
    return x.unsqueeze(0).to(DEVICE)


def rollout_sudoku_with_model(
    model: BlackboardTransformer,
    puzzle: np.ndarray,
    solution: np.ndarray | None = None,
    max_steps: int = 81,
) -> List[np.ndarray]:
    board = puzzle.copy()
    trajectory: List[np.ndarray] = [board.copy()]

    for _ in range(max_steps):
        if np.all(board != BLANK_TOKEN):
            break

        input_ids = board_to_tensor(board)
        with torch.no_grad():
            logits, _ = model(input_ids)

        logits = logits[0]
        probs = torch.softmax(logits, dim=-1)

        flat = board.reshape(-1)
        blank_idx = np.where(flat == BLANK_TOKEN)[0]
        if blank_idx.size == 0:
            break

        blank_probs = probs[blank_idx]
        blank_probs[:, BLANK_TOKEN] = -1e9

        per_blank_best_prob, per_blank_best_digit = blank_probs.max(dim=1)
        best_blank_idx = per_blank_best_prob.argmax()

        chosen_flat = blank_idx[best_blank_idx.item()]
        chosen_row = chosen_flat // 9
        chosen_col = chosen_flat % 9
        chosen_digit = per_blank_best_digit[best_blank_idx].item()

        board[chosen_row, chosen_col] = chosen_digit
        trajectory.append(board.copy())

    if solution is not None:
        correct_cells = (board == solution).sum()
        total_cells = board.size
        print(f"Cell-wise agreement with ground truth: {correct_cells}/{total_cells}")

    return trajectory


def main():
    os.makedirs("outputs", exist_ok=True)

    puzzle = np.array([
        [5, 3, 0, 0, 7, 8, 0, 1, 2],
        [6, 0, 2, 1, 9, 0, 0, 4, 8],
        [1, 9, 8, 3, 0, 2, 5, 6, 0],
        [8, 0, 9, 7, 6, 0, 4, 0, 3],
        [4, 2, 0, 8, 5, 3, 7, 0, 0],
        [7, 1, 3, 0, 2, 4, 8, 0, 0],
        [9, 0, 1, 0, 3, 7, 0, 8, 4],
        [2, 0, 7, 4, 1, 0, 0, 3, 5],
        [3, 4, 0, 2, 8, 0, 1, 7, 9],
    ], dtype=np.int64)

    solution = np.array([
        [5, 3, 4, 6, 7, 8, 9, 1, 2],
        [6, 7, 2, 1, 9, 5, 3, 4, 8],
        [1, 9, 8, 3, 4, 2, 5, 6, 7],
        [8, 5, 9, 7, 6, 1, 4, 2, 3],
        [4, 2, 6, 8, 5, 3, 7, 9, 1],
        [7, 1, 3, 9, 2, 4, 8, 5, 6],
        [9, 6, 1, 5, 3, 7, 2, 8, 4],
        [2, 8, 7, 4, 1, 9, 6, 3, 5],
        [3, 4, 5, 2, 8, 6, 1, 7, 9],
    ], dtype=np.int64)

    print("Initial puzzle:")
    print(puzzle)
    print("\nGround-truth solution:")
    print(solution)
    print()

    model = build_sudoku_model(n_heads=4)
    ckpt_path = os.path.join(CHECKPOINT_DIR, CKPT_NAME)
    state = torch.load(ckpt_path, map_location=DEVICE)
    model.load_state_dict(state)
    model.eval()

    trajectory = rollout_sudoku_with_model(model, puzzle, solution=solution, max_steps=81)

    out_path = "outputs/sudoku_model_rollout_custom.gif"
    save_sudoku_video(trajectory, out_path=out_path, fps=2)
    print(f"Saved rollout video to {out_path}")


if __name__ == "__main__":
    main()
