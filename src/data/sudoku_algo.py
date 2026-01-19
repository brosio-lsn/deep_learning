# src/data/sudoku_algo.py
from typing import List, Tuple, Optional
import os
import hashlib

import numpy as np

BLANK_TOKEN = 0
DIGITS = list(range(1, 10))

TRAJECTORY_DIR = "src/data/sudoku_trajectories"
os.makedirs(TRAJECTORY_DIR, exist_ok=True)


def find_empty_cell(board: np.ndarray) -> Optional[Tuple[int, int]]:
    for i in range(9):
        for j in range(9):
            if board[i, j] == BLANK_TOKEN:
                return i, j
    return None


def is_valid(board: np.ndarray, row: int, col: int, val: int) -> bool:
    if val in board[row, :]:
        return False
    if val in board[:, col]:
        return False

    br = (row // 3) * 3
    bc = (col // 3) * 3
    if val in board[br:br+3, bc:bc+3]:
        return False

    return True


def solve_with_trajectory(puzzle: np.ndarray) -> List[np.ndarray]:
    board = puzzle.copy()
    trajectory: List[np.ndarray] = [board.copy()]

    def backtrack() -> bool:
        empty = find_empty_cell(board)
        if empty is None:
            return True

        r, c = empty
        for val in DIGITS:
            if is_valid(board, r, c, val):
                board[r, c] = val
                trajectory.append(board.copy())
                if backtrack():
                    return True
                trajectory.pop()
                board[r, c] = BLANK_TOKEN
        return False

    solved = backtrack()
    if not solved:
        raise ValueError("Backtracking solver could not solve this puzzle.")

    return trajectory


def _puzzle_id(puzzle: np.ndarray) -> str:
    s = "".join(str(int(x)) for x in puzzle.reshape(-1))
    h = hashlib.sha1(s.encode("utf-8")).hexdigest()[:16]
    return h


def _trajectory_path(puzzle: np.ndarray) -> str:
    pid = _puzzle_id(puzzle)
    return os.path.join(TRAJECTORY_DIR, f"traj_{pid}.npz")


def solve_with_trajectory_cached(puzzle: np.ndarray) -> List[np.ndarray]:
    print("begin_traj")
    path = _trajectory_path(puzzle)
    if os.path.isfile(path):
        data = np.load(path)["boards"]
        return [data[i].copy() for i in range(data.shape[0])]

    traj = solve_with_trajectory(puzzle)
    arr = np.stack(traj, axis=0)  # (T, 9, 9)
    np.savez_compressed(path, boards=arr)
    print("traj completed")
    return traj
