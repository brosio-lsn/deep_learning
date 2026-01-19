from datasets import load_dataset
import numpy as np
from dataclasses import dataclass
from typing import List
import numpy as np


@dataclass
class SudokuProblem:
    puzzle: np.ndarray
    solution: np.ndarray


def string_to_grid(s: str) -> np.ndarray:
    arr = np.fromiter((int(c) for c in s), dtype=np.int64, count=81)
    return arr.reshape(9, 9)


def load_hf_sudoku_problems(
    split: str = "train",
    n: int | None = None,
    seed: int = 0,
) -> List[SudokuProblem]:
    subset_split = f"{split}[:10%]"
    ds = load_dataset("Ritvik19/Sudoku-Dataset", split=subset_split)
    print("begin")

    if n is not None:
        rng = np.random.default_rng(seed)
        n_effective = min(n, len(ds))
        indices = rng.choice(len(ds), size=n_effective, replace=False)
        ds = ds.select(indices)
    else:
        ds = ds.shuffle(seed=seed)

    problems: List[SudokuProblem] = []
    for ex in ds:
        puzzle_grid = string_to_grid(ex["puzzle"])
        solution_grid = string_to_grid(ex["solution"])
        problems.append(SudokuProblem(puzzle=puzzle_grid, solution=solution_grid))
    print("end")
    return problems