# src/data/board_dataset_sudoku.py
from typing import Dict, List
import numpy as np
import torch
from torch.utils.data import Dataset

from src.data.sudoku_hf import SudokuProblem
from src.data.sudoku_algo import solve_with_trajectory_cached, BLANK_TOKEN


class BlackboardSudokuStepDataset(Dataset):
    def __init__(self, problems: List[SudokuProblem]) -> None:
        super().__init__()
        assert len(problems) > 0
        self.problems = problems

        self.trajectories: List[List[np.ndarray]] = []
        self.step_index: List[tuple[int, int]] = []

        for p_idx, prob in enumerate(self.problems):
            traj = solve_with_trajectory_cached(prob.puzzle)
            self.trajectories.append(traj)
            for t in range(len(traj) - 1):
                self.step_index.append((p_idx, t))

    def __len__(self) -> int:
        return len(self.step_index)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        p_idx, t = self.step_index[idx]
        traj = self.trajectories[p_idx]
        S_t = traj[t]
        S_tp1 = traj[t + 1]

        S_t_flat = torch.from_numpy(S_t.astype(np.int64).reshape(-1))
        S_tp1_flat = torch.from_numpy(S_tp1.astype(np.int64).reshape(-1))

        diff = (S_tp1_flat != S_t_flat)
        mask = diff & (S_tp1_flat != BLANK_TOKEN)

        return {
            "input_ids": S_t_flat,
            "target_ids": S_tp1_flat,
            "mask": mask,
        }
