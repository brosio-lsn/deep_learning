# src/data/board_dataset.py
from typing import Dict, List
import torch
from torch.utils.data import Dataset

from src.data.addition_algo import BoardConfig, generate_trajectory_variant_A as generate_addition_trajectory
from src.data.subtraction_algo import generate_trajectory_variant_A as generate_subtraction_trajectory
from src.data.multi_addition_algo import generate_multi_trajectory
from src.data.problems import AdditionProblem, SubtractionProblem, MultiAdditionProblem


class BlackboardAdditionStepDataset(Dataset):
    """
    Dataset for masked-write training (variant A) on 2D blackboard addition.

    Each item corresponds to one step t in one trajectory:
      - input_ids:  S_t  flattened to shape (L,)
      - target_ids: S_{t+1} flattened to shape (L,)
      - mask:       M_t flattened to shape (L,), boolean

    Where L = H * W.
    """

    def __init__(self, problems: List[AdditionProblem]) -> None:
        super().__init__()
        assert len(problems) > 0, "Problems list must not be empty."
        self.problems = problems
        self.cfg: BoardConfig = problems[0].cfg
        self.n_steps = self.cfg.n_digits  # one step per digit

    def __len__(self) -> int:
        # one sample per (example, step)
        return len(self.problems) * self.n_steps

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        traj_idx = idx // self.n_steps
        step_idx = idx % self.n_steps

        problem = self.problems[traj_idx]
        xs = problem.operands

        S_seq, M_seq = generate_addition_trajectory(self.cfg, xs)

        S_t   = torch.from_numpy(S_seq[step_idx]).view(-1).long()
        S_tp1 = torch.from_numpy(S_seq[step_idx + 1]).view(-1).long()
        M_t   = torch.from_numpy(M_seq[step_idx]).view(-1)

        return {
            "input_ids": S_t,      # (L,)
            "target_ids": S_tp1,   # (L,)
            "mask": M_t,           # (L,) bool
        }

class BlackboardSubtractionStepDataset(Dataset):
    def __init__(self, problems: List[SubtractionProblem]) -> None:
        super().__init__()
        assert len(problems) > 0
        self.problems = problems
        self.cfg: BoardConfig = problems[0].cfg
        self.n_steps = self.cfg.n_digits

    def __len__(self) -> int:
        return len(self.problems) * self.n_steps

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        traj_idx = idx // self.n_steps
        step_idx = idx % self.n_steps

        problem = self.problems[traj_idx]
        xs = problem.operands

        S_seq, M_seq = generate_subtraction_trajectory(self.cfg, xs)

        S_t = torch.from_numpy(S_seq[step_idx]).view(-1).long()
        S_tp1 = torch.from_numpy(S_seq[step_idx + 1]).view(-1).long()
        M_t = torch.from_numpy(M_seq[step_idx]).view(-1)

        return {
            "input_ids": S_t,
            "target_ids": S_tp1,
            "mask": M_t,
        }
        

class BlackboardMultiAdditionStepDataset(Dataset):
    def __init__(self, problems: List[MultiAdditionProblem]) -> None:
        super().__init__()
        assert len(problems) > 0
        self.problems = problems
        self.cfg: BoardConfig = problems[0].cfg
        self.n_steps = self.cfg.n_digits

    def __len__(self) -> int:
        return len(self.problems) * self.n_steps

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        traj_idx = idx // self.n_steps
        step_idx = idx % self.n_steps

        problem = self.problems[traj_idx]
        xs = problem.operands

        S_seq, M_seq = generate_multi_trajectory(self.cfg, xs)

        S_t = torch.from_numpy(S_seq[step_idx]).view(-1).long()
        S_tp1 = torch.from_numpy(S_seq[step_idx + 1]).view(-1).long()
        M_t = torch.from_numpy(M_seq[step_idx]).view(-1)

        return {
            "input_ids": S_t,
            "target_ids": S_tp1,
            "mask": M_t,
        }