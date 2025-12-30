# src/data/board_dataset.py
from typing import Dict, List
import torch
from torch.utils.data import Dataset

from src.data.addition_algo import BoardConfig, generate_trajectory_variant_A as generate_addition_trajectory, BLANK_TOKEN, VOID_TOKEN
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
def _sample_bert_style_mask(eligible: torch.Tensor, rate: float, rng: torch.Generator) -> torch.Tensor:
    """Sample a boolean mask (same shape as `eligible`) from eligible positions
    eligible: bool tensor, True where masking is allowed
    rate: probability of selecting each eligible position
    Returns a bool tensor M with M[p]=True meaning "predict/denoise this position"
    """
    if rate <= 0.0:
        return torch.zeros_like(eligible, dtype=torch.bool)
    u = torch.rand(eligible.shape, generator=rng)
    return eligible & (u < rate)


def _apply_bert_corruption(input_ids: torch.Tensor, denoise_mask: torch.Tensor, rng: torch.Generator, random_token_ids: torch.Tensor, void_token_id: int = VOID_TOKEN, p_void: float = 0.8, p_random: float = 0.1) -> torch.Tensor:
    """Apply BERT-style corruption in-place on a copy of input_ids
    80% -> replace with VOID
    10% -> replace with random valid token
    10% -> keep unchanged
    """
    if denoise_mask.sum().item() == 0:
        return input_ids

    x = input_ids.clone()
    idx = denoise_mask.nonzero(as_tuple=False).view(-1)
    # sample decision per masked position
    r = torch.rand(idx.shape[0], generator=rng)

    # 80% VOID
    void_sel = idx[r < p_void]
    if void_sel.numel() > 0:
        x[void_sel] = int(void_token_id)

    # 10% random token, choose random tokens from provided set
    rand_sel = idx[(r >= p_void) & (r < (p_void + p_random))]
    if rand_sel.numel() > 0:
        choice = torch.randint(
            low=0,
            high=random_token_ids.numel(),
            size=(rand_sel.numel(),),
            generator=rng,
        )
        x[rand_sel] = random_token_ids[choice]

    # remaining 10%: keep unchanged
    return x

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
    
class BlackboardAdditionDenoisingStepDataset(Dataset):
    """Masked-write + BERT-style denoising dataset (Direction 2).

    Compared to BlackboardAdditionStepDataset, this dataset:
      - corrupts a subset of *eligible* cells in S_t using BERT-style corruption
        (80% VOID / 10% random / 10% unchanged, modulable)
      - expands the training mask to include denoising positions:
            M_loss = M_stepwrite ∪ M_denoise

    Two eligibility regimes:
      - setting="local"  (Setting 2): eligible = stepwrite cells + last-step cells
      - setting="global" (Setting 3): eligible = all editable cells (carry/result rows)
    """

    def __init__(self, problems: List[AdditionProblem], setting: str = "local", denoise_rate: float = 0.15) -> None:
        super().__init__()
        assert len(problems) > 0
        assert setting in {"local", "global"}
        self.problems = problems
        self.cfg: BoardConfig = problems[0].cfg
        self.n_steps = self.cfg.n_digits
        self.setting = setting
        self.denoise_rate = float(denoise_rate)

        # For random replacements we restrict to non-operator tokens.
        # (digits + BLANK + VOID)
        self._random_token_ids = torch.tensor(
            list(range(10)) + [BLANK_TOKEN, VOID_TOKEN], dtype=torch.long
        )

    def __len__(self) -> int:
        return len(self.problems) * self.n_steps

    def _eligible_positions(self, *, step_idx: int) -> torch.Tensor:
        """Return a bool mask of eligible denoising positions (flattened)."""
        H, W = self.cfg.H, self.cfg.W
        L = H * W

        eligible = torch.zeros(L, dtype=torch.bool)

        if self.setting == "global":
            # Setting 3: allow edits anywhere on editable rows (carry + result)
            for r in [self.cfg.carry_row, self.cfg.result_row]:
                start = r * W
                eligible[start : start + W] = True
            return eligible

        # Setting 2: local
        # Eligible = last-step written cells (result+carry of t-1) plus current step write mask
        col_end = W - 1
        col_cur = col_end - step_idx
        # current step cells, result at col_cur and carry at col_cur-1
        if 0 <= col_cur < W:
            eligible[self.cfg.result_row * W + col_cur] = True
        if 0 <= (col_cur - 1) < W:
            eligible[self.cfg.carry_row * W + (col_cur - 1)] = True

        # previous step cells
        if step_idx > 0:
            col_prev = col_end - (step_idx - 1)  # one column to the right
            if 0 <= col_prev < W:
                eligible[self.cfg.result_row * W + col_prev] = True
            if 0 <= (col_prev - 1) < W:
                eligible[self.cfg.carry_row * W + (col_prev - 1)] = True

        return eligible

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        traj_idx = idx // self.n_steps
        step_idx = idx % self.n_steps

        problem = self.problems[traj_idx]
        xs = problem.operands

        S_seq, M_seq = generate_addition_trajectory(self.cfg, xs)

        S_t = torch.from_numpy(S_seq[step_idx]).view(-1).long()
        S_tp1 = torch.from_numpy(S_seq[step_idx + 1]).view(-1).long()
        M_stepwrite = torch.from_numpy(M_seq[step_idx]).view(-1).bool()

        # Per-sample RNG (deterministic given seed + idx)
        rng = torch.Generator()
        rng.manual_seed(self.seed + idx)

        eligible = self._eligible_positions(step_idx=step_idx)
        M_denoise = _sample_bert_style_mask(eligible=eligible, rate=self.denoise_rate, rng=rng)

        input_noisy = _apply_bert_corruption(input_ids=S_t, denoise_mask=M_denoise, rng=rng, random_token_ids=self._random_token_ids, void_token_id=VOID_TOKEN,
        )

        M_loss = (M_stepwrite | M_denoise).bool()

        return {
            "input_ids": input_noisy,   # (L,)
            "target_ids": S_tp1,        # (L,)
            "mask": M_loss,             # (L,) bool
            "denoise_mask": M_denoise,  # (L,) bool
        }