# src/data/problems.py
from dataclasses import dataclass
from typing import List
import numpy as np

from src.data.addition_algo import BoardConfig, sample_operands, number_to_digits


@dataclass
class AdditionProblem:
    operands: np.ndarray  # shape (n_addends,)
    cfg: BoardConfig


def generate_problems(cfg: BoardConfig, n: int, seed: int) -> List[AdditionProblem]:
    """
    Simple i.i.d. random problems.
    """
    rng = np.random.default_rng(seed)
    problems: List[AdditionProblem] = []
    for _ in range(n):
        xs = sample_operands(cfg, rng)
        problems.append(AdditionProblem(xs, cfg))
    return problems


def digits_to_number(digits: np.ndarray, base: int = 10) -> int:
    """
    Convert least-significant-first digit array to integer.
    digits[i] is the digit for base^i.
    """
    x = 0
    pow_b = 1
    for d in digits:
        x += int(d) * pow_b
        pow_b *= base
    return x


def generate_diversified_problems(
    cfg: BoardConfig,
    n: int,
    seed: int,
) -> List[AdditionProblem]:
    """
    Generate problems where column-wise digit pairs (top_digit, bottom_digit)
    are as diversified as possible.

    Strategy:
      - Maintain a 10x10 matrix 'counts[d_top, d_bottom]' with how often each
        digit pair has appeared across all columns in the dataset so far.
      - For each new problem:
          * Find the least-seen digit pair (d_top, d_bottom).
          * Choose a random column index.
          * Construct two operands whose digits at that column are exactly
            (d_top, d_bottom), and whose other columns are random.
      - This guarantees that rare digit pairs get sampled more often, giving
        better coverage of the 10x10 space of column patterns.

    Note: This does NOT enforce anything about carries; it just balances
    the visible digit pairs in each column.
    """
    assert cfg.n_addends == 2, "Only implemented for 2-addend addition."

    rng = np.random.default_rng(seed)
    problems: List[AdditionProblem] = []

    # counts[d_top, d_bottom] = number of times this pair appeared in any column
    counts = np.zeros((cfg.base, cfg.base), dtype=np.int64)

    for _ in range(n):
        # choose the least-used digit pair so far
        d_top, d_bottom = np.unravel_index(np.argmin(counts), counts.shape)

        # choose a random column in [0, n_digits-1] that will host this pair
        col_idx = rng.integers(0, cfg.n_digits)

        # initialise random digits for all columns
        top_digits = rng.integers(0, cfg.base, size=cfg.n_digits, endpoint=False)
        bot_digits = rng.integers(0, cfg.base, size=cfg.n_digits, endpoint=False)

        # force the chosen pair at the chosen column
        top_digits[col_idx] = d_top
        bot_digits[col_idx] = d_bottom

        # convert digit arrays (least-significant-first) to integers
        top_val = digits_to_number(top_digits, base=cfg.base)
        bot_val = digits_to_number(bot_digits, base=cfg.base)

        xs = np.array([top_val, bot_val], dtype=np.int64)
        problems.append(AdditionProblem(xs, cfg))

        # update counts for ALL columns in this new problem
        for j in range(cfg.n_digits):
            counts[top_digits[j], bot_digits[j]] += 1

    return problems
