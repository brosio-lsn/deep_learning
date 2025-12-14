from typing import List, Tuple
import numpy as np

from src.data.addition_algo import (
    BoardConfig,
    DIGIT_TOKENS,
    PLUS_TOKEN,
    BLANK_TOKEN,
    number_to_digits,
)


def sample_multi_operands(cfg: BoardConfig, rng: np.random.Generator) -> np.ndarray:
    low = 0
    high = cfg.base ** cfg.n_digits
    xs = rng.integers(low, high, size=cfg.n_addends, endpoint=False)
    return xs.astype(np.int64)


def build_initial_board_multi(cfg: BoardConfig, xs: np.ndarray) -> np.ndarray:
    board = np.full((cfg.H, cfg.W), BLANK_TOKEN, dtype=np.int64)

    assert cfg.W >= cfg.n_digits + 1

    col_end = cfg.W - 1
    col_start = col_end - (cfg.n_digits - 1)

    for k in range(cfg.n_addends):
        row = cfg.top_row + k
        x = int(xs[k])
        digits = number_to_digits(x, cfg.n_digits, cfg.base)
        for j, d in enumerate(digits):
            col = col_end - j
            board[row, col] = d

    overflow_col = col_start - 1
    plus_col = overflow_col - 1
    if plus_col >= 0:
        last_operand_row = cfg.top_row + cfg.n_addends - 1
        board[last_operand_row, plus_col] = PLUS_TOKEN

    return board


def generate_multi_trajectory(
    cfg: BoardConfig,
    xs: np.ndarray,
) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    S_seq: List[np.ndarray] = []
    M_seq: List[np.ndarray] = []

    S_t = build_initial_board_multi(cfg, xs)
    S_seq.append(S_t.copy())

    carry = 0
    col_end = cfg.W - 1

    for step in range(cfg.n_digits):
        col = col_end - step

        column_sum = carry
        for row in range(cfg.top_row, cfg.top_row + cfg.n_addends):
            d = S_t[row, col]
            if d != BLANK_TOKEN:
                column_sum += int(d)

        new_digit = column_sum % cfg.base
        carry = column_sum // cfg.base

        S_next = S_t.copy()
        M_t = np.zeros_like(S_t, dtype=bool)

        S_next[cfg.result_row, col] = new_digit
        M_t[cfg.result_row, col] = True

        next_col = col - 1
        if next_col >= 0:
            if carry == 0:
                S_next[cfg.carry_row, next_col] = BLANK_TOKEN
            else:
                S_next[cfg.carry_row, next_col] = carry
            M_t[cfg.carry_row, next_col] = True

        S_seq.append(S_next)
        M_seq.append(M_t)

        S_t = S_next

    return S_seq, M_seq
