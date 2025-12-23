from typing import List, Tuple
import numpy as np

from src.data.addition_algo import (
    BoardConfig,
    DIGIT_TOKENS,
    PLUS_TOKEN,
    BLANK_TOKEN,
)

MINUS_TOKEN = PLUS_TOKEN


def sample_operands(cfg: BoardConfig, rng: np.random.Generator) -> np.ndarray:
    low = 0
    high = cfg.base ** cfg.n_digits
    xs = rng.integers(low, high, size=2, endpoint=False)
    top = int(xs.max())
    bottom = int(xs.min())
    return np.array([top, bottom], dtype=np.int64)


def number_to_digits(x: int, n_digits: int, base: int = 10) -> List[int]:
    digits = []
    for _ in range(n_digits):
        digits.append(x % base)
        x //= base
    return digits


def build_initial_board(cfg: BoardConfig, xs: np.ndarray) -> np.ndarray:
    board = np.full((cfg.H, cfg.W), BLANK_TOKEN, dtype=np.int64)

    assert cfg.n_addends == 2, "build_initial_board currently assumes 2 operands."
    assert cfg.W >= cfg.n_digits + 1, (
        "Board width W should be at least n_digits + 1 to allow an extra column."
    )

    col_end = cfg.W - 1
    col_start = col_end - (cfg.n_digits - 1)

    top, bottom = xs[0], xs[1]
    top_digits = number_to_digits(top, cfg.n_digits, cfg.base)
    bottom_digits = number_to_digits(bottom, cfg.n_digits, cfg.base)

    for j, d in enumerate(top_digits):
        col = col_end - j
        board[cfg.top_row, col] = d

    for j, d in enumerate(bottom_digits):
        col = col_end - j
        board[cfg.bottom_row, col] = d

    overflow_col = col_start - 1
    minus_col = overflow_col - 1
    if minus_col >= 0:
        board[cfg.bottom_row, minus_col] = MINUS_TOKEN

    return board


def generate_trajectory_variant_A(
    cfg: BoardConfig,
    xs: np.ndarray,
) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    S_seq: List[np.ndarray] = []
    M_seq: List[np.ndarray] = []

    S_t = build_initial_board(cfg, xs)
    S_seq.append(S_t.copy())

    borrow = 0
    col_end = cfg.W - 1

    for step in range(cfg.n_digits):
        col = col_end - step

        col_top = S_t[cfg.top_row, col]
        col_bottom = S_t[cfg.bottom_row, col]

        a = 0 if col_top == BLANK_TOKEN else int(col_top)
        b = 0 if col_bottom == BLANK_TOKEN else int(col_bottom)

        diff = a - b - borrow
        if diff < 0:
            diff += cfg.base
            new_borrow = 1
        else:
            new_borrow = 0

        S_next = S_t.copy()
        M_t = np.zeros_like(S_t, dtype=bool)

        S_next[cfg.result_row, col] = diff
        M_t[cfg.result_row, col] = True

        next_col = col - 1
        if next_col >= 0:
            if new_borrow == 0:
                S_next[cfg.carry_row, next_col] = BLANK_TOKEN
            else:
                S_next[cfg.carry_row, next_col] = new_borrow
            M_t[cfg.carry_row, next_col] = True

        S_seq.append(S_next)
        M_seq.append(M_t)

        S_t = S_next
        borrow = new_borrow

    return S_seq, M_seq
