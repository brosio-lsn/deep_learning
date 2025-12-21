import os
from typing import Tuple

import numpy as np
import torch

from src.data.addition_algo import BoardConfig
from src.data.multi_addition_algo import generate_multi_trajectory
from src.data.problems import digits_to_number
from src.models.blackboard_transformer import BlackboardTransformer
from src.models.positional_encodings import RelativePositionBias2D

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

BASE_DIR = os.path.dirname(__file__)
CHECKPOINT_DIR = os.path.join(BASE_DIR, "trained_weights")


def build_blackboard_relative_model_multi(cfg: BoardConfig) -> BlackboardTransformer:
    d_model = 128
    n_heads = 1
    num_layers = 4
    dim_feedforward = 512
    max_len = cfg.H * cfg.W
    vocab_size = 12

    pos_enc = RelativePositionBias2D(n_heads, cfg.H, cfg.W)

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


def load_multi_add_model(cfg: BoardConfig) -> BlackboardTransformer:
    model = build_blackboard_relative_model_multi(cfg)
    ckpt_path = os.path.join(
        CHECKPOINT_DIR,
        "blackboard_relative_pe_multiadd_lastlayer.pt",
    )
    state = torch.load(ckpt_path, map_location=DEVICE)
    model.load_state_dict(state)
    model.eval()
    return model


def run_model_multi_addition(
    model: BlackboardTransformer,
    cfg: BoardConfig,
    xs: np.ndarray,
) -> Tuple[int, np.ndarray]:
    S_seq, M_seq = generate_multi_trajectory(cfg, xs)

    S_t = S_seq[0].copy()

    for step in range(cfg.n_digits):
        board_flat = torch.from_numpy(S_t.astype(np.int64)).view(1, -1).to(DEVICE)
        with torch.no_grad():
            logits, _ = model(board_flat)

        logits_step = logits[0]
        preds = logits_step.argmax(dim=-1).cpu().numpy()

        M_t = M_seq[step].reshape(-1)
        S_t_flat = S_t.reshape(-1)
        S_t_flat[M_t] = preds[M_t]
        S_t = S_t_flat.reshape(S_t.shape)

    result_row = cfg.result_row
    col_end = cfg.W - 1

    digits = []
    for j in range(cfg.n_digits):
        col = col_end - j
        d = int(S_t[result_row, col])
        digits.append(d)

    result = digits_to_number(np.array(digits, dtype=np.int64), base=cfg.base)
    return result, S_t


def print_board(board: np.ndarray) -> None:
    for r in range(board.shape[0]):
        row_vals = " ".join(f"{int(v):2d}" for v in board[r])
        print(row_vals)
    print()


def main():
    cfg_multi = BoardConfig(
        H=5,
        W=5,
        n_digits=3,
        n_addends=3,
        carry_row=0,
        top_row=1,
        bottom_row=3,
        result_row=4,
    )

    model = load_multi_add_model(cfg_multi)

    a = 111
    b = 222
    c = 333
    xs = np.array([a, b, c], dtype=np.int64)

    pred, board = run_model_multi_addition(model, cfg_multi, xs)
    true_res = int(a + b + c)

    print("Operands:")
    print(f"a = {a}")
    print(f"b = {b}")
    print(f"c = {c}")
    print()
    print(f"True sum   : {true_res}")
    print(f"Model sum  : {pred}")
    print()
    print("Final board (token ids):")
    print_board(board)


if __name__ == "__main__":
    main()
