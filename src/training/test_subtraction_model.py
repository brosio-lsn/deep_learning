import os
from typing import Tuple

import numpy as np
import torch

from src.data.addition_algo import BoardConfig
from src.data.subtraction_algo import generate_trajectory as generate_subtraction_trajectory
from src.data.problems import digits_to_number
from src.models.blackboard_transformer import BlackboardTransformer
from src.models.positional_encodings import RelativePositionBias2D

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

BASE_DIR = os.path.dirname(__file__)
CHECKPOINT_DIR = os.path.join(BASE_DIR, "trained_weights")


def build_blackboard_relative_model(cfg: BoardConfig) -> BlackboardTransformer:
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


def load_subtraction_model(cfg: BoardConfig) -> BlackboardTransformer:
    model = build_blackboard_relative_model(cfg)
    ckpt_path = os.path.join(
        CHECKPOINT_DIR,
        "blackboard_relative_pe_subtraction_lastlayer.pt",
    )
    state = torch.load(ckpt_path, map_location=DEVICE)
    model.load_state_dict(state)
    model.eval()
    return model


def run_model_subtraction(
    model: BlackboardTransformer,
    cfg: BoardConfig,
    top: int,
    bottom: int,
) -> Tuple[int, np.ndarray]:
    xs = np.array([top, bottom], dtype=np.int64)
    S_seq, M_seq = generate_subtraction_trajectory(cfg, xs)

    S_t = S_seq[0].copy()

    for step in range(cfg.n_digits):
        board_flat = torch.from_numpy(S_t.astype(np.int64)).view(1, -1).to(DEVICE)
        with torch.no_grad():
            logits, _ = model(board_flat)

        logits_step = logits[0]  # (L, V)
        preds = logits_step.argmax(dim=-1).cpu().numpy()  # (L,)

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


def main():
    cfg = BoardConfig(H=4, W=5, n_digits=3)
    model = load_subtraction_model(cfg)

    top = 103
    bottom = 101

    pred, board = run_model_subtraction(model, cfg, top, bottom)
    true_res = top - bottom

    print(f"Top    : {top}")
    print(f"Bottom : {bottom}")
    print(f"True   : {true_res}")
    print(f"Model  : {pred}")


if __name__ == "__main__":
    main()
