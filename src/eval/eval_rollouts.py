import os
import argparse
import numpy as np
import torch


from src.data.addition_algo import BoardConfig, generate_trajectory_variant_A, VOID_TOKEN, BLANK_TOKEN, VOCAB_SIZE
from src.data.problems import generate_diversified_problems
from src.models.transformers import BlackboardTransformer
from src.models.positional_encodings import (
    LearnedPositionalEncoding1D,
    SinusoidalPositionalEncoding,
    LearnedPositionalEncoding2D,
    Abs2DPlusRelBias2D,
    SinusoidalPositionalEncoding2D,
    RelativePositionBias2D,
)
from src.training.configs import ModelConfig


def stepwrite_mask(cfg: BoardConfig, step_idx: int) -> torch.Tensor:
    H, W = cfg.H, cfg.W
    L = H * W
    m = torch.zeros(L, dtype=torch.bool)
    col_end = W - 1
    col = col_end - step_idx
    if 0 <= col < W:
        m[cfg.result_row * W + col] = True
    if 0 <= (col - 1) < W:
        m[cfg.carry_row * W + (col - 1)] = True
    return m


def prev_step_mask(cfg: BoardConfig, step_idx: int) -> torch.Tensor:
    H, W = cfg.H, cfg.W
    L = H * W
    m = torch.zeros(L, dtype=torch.bool)
    if step_idx <= 0:
        return m
    col_end = W - 1
    col_prev = col_end - (step_idx - 1)
    if 0 <= col_prev < W:
        m[cfg.result_row * W + col_prev] = True
    if 0 <= (col_prev - 1) < W:
        m[cfg.carry_row * W + (col_prev - 1)] = True
    return m


def editable_mask_global(cfg: BoardConfig) -> torch.Tensor:
    H, W = cfg.H, cfg.W
    L = H * W
    m = torch.zeros(L, dtype=torch.bool)
    for r in [cfg.carry_row, cfg.result_row]:
        m[r * W : r * W + W] = True
    return m


def inject_rollout_noise_inplace(board: torch.Tensor, cfg: BoardConfig, step_idx: int, rng: torch.Generator, p_noise: float, n_noise: int = 1, noise_kind: str = "flip_digit", allow_void: bool = True):
    """
    Inject mistakes into the CURRENT board during rollout

    We corrupt already-written cells only (result + carry that correspond to steps < step_idx)
    - step_idx = current step pointer t
    - if step_idx == 0, nothing is written yet => no noise injected.

    noise_kind:
      - "flip_digit": if cell is 0..9, replace by a different digit (most realistic)
      - "random_digit": replace by random digit 0..9
      - "blank": set to BLANK_TOKEN
      - "void": set to VOID_TOKEN (only if allow_void=True / vocab supports it, e.g not for classic setting)
    """
    if p_noise <= 0.0 or step_idx <= 0:
        return

    # with prob p_noise, inject n_noise corruptions
    u = torch.rand((), generator=rng, device=board.device).item()
    if u > p_noise:
        return

    H, W = cfg.H, cfg.W

    # Candidate positions: already-written result columns and carry columns
    # Result col for step s is (W-1 - s)
    # Carry col for step s is (W-2 - s)  (if in bounds)
    cand = []

    # steps written so far are s = 0..(step_idx-1)
    for s in range(step_idx):
        c_res = (W - 1) - s
        if 0 <= c_res < W:
            cand.append(cfg.result_row * W + c_res)

        c_car = (W - 2) - s
        if 0 <= c_car < W:
            cand.append(cfg.carry_row * W + c_car)

    if len(cand) == 0:
        return

    # Apply up to n_noise corruptions (with replacement)
    for _ in range(n_noise):
        j = int(torch.randint(low=0, high=len(cand), size=(1,), generator=rng, device=board.device).item())
        pos = cand[j]
        old = int(board[pos].item())

        if noise_kind == "flip_digit":
            # If it's a digit, flip to a different digit; otherwise set a random digit
            if 0 <= old <= 9:
                d = int(torch.randint(0, 9, (1,), generator=rng, device=board.device).item())
                # map 0..8 to a digit != old
                new = d if d < old else d + 1
            else:
                new = int(torch.randint(0, 10, (1,), generator=rng, device=board.device).item())
            board[pos] = new

        elif noise_kind == "random_digit":
            new = int(torch.randint(0, 10, (1,), generator=rng, device=board.device).item())
            board[pos] = new

        elif noise_kind == "blank":
            board[pos] = BLANK_TOKEN

        elif noise_kind == "void":
            if allow_void:
                board[pos] = VOID_TOKEN
            else:
                board[pos] = BLANK_TOKEN

        else:
            raise ValueError(f"Unknown noise_kind: {noise_kind}")


@torch.no_grad()
def rollout_one(model, cfg: BoardConfig, xs: np.ndarray, setting: str, max_iters: int = 50, p_noise: float = 0.0, n_noise: int = 1, noise_kind: str = "flip_digit", allow_void_noise: bool = False, seed: int = 0) -> torch.Tensor:
    # teacher gives us S0 and final target to compare against
    device = next(model.parameters()).device
    rng_torch = torch.Generator(device=device)
    rng_torch.manual_seed(seed)

    S_seq, _ = generate_trajectory_variant_A(cfg, xs)
    board = torch.from_numpy(S_seq[0]).view(-1).long().to(device)

    t = 0
    iters = 0
    W = cfg.W
    col_end = W - 1

    while t < cfg.n_digits and iters < max_iters:
        iters += 1
        inject_rollout_noise_inplace(
            board=board,
            cfg=cfg,
            step_idx=t,
            rng=rng_torch,
            p_noise=p_noise,
            n_noise=n_noise,
            noise_kind=noise_kind,
            allow_void=allow_void_noise,
        )
        logits, _ = model(board.unsqueeze(0))     # (1, L, V)
        pred = logits.argmax(dim=-1).squeeze(0)   # (L,)

        if setting == "local":
            # if model indicates revert by writing VOID in previous-step cells -> go back
            pm = prev_step_mask(cfg, t).to(board.device)
            if pm.any() and (pred[pm] == VOID_TOKEN).any():
                board[pm] = VOID_TOKEN
                t = max(t - 1, 0)
                continue

            # otherwise: apply stepwrite for current step and advance
            sm = stepwrite_mask(cfg, t).to(board.device)
            board[sm] = pred[sm]
            t += 1

        else:
            # global: if it writes VOID anywhere editable, treat as "erase" and jump back
            em = editable_mask_global(cfg).to(board.device)
            void_pos = em & (pred == VOID_TOKEN)
            if void_pos.any():
                board[void_pos] = VOID_TOKEN

                # compute digit columns erased (prefer result row; if only carry erased, map carry_col+1)
                erased_cols = []
                idxs = void_pos.nonzero(as_tuple=False).view(-1).tolist()
                for idx in idxs:
                    r = idx // W
                    c = idx % W
                    if r == cfg.result_row:
                        erased_cols.append(c)
                    elif r == cfg.carry_row:
                        erased_cols.append(c + 1)

                erased_cols = [c for c in erased_cols if 0 <= c < W]
                if len(erased_cols) > 0:
                    # recompute from the rightmost erased column (smallest step index)
                    cmax = max(erased_cols)
                    t = max(0, col_end - cmax)
                else:
                    t = max(t - 1, 0)
                continue

            # otherwise advance normally
            sm = stepwrite_mask(cfg, t).to(board.device)
            board[sm] = pred[sm]
            t += 1

    return board.cpu()


def make_pe(pe_name: str, cfg_model: ModelConfig, cfg_board: BoardConfig):
    if pe_name == "abs_1d_learned":
        return LearnedPositionalEncoding1D(cfg_model.d_model, cfg_board.H * cfg_board.W)

    if pe_name == "abs_1d_sinusoidal":
        return SinusoidalPositionalEncoding(cfg_model.d_model, cfg_model.max_len)

    if pe_name == "abs_2d_learned":
        return LearnedPositionalEncoding2D(cfg_model.d_model, cfg_board.H, cfg_board.W)

    if pe_name == "abs_2d_sin+rel_2d_bias":
        return Abs2DPlusRelBias2D(
            abs_pe=SinusoidalPositionalEncoding2D(cfg_model.d_model, cfg_board.H, cfg_board.W),
            rel_bias=RelativePositionBias2D(cfg_model.nhead, cfg_board.H, cfg_board.W),
        )

    if pe_name == "abs_2d_sinusoidal":
        return SinusoidalPositionalEncoding2D(cfg_model.d_model, cfg_board.H, cfg_board.W)

    if pe_name == "rel_2d_bias":
        return RelativePositionBias2D(cfg_model.nhead, cfg_board.H, cfg_board.W)

    raise ValueError(f"Unknown pe_name: {pe_name}")



def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", type=str, required=True)
    p.add_argument("--setting", choices=["local", "global"], required=True)
    p.add_argument("--pe", choices=["abs_1d_learned", "abs_1d_sinusoidal", "abs_2d_learned", "abs_2d_sin+rel_2d_bias", "abs_2d_sinusoidal", "rel_2d_bias"], required=True)
    p.add_argument("--n-test", type=int, default=2000)
    p.add_argument("--vocab-size", type=int, default=13)
    p.add_argument("--p-noise", type=float, default=0.0, help="Probability per rollout step to inject noise into already-written cells.")
    p.add_argument("--n-noise", type=int, default=1, help="How many cells to corrupt when noise triggers.")
    p.add_argument("--noise-kind", choices=["flip_digit", "random_digit", "blank", "void"], default="flip_digit")
    p.add_argument("--allow-void-noise", action="store_true", help="Allow injecting VOID token during noise (only makes sense for vocab=13 models).")
    p.add_argument("--seed", type=int, default=42)

    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    board_cfg = BoardConfig(H=4, W=5, n_digits=3)

    # must match training cfg
    model_cfg = ModelConfig(d_model=128, nhead=4, num_layers=4, dim_feedforward=512, dropout=0.1, max_len=200)
    pe = make_pe(args.pe, model_cfg, board_cfg)

    model = BlackboardTransformer(vocab_size=args.vocab_size, pos_enc=pe, **model_cfg.__dict__).to(device)
    state = torch.load(args.ckpt, map_location=device)
    model.load_state_dict(state["model_state_dict"])
    model.eval()

    problems = generate_diversified_problems(board_cfg, args.n_test, seed=123)

    n_ok = 0
    for sample_idx, pr in enumerate(problems):
        xs = pr.operands
        # teacher final for comparison
        S_seq, _ = generate_trajectory_variant_A(board_cfg, xs)
        target_final = torch.from_numpy(S_seq[-1]).view(-1).long()

        pred_final = rollout_one(model, board_cfg, xs, setting=args.setting, max_iters=100, p_noise=args.p_noise, n_noise=args.n_noise, noise_kind=args.noise_kind, allow_void_noise=args.allow_void_noise, seed=args.seed + sample_idx)

        if torch.equal(pred_final, target_final):
            n_ok += 1

    acc = n_ok / len(problems)
    print(f"Rollout exact-final-board accuracy: {acc:.4f} ({n_ok}/{len(problems)})")

if __name__ == "__main__":
    main()
