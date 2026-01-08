from src.models.positional_encodings import (
    LearnedPositionalEncoding1D,
    SinusoidalPositionalEncoding,
    LearnedPositionalEncoding2D,
    Abs2DPlusRelBias2D,
    SinusoidalPositionalEncoding2D,
    RelativePositionBias2D,
)

def make_pes(model_cfg, board_cfg):
    return [
        ("abs_1d_learned",
         LearnedPositionalEncoding1D(model_cfg.d_model, board_cfg.H * board_cfg.W)),
        ("abs_1d_sinusoidal",
         SinusoidalPositionalEncoding(model_cfg.d_model, model_cfg.max_len)),
        ("abs_2d_learned",
         LearnedPositionalEncoding2D(model_cfg.d_model, board_cfg.H, board_cfg.W)),
        ("abs_2d_sin+rel_2d_bias",
         Abs2DPlusRelBias2D(
             abs_pe=SinusoidalPositionalEncoding2D(model_cfg.d_model, board_cfg.H, board_cfg.W),
             rel_bias=RelativePositionBias2D(model_cfg.nhead, board_cfg.H, board_cfg.W),
         )),
        ("abs_2d_sinusoidal",
         SinusoidalPositionalEncoding2D(model_cfg.d_model, board_cfg.H, board_cfg.W)),
        ("rel_2d_bias",
         RelativePositionBias2D(model_cfg.nhead, board_cfg.H, board_cfg.W)),
    ]
