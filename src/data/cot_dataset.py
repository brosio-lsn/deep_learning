# src/data/cot_dataset.py

from typing import List, Tuple

import torch
from torch.utils.data import Dataset

from src.data.problems import AdditionProblem
from src.data.addition_algo import number_to_digits


# ---------------------------------------------------------------------------
# Simple word-level CoT vocabulary
# ---------------------------------------------------------------------------

COT_VOCAB_TOKENS: List[str] = [
    "0", "1", "2", "3", "4", "5", "6", "7", "8", "9",
    "+", "=", ",", "PAD"
]

VOCAB = {tok: i for i, tok in enumerate(COT_VOCAB_TOKENS)}
ID2TOK = {i: tok for tok, i in VOCAB.items()}


def _encode_token(tok: str) -> int:
    if tok not in VOCAB:
        raise ValueError(f"Unknown CoT token: {tok}")
    return VOCAB[tok]


# ---------------------------------------------------------------------------
# Full CoT encoding with a mask on result / carry digits
# ---------------------------------------------------------------------------

def encode_cot_full(problem: AdditionProblem) -> Tuple[List[int], List[bool]]:
    """
    Encode a single AdditionProblem as a full Chain-of-Thought token sequence,
    together with a boolean mask that is True exactly at positions where
    the token corresponds to:
      - a result digit for a column, or
      - a carry digit for a column.

    Sequence format, e.g. (with leading zeros to match fixed-width grid):

      1 1 2 + 2 3 5
      2 + 5 + 0 = 7 , 0 
      1 + 3 + 0 = 4 , 0 
      1 + 2 + 0 = 3 , 0 


    mask[i] is True on the '7' and '0' in Step1, '4' and '0' in Step2, etc.
    """
    xs = problem.operands
    cfg = problem.cfg
    assert cfg.n_addends == 2, "CoT encoder currently assumes 2 addends."

    top_val = int(xs[0])
    bot_val = int(xs[1])
    base = cfg.base

    # Compute column-wise digits (least significant first), with fixed length
    top_digits = number_to_digits(top_val, cfg.n_digits, base=base)
    bot_digits = number_to_digits(bot_val, cfg.n_digits, base=base)

    tokens: List[str] = []
    mask: List[bool] = []

    def add(tok: str, supervise: bool = False):
        tokens.append(tok)
        mask.append(supervise)

    # ---- Input line ----
    # "Input : <n_digits digits of top> + <n_digits digits of bottom> ."
    # Use leading zeros, most-significant digit first (reverse of number_to_digits).
    for d in reversed(top_digits):
        add(str(d))
    add("+")
    for d in reversed(bot_digits):
        add(str(d))

    # ---- Per-column steps (least-significant column is Step 1) ----
    carry = 0
    for step_idx in range(cfg.n_digits):
        a = top_digits[step_idx]   # least-significant first
        b = bot_digits[step_idx]
        cin = carry
        column_sum = a + b + cin
        s = column_sum % base
        cout = column_sum // base
        carry = cout

        # "a + b + cin = s , cout"
        add(str(a))
        add("+")
        add(str(b))
        add("+")
        add(str(cin))
        add("=")
        # result digit token (supervised)
        add(str(s), supervise=True)
        add(",")
        add(str(cout), supervise=True)


    """
    # ---- Final output line ----
    # We do NOT supervise these digits here (already supervised per column).
    out_val = top_val + bot_val
    out_digits = [int(ch) for ch in str(out_val)]

    add("Output")
    add(":")
    if len(out_digits) == 3:
        add(str(0), supervise=False)
    for d in out_digits:
        add(str(d), supervise=False)
    add(".")
    """

    # Map tokens to ids
    token_ids = [_encode_token(t) for t in tokens]
    return token_ids, mask


def build_block_mask(seq_len: int, this_step_start: int):
    #avoid tokens from the same current step to attend to each other
    mask = torch.ones((seq_len, seq_len))
    for q in range(this_step_start, seq_len):
        for k in range(this_step_start, seq_len):
            if k != q:                     # allow self-attention
                mask[q, k] = 0
    return mask.bool()


def build_decoder_mask(seq_len: int):
    mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1)  # 1 above diagonal
    mask = mask.masked_fill(mask == 1, float("-inf"))
    return mask


# ---------------------------------------------------------------------------
# Step-wise CoT dataset: each sample is C_{t-1} -> C_t
# ---------------------------------------------------------------------------

class CoTAdditionDataset(Dataset):
    """
    Step-wise dataset for Chain-of-Thought addition.

    For each AdditionProblem with cfg.n_digits columns, we create cfg.n_digits
    samples (one per step). For step t (1-based), the sample is:

        input_ids  ~ tokens for C_{t-1}   (prefix up to previous step)
        target_ids ~ tokens for C_t[1:]   (full sequence up to step t, shifted)
        mask       ~ mask on target_ids, True only for result/carry digits
                     of step t.

    This matches the intuition:
      - when predicting Step 1, the input only contains the Input line,
        and the target contains "Input ... Step 1 : ...",
      - when predicting Step 2, the input contains Input + Step1, etc.

    Note: input_ids and target_ids can have different lengths; training code
    must handle that (e.g. by using seq2seq-style heads rather than simple
    same-length masked LM).
    """

    def __init__(self, problems: List[AdditionProblem]):
        self.problems = problems
        self.encodings = {}
        for i in range(len(problems)):
            self.encodings[i] = encode_cot_full(problems[i])
        if len(problems) == 0:
            raise ValueError("CoTAdditionDataset requires at least one problem.")
        self.cfg = problems[0].cfg
        self.n_steps = self.cfg.n_digits  # one step per digit column

    def __len__(self) -> int:
        return len(self.problems) * self.n_steps
        #return len(self.problems)

    def __getitem__(self, idx: int):
        """
        problem_idx = idx // self.n_steps
        step_idx = idx % self.n_steps  # 0..n_steps-1 

        token_ids, mask_full = self.encodings[problem_idx]  # length L full CoT

        # Find all ',' starts
        comma_positions = [i for i, tok_id in enumerate(token_ids)
                        if ID2TOK[tok_id] == ","]

  
        # Range for all completed steps (including this one)
        comma_position = comma_positions[step_idx]

        supervised_positions = [i for i, m in enumerate(mask_full) if m]
        digit_pos = supervised_positions[2 * step_idx]
        carry_pos = supervised_positions[2 * step_idx + 1]
        # Full sequence up to this step

        labels = token_ids[:carry_pos+2] 
        inputs =  labels.copy()

        # shifts for decoder
        labels = labels[1:] 
        inputs = inputs[:-1]

        seq_len = len(inputs)

        # Build loss mask aligned with labels
        loss_mask = [True] * seq_len
        # we do not care about prompt
        input_len = self.cfg.n_digits * 2 + 1 
        loss_mask[:input_len] = [False] * input_len
        #loss_mask[this_step_start:] = [True] * (seq_len - this_step_start)
        # Build attention mask
        attn_mask = build_decoder_mask(seq_len)

        return {
            "input_ids": torch.tensor(inputs, dtype=torch.long), # (L_prefix + L_step, ) # all real ids (without last one)
            "label_ids": torch.tensor(labels, dtype=torch.long), # (L_prefix + L_step, ) # all real ids (without first one)
            "loss_mask": torch.tensor(loss_mask, dtype=torch.bool), # (L_prefix + L_step, ) # only true for current step
            "attn_mask": attn_mask,   # (L_prefix + L_step, L_prefix + L_step) float tensor with -inf for positions a token cannot attend
            "digit_pos": digit_pos -  1,
            "carry_pos": carry_pos - 1,
            "this_step_start": this_step_start
        }
        """
        problem_idx = idx // self.n_steps
        step_idx = idx % self.n_steps  # 0..n_steps-1 

        token_ids, mask_full = self.encodings[problem_idx]  # length L full CoT

        comma_positions = [i for i, tok_id in enumerate(token_ids)
                        if ID2TOK[tok_id] == ","]

        # Range for all completed steps (including this one)
        comma_position = comma_positions[step_idx]
        digit_pos = comma_position - 1
        carry_pos = comma_position + 1
        this_step_start = comma_position - 7

        token_ids = token_ids[:(carry_pos + 1)]
        # shifts for decoder
        labels = token_ids[1:] 
        inputs = token_ids[:-1]

        seq_len = len(inputs)

        # Build loss mask aligned with labels
        loss_mask = [False] * seq_len
        # we do not care about prompt
        loss_mask[this_step_start:] = [True] * (seq_len - this_step_start)
        #loss_mask[this_step_start:] = [True] * (seq_len - this_step_start)
        # Build attention mask
        attn_mask = build_decoder_mask(seq_len)

        return {
            "input_ids": torch.tensor(inputs, dtype=torch.long), # (L_prefix + L_step, ) # all real ids (without last one)
            "label_ids": torch.tensor(labels, dtype=torch.long), # (L_prefix + L_step, ) # all real ids (without first one)
            "loss_mask": torch.tensor(loss_mask, dtype=torch.bool), # (L_prefix + L_step, ) # only true for current step
            "attn_mask": attn_mask,   # (L_prefix + L_step, L_prefix + L_step) float tensor with -inf for positions a token cannot attend
            "digit_pos": digit_pos -  1,
            "carry_pos": carry_pos - 1,
            "this_step_start": this_step_start
        }
