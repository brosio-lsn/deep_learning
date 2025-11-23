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
    "Input", "Step", "Output", "carry",
    "0", "1", "2", "3", "4", "5", "6", "7", "8", "9",
    "+", "=", ".", ",", ":", "PAD"
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

      Input : 0 1 1 2 + 0 2 3 5 .
      Step 1 : 2 + 5 + 0 = 7 , carry 0 .
      Step 2 : 1 + 3 + 0 = 4 , carry 0 .
      Step 3 : 1 + 2 + 0 = 3 , carry 0 .
      Output : 0 3 4 7 .

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
    add("Input")
    add(":")
    for d in reversed(top_digits):
        add(str(d))
    add("+")
    for d in reversed(bot_digits):
        add(str(d))
    add(".")

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

        # "Step k : a + b + cin = s , carry cout ."
        add("Step")
        add(str(step_idx + 1))
        add(":")
        add(str(a))
        add("+")
        add(str(b))
        add("+")
        add(str(cin))
        add("=")
        # result digit token (supervised)
        add(str(s), supervise=True)
        add(",")
        add("carry")
        # carry digit token (supervised)
        add(str(cout), supervise=True)
        add(".")

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

    # Map tokens to ids
    token_ids = [_encode_token(t) for t in tokens]
    return token_ids, mask


def build_block_mask(seq_len: int, this_step_start: int):
    #avoid tokens from the same current step to attend to each other
    mask = torch.zeros((seq_len, seq_len))
    for q in range(this_step_start, seq_len):
        for k in range(this_step_start, seq_len):
            if k != q:                     # allow self-attention
                mask[q, k] = float("-inf")
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

    def __getitem__(self, idx: int):

        """
        # Map global idx to (problem_idx, step_idx)
        problem_idx = idx // self.n_steps
        step_idx = idx % self.n_steps  # 0 .. n_digits-1

        problem = self.problems[problem_idx]
        token_ids, mask_full = encode_cot_full(problem)  # length L

        # --- Find all 'Step' token positions (starts of steps) ---
        step_positions = [i for i, tok_id in enumerate(token_ids)
                          if ID2TOK[tok_id] == "Step"]

        assert len(step_positions) == self.n_steps, "Expected one 'Step' per column."

        # This step's 'Step' token index:
        this_step_start = step_positions[step_idx]

        # Prefix up to previous step: tokens before this 'Step'
        # For step 0 (Step1), this is just the Input line.
        prefix_end = this_step_start  # exclusive
        prefix_token_ids = token_ids[:prefix_end]

        # --- Identify supervised positions (result & carry) for this step ---
        supervised_positions = [i for i, m in enumerate(mask_full) if m]
        # 2 supervised tokens per step: (result_digit, carry_digit)
        digit_pos = supervised_positions[2 * step_idx]
        carry_pos = supervised_positions[2 * step_idx + 1]

        # --- Full sequence up to and including this step's carry digit ---
        step_end = carry_pos  # inclusive index
        full_until_step = token_ids[:step_end + 1]

        # Target is shifted version (standard next-token prediction)
        target_ids = full_until_step[1:]

        # Build target mask: True only where the *target* token is this step's
        # result digit or carry digit. A target position j corresponds to
        # full-sequence index i = j+1.
        mask_tgt_list: List[bool] = []
        for j in range(len(target_ids)):
            full_idx = j + 1  # index in full_until_step / token_ids
            if full_idx == digit_pos or full_idx == carry_pos:
                mask_tgt_list.append(True)
            else:
                mask_tgt_list.append(False)
        

        attn_mask = build_block_mask(seq_ids, this_step_start)

        # Input is prefix only (no tokens from this step)
        input_ids_t = torch.tensor(prefix_token_ids, dtype=torch.long)
        target_ids_t = torch.tensor(target_ids, dtype=torch.long)
        mask_tgt_t = torch.tensor(mask_tgt_list, dtype=torch.bool)

        return {
            "input_ids":  input_ids_t,   # (L_prefix,)
            "target_ids": target_ids_t,  # (L_step-1,)
            "mask":       mask_tgt_t,    # (L_step-1,), exactly 2 True entries
        }
        """

        problem_idx = idx // self.n_steps
        step_idx = idx % self.n_steps  # 0..n_steps-1 

        token_ids, mask_full = self.encodings[problem_idx]  # length L full CoT

        # Find all 'Step' starts
        step_positions = [i for i, tok_id in enumerate(token_ids)
                        if ID2TOK[tok_id] == "Step"]

        # Range for Input
        input_end = step_positions[0]

        # Range for all completed steps (including this one)
        this_step_start = step_positions[step_idx]

        supervised_positions = [i for i, m in enumerate(mask_full) if m]
        digit_pos = supervised_positions[2 * step_idx]
        carry_pos = supervised_positions[2 * step_idx + 1]
        # Full sequence up to this step

        labels = token_ids[:carry_pos+1]  # NOT shifted
        inputs =  labels.copy()
        inputs[this_step_start:] = VOCAB["PAD"]
        seq_len = len(labels)

        # Build loss mask aligned with labels
        loss_mask = [False] * seq_len
        for j in range(seq_len):
            if j == digit_pos or j == carry_pos:
                loss_mask[j] = True

        # Build attention mask
        attn_mask = build_block_mask(seq_len, this_step_start)

        return {
            "input_ids": torch.tensor(inputs, dtype=torch.long), # (L_prefix + L_step, ) # real ids for prefix tokens, PAD IDS for the current step tokens
            "label_ids": torch.tensor(labels, dtype=torch.long), # (L_prefix + L_step, ) # all real ids
            "loss_mask": torch.tensor(loss_mask, dtype=torch.bool), # (L_prefix + L_step, ) # only true for resultig digit and carry of the current step
            "attn_mask": attn_mask,   # (L_prefix + L_step, L_prefix + L_step) float tensor with -inf for positions a token cannot attend
            "this_step_start": this_step_start
        }
