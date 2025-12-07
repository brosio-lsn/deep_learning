from typing import List, Tuple, Iterable, Dict
import warnings
import numpy as np

from src.data.addition_algo import BoardConfig, number_to_digits
from src.data.problems import AdditionProblem, digits_to_number


Triplet = Tuple[int, int, int]  # (carry_in, top_digit, bottom_digit)


# ---------------------------------------------------------------------------
# Helper: extract per-column (carry_in, a, b) triplets for ONE problem
# ---------------------------------------------------------------------------

def extract_triplets(cfg: BoardConfig, xs: np.ndarray) -> List[Triplet]:
    """
    For a 2-addend problem, compute per-column (carry_in, a, b) triplets.
    Least-significant column is index 0 (consistent with number_to_digits).
    """
    assert cfg.n_addends == 2, "extract_triplets assumes 2 addends."
    base = cfg.base
    top_val, bot_val = int(xs[0]), int(xs[1])

    top_digits = number_to_digits(top_val, cfg.n_digits, base=base)
    bot_digits = number_to_digits(bot_val, cfg.n_digits, base=base)

    triplets: List[Triplet] = []
    carry = 0
    for col in range(cfg.n_digits):
        a = top_digits[col]
        b = bot_digits[col]
        cin = carry
        s = a + b + cin
        carry = s // base
        triplets.append((cin, a, b))
    return triplets


def _check_budget(target: int, got: int, what: str):
    if got < target:
        warnings.warn(
            f"[sample_efficiency] Only generated {got} {what} "
            f"(requested {target}). Perhaps constraints are too strict."
        )


# ---------------------------------------------------------------------------
# Exhaustive enumeration over ALL top,bottom pairs (small n_digits only)
# ---------------------------------------------------------------------------

def enumerate_all_problems(cfg: BoardConfig) -> List[AdditionProblem]:
    """
    Enumerate ALL possible problems (top, bottom) for given n_digits and base.

    Number of pairs = (base^n_digits)^2, so only safe for small n_digits
    (e.g. 3).
    """
    base = cfg.base
    n_digits = cfg.n_digits
    assert n_digits <= 3, (
        "enumerate_all_problems is only intended for small n_digits (<=3)."
    )

    max_val = base ** n_digits
    problems: List[AdditionProblem] = []
    for top in range(max_val):
        for bottom in range(max_val):
            xs = np.array([top, bottom], dtype=np.int64)
            problems.append(AdditionProblem(xs, cfg))
    return problems


# ======================================================================
# Setting 1: simple random subset from all possible 3-digit problems
# ======================================================================

def generate_setting1_random_fraction(
    cfg: BoardConfig,
    n_train: int,
    n_test: int,
    seed: int,
) -> Tuple[List[AdditionProblem], List[AdditionProblem]]:
    """
    Setting 1 (baseline sample efficiency):

      - Enumerate all possible problems (top, bottom) for this cfg.
      - Shuffle them.
      - Take first n_train for train, next n_test for test.

    Intended for cfg.n_digits <= 3.
    """
    assert cfg.n_digits <= 3, "Setting 1 is only meant for small n_digits (<=3)."

    all_problems = enumerate_all_problems(cfg)
    total = len(all_problems)

    rng = np.random.default_rng(seed)
    indices = np.arange(total)
    rng.shuffle(indices)

    n_total_needed = n_train + n_test
    if n_total_needed > total:
        warnings.warn(
            f"[setting1] Requested {n_total_needed} total problems but only "
            f"{total} exist. Using all {total} instead."
        )
        n_total_needed = total

    train_end = min(n_train, n_total_needed)
    test_end = min(n_total_needed - train_end, n_test)

    train_idx = indices[:train_end]
    test_idx = indices[train_end:train_end + test_end]

    train_problems = [all_problems[i] for i in train_idx]
    test_problems  = [all_problems[i] for i in test_idx]

    _check_budget(n_train, len(train_problems), "train problems (setting1)")
    _check_budget(n_test,  len(test_problems),  "test problems  (setting1)")

    return train_problems, test_problems


# ======================================================================
# Setting 2: position split with triplets_of_interest, generative version
# ======================================================================

def _sample_train_problem_setting2(
    cfg: BoardConfig,
    rng: np.random.Generator,
    T: List[Triplet],
    col_to_triplets: Dict[int, List[Triplet]],
) -> AdditionProblem:
    """
    Sample ONE training problem for setting 2.

    For each column:
      - Let carry_in be determined by previous columns.
      - Let allowed triplets at this column be those in col_to_triplets[col]
        whose cin == carry_in.
      - If no such allowed triplet:
          pick random digits (a,b) ~ Uniform({0..base-1}), consistent with carry_in.
        Else:
          with prob 0.5: same random digits,
          with prob 0.5: pick one of the allowed triplets (cin,a,b).

    So triplets_of_interest appear ONLY in columns they were assigned to
    (via allowed_cols), but other columns can use arbitrary (cin,a,b).
    """
    base = cfg.base
    n_digits = cfg.n_digits

    top_digits: List[int] = []
    bot_digits: List[int] = []
    carry_in = 0

    for col in range(n_digits):
        # Triplets of interest allowed at this column (matching carry_in)
        candidates = [
            t for t in col_to_triplets.get(col, [])
            if t[0] == carry_in
        ]

        use_special = False
        if candidates:
            if rng.random() < 0.5:
                use_special = True

        if use_special:
            idx = int(rng.integers(0, len(candidates)))
            cin, a, b = candidates[idx]
            # cin should equal carry_in by construction
        else:
            # Random digits, arbitrary (not constrained to T)
            a = int(rng.integers(0, base))
            b = int(rng.integers(0, base))
            cin = carry_in

        s = a + b + cin
        carry_out = s // base

        top_digits.append(a)
        bot_digits.append(b)
        carry_in = carry_out

    top_arr = np.array(top_digits, dtype=np.int64)
    bot_arr = np.array(bot_digits, dtype=np.int64)
    top_val = digits_to_number(top_arr, base=base)
    bot_val = digits_to_number(bot_arr, base=base)
    xs = np.array([top_val, bot_val], dtype=np.int64)
    return AdditionProblem(xs, cfg)


def _sample_test_problem_setting2(
    cfg: BoardConfig,
    rng: np.random.Generator,
    T: List[Triplet],
    col_to_triplets: Dict[int, List[Triplet]],
) -> AdditionProblem:
    """
    Sample ONE test problem for setting 2.

    For each column:
      - Let carry_in be determined by previous columns.
      - Let forbidden triplets at this column be those in T that:
          * are NOT in col_to_triplets[col], and
          * have cin == carry_in.
      - With prob 0.5 and if forbidden candidates exist:
          pick one of these forbidden triplets (cin,a,b) → "novel position".
        Else:
          pick random digits (a,b) ~ Uniform({0..base-1}), consistent with cin.

    We additionally require that at least ONE column in the problem used
    a forbidden triplet (i.e., a triplet_of_interest at a new column index);
    otherwise we resample.
    """
    base = cfg.base
    n_digits = cfg.n_digits
    T_set = set(T)

    # Precompute "train-allowed" set for quick membership checks
    col_allowed_set: Dict[int, set] = {
        col: set(trips) for col, trips in col_to_triplets.items()
    }

    while True:
        top_digits: List[int] = []
        bot_digits: List[int] = []
        carry_in = 0
        used_forbidden = False

        for col in range(n_digits):
            allowed_here = col_allowed_set.get(col, set())

            # Triplets in T that are NOT allowed here and whose cin matches
            forbidden_candidates = [
                t for t in T
                if t not in allowed_here and t[0] == carry_in
            ]

            use_forbidden = False
            if forbidden_candidates and (rng.random() < 0.5):
                use_forbidden = True

            if use_forbidden:
                idx = int(rng.integers(0, len(forbidden_candidates)))
                cin, a, b = forbidden_candidates[idx]
                used_forbidden = True
            else:
                a = int(rng.integers(0, base))
                b = int(rng.integers(0, base))
                cin = carry_in

            s = a + b + cin
            carry_out = s // base

            top_digits.append(a)
            bot_digits.append(b)
            carry_in = carry_out

        if not used_forbidden:
            # This sample never used a "new-position" triplet → resample
            continue

        top_arr = np.array(top_digits, dtype=np.int64)
        bot_arr = np.array(bot_digits, dtype=np.int64)
        top_val = digits_to_number(top_arr, base=base)
        bot_val = digits_to_number(bot_arr, base=base)
        xs = np.array([top_val, bot_val], dtype=np.int64)
        return AdditionProblem(xs, cfg)


def generate_setting2_position_split(
    cfg: BoardConfig,
    n_train: int,
    n_test: int,
    seed: int,
    triplets_of_interest: Iterable[Triplet],
    frac_positions: float,
) -> Tuple[List[AdditionProblem], List[AdditionProblem], Dict[Triplet, set]]:
    """
    Setting 2 (generative version):

      We have a small set T of triplets_of_interest, e.g.
          T = [(0,5,5), (1,6,6), ...]

      For EACH t in T, we choose a subset A_t of column indices that are
      "allowed" positions for that triplet in TRAIN, of size
          floor(frac_positions * n_digits), at least 1.

      Then:

      - TRAIN generation:
          For each column col, we know which triplets_of_interest are
          allowed there: map[col] = { t in T | col in A_t }.

          We build each problem column by column, with carry consistency:
            * carry_in is propagated from previous columns;
            * if map[col] has some triplets whose cin == carry_in:
                - with prob 0.5: pick random digits (a,b);
                - with prob 0.5: pick one of those allowed triplets;
              otherwise (no candidate) → random digits.

          So triplets_of_interest only appear (if at all) in columns A_t.

      - TEST generation:
          We want to see triplets_of_interest at columns where they were
          *not* allowed in train. For a column col with carry_in, define:

              forbidden_candidates(col) = { t in T \ map[col] | t.cin == carry_in }

          For each column:
            * with prob 0.5 and if forbidden_candidates(col) non-empty:
                pick a t from forbidden_candidates(col);
              else:
                pick random digits (a,b) (not constrained to T).

          We also enforce that at least one column in the problem actually
          used a forbidden triplet (otherwise the sample is resampled).

      Returns:
        train_problems, test_problems, allowed_cols

      allowed_cols: dict mapping t ∈ T → set of column indices A_t used for train.
    """
    assert cfg.n_addends == 2
    rng = np.random.default_rng(seed)
    T = list(triplets_of_interest)

    # 1) For each triplet t in T, choose its allowed TRAIN columns A_t
    cols = np.arange(cfg.n_digits)
    allowed_cols: Dict[Triplet, set] = {}
    for t in T:
        n_allowed = max(1, int(cfg.n_digits * frac_positions))
        choice = rng.choice(cols, size=n_allowed, replace=False)
        allowed_cols[t] = set(int(c) for c in choice)

    # Build col -> list of triplets_of_interest allowed in that column
    col_to_triplets: Dict[int, List[Triplet]] = {col: [] for col in range(cfg.n_digits)}
    for t, cols_set in allowed_cols.items():
        for col in cols_set:
            col_to_triplets[col].append(t)

    # 2) Generate train problems
    train_problems: List[AdditionProblem] = []
    for _ in range(n_train):
        prob = _sample_train_problem_setting2(cfg, rng, T, col_to_triplets)
        train_problems.append(prob)

    # 3) Generate test problems
    test_problems: List[AdditionProblem] = []
    max_tries = 50 * n_test if n_test > 0 else 0
    tries = 0
    while len(test_problems) < n_test and tries < max_tries:
        tries += 1
        prob = _sample_test_problem_setting2(cfg, rng, T, col_to_triplets)
        test_problems.append(prob)

    _check_budget(n_train, len(train_problems), "train problems (setting2)")
    _check_budget(n_test,  len(test_problems),  "test problems  (setting2)")

    return train_problems, test_problems, allowed_cols


# ======================================================================
# Setting 3: order-of-triplets constraint (train pattern vs test pattern)
# ======================================================================

# ======================================================================
# Setting 3: order-of-triplets pattern vs random (generative version)
# ======================================================================

def generate_setting3_order_constraint(
    cfg: BoardConfig,
    n_train: int,
    n_test: int,
    seed: int,
    pattern_train: List[Triplet],
    pattern_test: List[Triplet],
) -> Tuple[List[AdditionProblem], List[AdditionProblem]]:
    """
    Setting 3 (generative):

      - pattern_train: e.g. [tA, tB, tC]
      - pattern_test:  e.g. [tA, tC, tB]
        where each tX = (cin, a, b) is a triplet for a *column*.

    For train and test we do NOT enforce that the whole prefix equals
    the pattern. Instead:

      • For TRAIN:
          For each column col:
            - Keep track of carry_in from previous columns.
            - If col < len(pattern_train) and pattern_train[col].cin == carry_in:
                with probability 0.5 → use the pattern triplet at this column;
                with probability 0.5 → sample (a,b) randomly.
              Else:
                always sample (a,b) randomly.
            - Update carry_out from a + b + cin.

      • For TEST:
          Same logic, but using pattern_test instead of pattern_train.

    So each column in the first len(pattern_*) positions is a 50/50 mix of:
      - pattern triplet
      - random triplet
    (subject to carry consistency).
    Columns beyond len(pattern_*) are fully random.

    This creates a distributional bias where train and test have different
    *preferred* column-wise patterns, but both still contain random columns.
    """
    assert cfg.n_addends == 2
    base = cfg.base
    n_digits = cfg.n_digits
    rng = np.random.default_rng(seed)

    def _sample_from_pattern(pattern: List[Triplet]) -> AdditionProblem:
        top_digits: List[int] = []
        bot_digits: List[int] = []
        carry_in = 0

        for col in range(n_digits):
            use_pattern = False
            if col < len(pattern):
                t_cin, t_a, t_b = pattern[col]
                if t_cin == carry_in:
                    # 50% chance to actually use the pattern here
                    if rng.random() < 0.5:
                        use_pattern = True

            if use_pattern:
                cin, a, b = pattern[col]
            else:
                # Random digits, arbitrary (consistent only with current carry_in)
                a = int(rng.integers(0, base))
                b = int(rng.integers(0, base))
                cin = carry_in

            s = a + b + cin
            carry_out = s // base

            top_digits.append(a)
            bot_digits.append(b)
            carry_in = carry_out

        top_arr = np.array(top_digits, dtype=np.int64)
        bot_arr = np.array(bot_digits, dtype=np.int64)
        top_val = digits_to_number(top_arr, base=base)
        bot_val = digits_to_number(bot_arr, base=base)
        xs = np.array([top_val, bot_val], dtype=np.int64)
        return AdditionProblem(xs, cfg)

    # Generate train and test samples
    train_problems: List[AdditionProblem] = [
        _sample_from_pattern(pattern_train) for _ in range(n_train)
    ]
    test_problems: List[AdditionProblem] = [
        _sample_from_pattern(pattern_test) for _ in range(n_test)
    ]

    return train_problems, test_problems


# ======================================================================
# Setting 4: full triplet hold-out (never seen in train, required in test)
# ======================================================================

# ======================================================================
# Setting 4: full triplet hold-out (generative version)
# ======================================================================

def generate_setting4_triplet_holdout(
    cfg: BoardConfig,
    n_train: int,
    n_test: int,
    seed: int,
    forbidden_triplets: Iterable[Triplet],
) -> Tuple[List[AdditionProblem], List[AdditionProblem]]:
    """
    Setting 4 (generative):

      Let H be a set of forbidden triplets, e.g. H = [(0, 9, 4), (1, 9, 4)].

      • TRAIN:
          Build problems column by column (respecting carry). For each column:
            - carry_in is determined by previous columns.
            - Sample random digits (a,b) uniformly in {0..base-1}^2
              until (carry_in, a, b) ∉ H.
          So NO forbidden triplet ever appears in train.

      • TEST:
          Build problems column by column (respecting carry). For each column:
            - carry_in is determined by previous columns.
            - With probability 0.5:
                sample random (a,b) with (carry_in, a, b) ∉ H.
              With probability 0.5:
                sample a random triplet from H whose cin == carry_in.
                If no such forbidden triplet exists for this carry_in,
                fall back to the "random not in H" case.

          Additionally, we enforce that each test problem contains at least
          ONE forbidden triplet somewhere; otherwise we resample that problem.

      Returns:
        train_problems, test_problems
    """
    assert cfg.n_addends == 2
    base = cfg.base
    n_digits = cfg.n_digits
    rng = np.random.default_rng(seed)

    H = set(forbidden_triplets)

    # Helper: random (a,b) such that (cin,a,b) ∉ H
    def _sample_not_in_H(cin: int) -> Tuple[int, int]:
        while True:
            a = int(rng.integers(0, base))
            b = int(rng.integers(0, base))
            if (cin, a, b) not in H:
                return a, b

    # Helper: random (cin,a,b) ∈ H with given cin, or None if impossible
    def _sample_forbidden_with_cin(cin: int) -> Tuple[int, int] | None:
        candidates = [t for t in H if t[0] == cin]
        if not candidates:
            return None
        t = candidates[int(rng.integers(0, len(candidates)))]
        _, a, b = t
        return a, b

    # --- TRAIN: only non-forbidden triplets ---
    def _sample_train_problem() -> AdditionProblem:
        top_digits: List[int] = []
        bot_digits: List[int] = []
        carry_in = 0

        for col in range(n_digits):
            a, b = _sample_not_in_H(carry_in)
            s = a + b + carry_in
            carry_out = s // base

            top_digits.append(a)
            bot_digits.append(b)
            carry_in = carry_out

        top_arr = np.array(top_digits, dtype=np.int64)
        bot_arr = np.array(bot_digits, dtype=np.int64)
        top_val = digits_to_number(top_arr, base=base)
        bot_val = digits_to_number(bot_arr, base=base)
        xs = np.array([top_val, bot_val], dtype=np.int64)
        return AdditionProblem(xs, cfg)

    # --- TEST: mix of forbidden and non-forbidden, but at least one forbidden ---
    def _sample_test_problem() -> AdditionProblem:
        while True:
            top_digits: List[int] = []
            bot_digits: List[int] = []
            carry_in = 0
            used_forbidden = False

            for col in range(n_digits):
                use_forbidden = (rng.random() < 0.5)
                if use_forbidden:
                    fb = _sample_forbidden_with_cin(carry_in)
                    if fb is not None:
                        a, b = fb
                        used_forbidden = True
                    else:
                        # No compatible forbidden triplet for this carry_in → fallback
                        a, b = _sample_not_in_H(carry_in)
                else:
                    a, b = _sample_not_in_H(carry_in)

                s = a + b + carry_in
                carry_out = s // base

                top_digits.append(a)
                bot_digits.append(b)
                carry_in = carry_out

            if not used_forbidden:
                # We want at least one forbidden triplet somewhere → resample
                continue

            top_arr = np.array(top_digits, dtype=np.int64)
            bot_arr = np.array(bot_digits, dtype=np.int64)
            top_val = digits_to_number(top_arr, base=base)
            bot_val = digits_to_number(bot_arr, base=base)
            xs = np.array([top_val, bot_val], dtype=np.int64)
            return AdditionProblem(xs, cfg)

    # Build train / test
    train_problems: List[AdditionProblem] = [
        _sample_train_problem() for _ in range(n_train)
    ]

    test_problems: List[AdditionProblem] = []
    max_tries = 50 * n_test if n_test > 0 else 0
    tries = 0
    while len(test_problems) < n_test and tries < max_tries:
        tries += 1
        test_problems.append(_sample_test_problem())

    _check_budget(n_train, len(train_problems), "train problems (setting4)")
    _check_budget(n_test,  len(test_problems),  "test problems  (setting4)")

    return train_problems, test_problems


# ======================================================================
# Small debug main
# ======================================================================

def _debug_print_problems(label: str, cfg: BoardConfig, problems: List[AdditionProblem], max_n: int = 5):
    print(f"\n=== {label} (showing up to {max_n}) ===")
    for i, prob in enumerate(problems[:max_n]):
        xs = prob.operands
        trips = extract_triplets(cfg, xs)
        print(f"  [{i}] {xs[0]} + {xs[1]}  -> triplets: {trips}")


def main():
    # Simple config: 3-digit addition, base 10
    cfg = BoardConfig(H=4, W=5, n_digits=3)

    print("######## Setting 1: random fraction ########")
    
    #train1, test1 = generate_setting1_random_fraction(cfg, n_train=10, n_test=5, seed=0)
    #_debug_print_problems("Setting 1 - train", cfg, train1)
    #_debug_print_problems("Setting 1 - test",  cfg, test1)
    

    print("\n######## Setting 2: position split with limited triplets ########")
    #cfg = BoardConfig(H=4, W=5, n_digits=10)
    # Small triplet set: only two types of columns allowed
    T2: List[Triplet] = [
        (0, 6, 5),  # carry_in 0, 5+5
        (0, 3, 4),  # carry_in 0, 3+4
    ]
    train2, test2, allowed_cols2 = generate_setting2_position_split(
        cfg,
        n_train=10,
        n_test=5,
        seed=3,
        triplets_of_interest=T2,
        frac_positions=0.9,
    )
    print(f"Allowed training columns per triplet (setting 2): {allowed_cols2}")
    #_debug_print_problems("Setting 2 - train", cfg, train2)
    #_debug_print_problems("Setting 2 - test",  cfg, test2)
    
    print("\n######## Setting 3: order constraint ########")
    # Define some artificial patterns
    tA: Triplet = (0, 1, 2)
    tB: Triplet = (0, 3, 4)
    tC: Triplet = (0, 5, 6)
    pattern_train = [tA, tB, tC]
    tA : Triplet = (0, 0,0)
    tB : Triplet = (0, 1,1)
    tC : Triplet = (0, 2,2)
    pattern_test  = [tA, tC, tB]

    train3, test3 = generate_setting3_order_constraint(
        cfg,
        n_train=5,
        n_test=5,
        seed=2,
        pattern_train=pattern_train,
        pattern_test=pattern_test,
    )
    #_debug_print_problems("Setting 3 - train", cfg, train3)
    #_debug_print_problems("Setting 3 - test",  cfg, test3)
    
    print("\n######## Setting 4: triplet hold-out ########")
    forbidden: List[Triplet] = [(0, 9, 4), (1, 9, 4)]
    train4, test4 = generate_setting4_triplet_holdout(
        cfg,
        n_train=10,
        n_test=5,
        seed=10,
        forbidden_triplets=forbidden,
    )
    _debug_print_problems("Setting 4 - train (no forbidden triplets)", cfg, train4)
    _debug_print_problems("Setting 4 - test  (contains forbidden triplets)", cfg, test4)
    

if __name__ == "__main__":
    main()