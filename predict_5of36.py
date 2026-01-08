#!/usr/bin/env python3
"""
Ithuba 5/36 predictor (prototype) — extended with EV & historical simulation.

See README.md for usage and examples.
"""
import argparse
import csv
import math
import random
import re
from collections import Counter, defaultdict
from itertools import combinations
from typing import List, Tuple, Dict, Any

import numpy as np
import pandas as pd

# Optional ML imports
try:
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.multioutput import MultiOutputClassifier
    from sklearn.model_selection import train_test_split
    SKLEARN_AVAILABLE = True
except Exception:
    SKLEARN_AVAILABLE = False

NUM_MAX = 36
PICK = 5
TOTAL_COMBINATIONS = math.comb(NUM_MAX, PICK)
DEFAULT_PRIZES = {
    2: 20.0,
    3: 100.0,
    4: 1200.0,
}
DEFAULT_TICKET_COST = 3.0
DEFAULT_JACKPOT_PLACEHOLDER = 1_000_000.0


def parse_args():
    p = argparse.ArgumentParser(description="Ithuba 5/36 predictor (prototype) with EV & simulation")
    p.add_argument("--input", "-i", required=True, help="CSV of past draws")
    p.add_argument("--method", "-m", choices=("freq", "weighted", "ml"), default="weighted",
                   help="prediction method: freq, weighted, or ml (experimental)")
    p.add_argument("--lines", "-n", type=int, default=5, help="how many lines to output per run/draw")
    p.add_argument("--candidates", "-c", type=int, default=1000, help="candidates to sample")
    p.add_argument("--decay", "-d", type=float, default=0.95, help="decay factor for recency-weighted scores")
    p.add_argument("--pair-weight", type=float, default=0.5, help="pair co-occurrence bonus weight")
    p.add_argument("--window", type=int, default=None, help="consider only the last WINDOW draws")
    p.add_argument("--output", "-o", default="recommendations.csv", help="CSV to save recommended lines (one-off)")
    p.add_argument("--seed", type=int, default=42, help="random seed")

    # EV / prize args
    p.add_argument("--ev", action="store_true", help="print theoretical EV using prize config and exit")
    p.add_argument("--prize-match2", type=float, default=DEFAULT_PRIZES[2], help="prize for matching 2 numbers")
    p.add_argument("--prize-match3", type=float, default=DEFAULT_PRIZES[3], help="prize for matching 3 numbers")
    p.add_argument("--prize-match4", type=float, default=DEFAULT_PRIZES[4], help="prize for matching 4 numbers")
    p.add_argument("--default-jackpot", type=float, default=DEFAULT_JACKPOT_PLACEHOLDER,
                   help="default jackpot to use if CSV has no jackpot column (Rands)")
    p.add_argument("--ticket-cost", type=float, default=DEFAULT_TICKET_COST, help="cost per line (Rands)")

    # Simulation args
    p.add_argument("--simulate", action="store_true", help="run walk-forward historical simulation")
    p.add_argument("--min-history", type=int, default=50, help="minimum number of draws before starting simulation")
    p.add_argument("--expected-jackpot-winners", type=float, default=1.0,
                   help="expected number of jackpot winners per draw (used when jackpot column is 'total')")
    p.add_argument("--jackpot-mode", choices=("total", "per_winner"), default="total",
                   help="interpretation of jackpot column: 'total' means it's shared and will be divided by expected_winners")
    p.add_argument("--sim-output", default="simulation_results.csv", help="CSV path to save per-draw simulation results")

    return p.parse_args()


def clean_money(val: Any) -> float:
    """Remove currency symbols and thousands separators and return float (or NaN)."""
    if pd.isna(val):
        return float("nan")
    s = str(val).strip()
    s = s.replace("R", "").replace("r", "")
    s = s.replace(",", "")
    s = s.replace(" ", "")
    # allow parentheses or minus
    try:
        return float(s)
    except Exception:
        return float("nan")


def load_draws_with_meta(path: str) -> List[Dict[str, Any]]:
    """
    Load draw CSV and return list of dicts:
    { 'date': <str or NaT>, 'draw_no': <int or None>, 'numbers': [int,...], 'jackpot': float or NaN }
    Order: assumed oldest -> newest (chronological) when possible.
    """
    df = pd.read_csv(path)
    # normalize column names
    cols_lower = {c.lower(): c for c in df.columns}
    numbers_col = cols_lower.get("numbers")
    date_col = cols_lower.get("date")
    drawcol = cols_lower.get("draw_no") or cols_lower.get("draw") or cols_lower.get("drawno") or cols_lower.get("drawnumber")
    # attempt to find number columns like n1.. or num1..
    ncols = []
    for name in df.columns:
        ln = name.lower()
        if ln.startswith("n") and ln[1:].isdigit():
            ncols.append(name)
        elif ln.startswith("num") and ln[3:].isdigit():
            ncols.append(name)
    if len(ncols) < PICK:
        # fallback: first 5 columns that look numeric
        ncols = []
        for name in df.columns:
            try:
                if pd.to_numeric(df[name], errors="coerce").notna().any():
                    ncols.append(name)
            except Exception:
                continue
            if len(ncols) >= PICK:
                break

    # parse rows
    rows = []
    for _, row in df.iterrows():
        # parse numbers
        nums = None
        if numbers_col:
            raw = str(row[numbers_col])
            sep = "," if "," in raw else None
            parts = [p.strip() for p in (raw.split(sep) if sep else raw.split()) if p.strip()]
            try:
                cand = [int(p) for p in parts]
            except Exception:
                cand = None
            if cand and len(cand) == PICK:
                nums = sorted(cand)
        elif len(ncols) >= PICK:
            try:
                cand = [int(row[c]) for c in ncols[:PICK]]
                nums = sorted(cand)
            except Exception:
                nums = None
        if not nums or len(nums) != PICK:
            continue
        # parse jackpot if present
        jackpot = float("nan")
        if "jackpot" in cols_lower:
            jackpot = clean_money(row[cols_lower["jackpot"]])
        else:
            # maybe column named prize or jackpot_r or similar
            for alt in ("jackpot_r", "jackpot_amt", "prize", "prize_pool"):
                if alt in cols_lower:
                    jackpot = clean_money(row[cols_lower[alt]])
                    break
        # parse date and draw
        date_val = row[date_col] if date_col else None
        draw_no_val = row[drawcol] if drawcol else None
        try:
            draw_no_val = int(draw_no_val) if not pd.isna(draw_no_val) else None
        except Exception:
            draw_no_val = None
        rows.append({
            "date": date_val,
            "draw_no": draw_no_val,
            "numbers": nums,
            "jackpot": jackpot
        })

    # If date column exists and is parseable, sort by date
    if date_col:
        try:
            df_dates = pd.to_datetime(df[date_col], errors="coerce")
            if df_dates.notna().any():
                # build ordering map by original rows with parseable date
                rows_sorted = sorted(rows, key=lambda r: pd.to_datetime(r["date"], errors="coerce") if r["date"] is not None else pd.Timestamp.min)
                rows = rows_sorted
        except Exception:
            pass

    return rows


def extract_numbers_only(meta_rows: List[Dict[str, Any]]) -> List[List[int]]:
    return [r["numbers"] for r in meta_rows]


def compute_frequency_scores(draws: List[List[int]], window: int = None) -> np.ndarray:
    if window:
        draws = draws[-window:]
    c = Counter()
    for d in draws:
        c.update(d)
    arr = np.array([c.get(i, 0) for i in range(1, NUM_MAX + 1)], dtype=float)
    return arr


def compute_recency_weighted_scores(draws: List[List[int]], decay: float = 0.95, window: int = None) -> np.ndarray:
    if window:
        draws = draws[-window:]
    scores = np.zeros(NUM_MAX, dtype=float)
    for age, draw in enumerate(reversed(draws)):
        weight = (decay ** age)
        for n in draw:
            scores[n - 1] += weight
    return scores


def compute_pair_scores(draws: List[List[int]], window: int = None) -> defaultdict:
    if window:
        draws = draws[-window:]
    pair_counts = defaultdict(int)
    for d in draws:
        for a, b in combinations(sorted(d), 2):
            pair_counts[(a, b)] += 1
    return pair_counts


def line_pair_score(line: Tuple[int, ...], pair_counts: dict) -> float:
    s = 0.0
    for a, b in combinations(sorted(line), 2):
        s += pair_counts.get((a, b), 0)
    return s


def generate_candidate_lines(num_candidates: int, number_scores: np.ndarray, pair_counts: dict,
                             pair_weight: float = 0.5, rng=None) -> List[Tuple[Tuple[int, ...], float]]:
    rng = rng or random.Random()
    probs = number_scores.copy()
    if probs.sum() <= 0:
        probs = np.ones_like(probs)
    probs = probs / probs.sum()
    candidates = {}
    attempts = 0
    max_attempts = num_candidates * 5 + 1000
    while len(candidates) < num_candidates and attempts < max_attempts:
        chosen = np.random.choice(np.arange(1, NUM_MAX + 1), size=PICK, replace=False, p=probs)
        chosen_sorted = tuple(sorted(int(x) for x in chosen))
        if chosen_sorted in candidates:
            attempts += 1
            continue
        base_score = sum(number_scores[x - 1] for x in chosen_sorted)
        pscore = line_pair_score(chosen_sorted, pair_counts)
        combined = base_score + pair_weight * pscore
        candidates[chosen_sorted] = combined
        attempts += 1
    cand_list = sorted(candidates.items(), key=lambda kv: kv[1], reverse=True)
    return [(k, v) for k, v in cand_list]


def save_recommendations(recs: List[Tuple[Tuple[int, ...], float]], path: str, method: str):
    header = ["line_id", "n1", "n2", "n3", "n4", "n5", "score", "method", "notes"]
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        for idx, (line, score) in enumerate(recs, start=1):
            writer.writerow([idx, *line, score, method, "generated"])


def theoretical_match_probabilities() -> Dict[int, float]:
    """
    Returns exact probabilities of matching exactly k numbers when picking 5 vs 5 drawn from 36.
    P(k) = C(5,k) * C(31,5-k) / C(36,5)
    """
    probs = {}
    for k in range(0, PICK + 1):
        probs[k] = (math.comb(PICK, k) * math.comb(NUM_MAX - PICK, PICK - k)) / TOTAL_COMBINATIONS
    return probs


def compute_theoretical_ev(prize_match2: float, prize_match3: float, prize_match4: float, jackpot_value: float, ticket_cost: float) -> float:
    probs = theoretical_match_probabilities()
    ev = 0.0
    ev += probs.get(2, 0.0) * prize_match2
    ev += probs.get(3, 0.0) * prize_match3
    ev += probs.get(4, 0.0) * prize_match4
    ev += probs.get(5, 0.0) * jackpot_value
    ev -= ticket_cost
    return ev


def count_matches(line: Tuple[int, ...], draw: List[int]) -> int:
    return len(set(line).intersection(set(draw)))


def run_ml_predictor(draws: List[List[int]], lookback: int = 5, test_size: float = 0.2, random_state: int = 42):
    if not SKLEARN_AVAILABLE:
        raise RuntimeError("scikit-learn not available; cannot run ML method. Install scikit-learn to use ML mode.")
    if len(draws) < lookback + 10:
        raise RuntimeError(f"Not enough draws for ML training. Need at least {lookback + 10} draws; got {len(draws)}")
    X = []
    Y = []
    for i in range(lookback, len(draws) - 1):
        window = draws[i - lookback:i]
        feat = []
        for w in window:
            row = np.zeros(NUM_MAX, dtype=int)
            for n in w:
                row[n - 1] = 1
            feat.extend(row.tolist())
        X.append(feat)
        target_row = np.zeros(NUM_MAX, dtype=int)
        for n in draws[i + 1]:
            target_row[n - 1] = 1
        Y.append(target_row.tolist())
    X = np.array(X)
    Y = np.array(Y)
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=test_size, random_state=random_state)
    clf = MultiOutputClassifier(RandomForestClassifier(n_estimators=100, random_state=random_state), n_jobs=-1)
    clf.fit(X_train, Y_train)
    try:
        score = clf.score(X_test, Y_test)
        print(f"[ML] model score (accuracy-like) on held-out data: {score:.4f}")
    except Exception:
        pass
    recent_window = draws[-lookback:]
    feat = []
    for w in recent_window:
        row = np.zeros(NUM_MAX, dtype=int)
        for n in w:
            row[n - 1] = 1
        feat.extend(row.tolist())
    feat = np.array(feat).reshape(1, -1)
    proba = clf.predict_proba(feat)
    probs = np.zeros(NUM_MAX, dtype=float)
    for i, arr in enumerate(proba):
        try:
            if arr.shape[1] == 2:
                probs[i] = arr[0, 1]
            else:
                probs[i] = arr[0, -1]
        except Exception:
            probs[i] = 0.0
    return probs


def simulate_walkforward(meta_rows: List[Dict[str, Any]], args) -> None:
    """
    Walk-forward simulation:
    For t in range(min_history, len(meta_rows)):
      - train/score on draws[0:t]
      - generate lines for draw t
      - compare to actual draw t and compute payout using prize table and jackpot assumptions
    Saves per-draw simulation results to args.sim_output.
    """
    if len(meta_rows) < args.min_history + 1:
        print("Not enough draws to simulate with the requested min-history.")
        return

    total_spent = 0.0
    total_payout = 0.0
    sim_rows = []
    rng = random.Random(args.seed)
    for t in range(args.min_history, len(meta_rows)):
        history = [r["numbers"] for r in meta_rows[:t]]
        actual = meta_rows[t]["numbers"]
        jackpot_row = meta_rows[t].get("jackpot", float("nan"))
        # determine jackpot prize per-winner to use for this draw
        if not math.isnan(jackpot_row):
            if args.jackpot_mode == "total":
                per_winner_jackpot = float(jackpot_row) / max(1.0, float(args.expected_jackpot_winners))
            else:
                per_winner_jackpot = float(jackpot_row)
        else:
            per_winner_jackpot = args.default_jackpot

        # build number scores from history
        if args.method == "freq":
            number_scores = compute_frequency_scores(history, window=args.window)
        elif args.method == "weighted":
            number_scores = compute_recency_weighted_scores(history, decay=args.decay, window=args.window)
        elif args.method == "ml":
            try:
                number_scores = run_ml_predictor(history, lookback=5, random_state=args.seed)
            except Exception as e:
                print(f"[SIM] ML method failed at t={t}: {e}")
                return
        else:
            number_scores = compute_recency_weighted_scores(history, decay=args.decay, window=args.window)

        pair_counts = compute_pair_scores(history, window=args.window)
        candidates = generate_candidate_lines(args.candidates, number_scores, pair_counts, pair_weight=args.pair_weight, rng=rng)
        # pick top lines
        recommended = [line for line, _ in candidates[: args.lines]]
        # compute payout for these lines
        payout = 0.0
        matches_record = []
        for line in recommended:
            m = count_matches(line, actual)
            matches_record.append(m)
            if m == 5:
                payout += per_winner_jackpot
            elif m == 4:
                payout += args.prize_match4
            elif m == 3:
                payout += args.prize_match3
            elif m == 2:
                payout += args.prize_match2
            # 0 or 1 -> no payout
        spent = args.lines * args.ticket_cost
        net = payout - spent
        total_spent += spent
        total_payout += payout
        sim_rows.append({
            "index": t,
            "date": meta_rows[t].get("date"),
            "draw_no": meta_rows[t].get("draw_no"),
            "actual_numbers": " ".join(str(x) for x in actual),
            "recommended_lines": " | ".join(" ".join(str(n) for n in line) for line in recommended),
            "matches": " | ".join(str(m) for m in matches_record),
            "payout": round(payout, 2),
            "spent": round(spent, 2),
            "net": round(net, 2),
            "jackpot_used_per_winner": round(per_winner_jackpot, 2)
        })
    roi = (total_payout - total_spent) / total_spent if total_spent > 0 else float("nan")
    print("Simulation summary:")
    print(f"  Draws simulated: {len(sim_rows)}")
    print(f"  Total spent: R{total_spent:,.2f}")
    print(f"  Total payout: R{total_payout:,.2f}")
    print(f"  Net: R{(total_payout - total_spent):,.2f}")
    print(f"  ROI: {roi * 100:.2f}%")
    # save per-draw CSV
    if args.sim_output:
        keys = ["index", "date", "draw_no", "actual_numbers", "recommended_lines", "matches", "payout", "spent", "net", "jackpot_used_per_winner"]
        with open(args.sim_output, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=keys)
            writer.writeheader()
            for r in sim_rows:
                writer.writerow(r)
        print(f"Saved simulation details to {args.sim_output}")


def main():
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)

    meta_rows = load_draws_with_meta(args.input)
    if not meta_rows:
        print("No valid draws found in input.")
        return
    draws = extract_numbers_only(meta_rows)
    # If user requested EV calculation only
    # Determine jackpot to use for theoretical EV: prefer default-jackpot unless CSV has 'jackpot'
    # For theoretical EV print we use args.default_jackpot by default (user can override).
    if args.ev:
        jackpot_for_ev = args.default_jackpot
        # if the CSV has at least one jackpot entry, use average of non-NaN jackpots as default for EV
        jackpots = [r["jackpot"] for r in meta_rows if not math.isnan(r["jackpot"])]
        if jackpots:
            jackpot_for_ev = float(sum(jackpots) / len(jackpots))
        ev = compute_theoretical_ev(args.prize_match2, args.prize_match3, args.prize_match4, jackpot_for_ev, args.ticket_cost)
        probs = theoretical_match_probabilities()
        print("Theoretical match probabilities (exact):")
        for k in sorted(probs.keys()):
            print(f"  match {k}: {probs[k]:.8f}")
        print(f"\nUsing jackpot={jackpot_for_ev:,.2f} (R), prizes: 2->{args.prize_match2}, 3->{args.prize_match3}, 4->{args.prize_match4}, ticket_cost={args.ticket_cost}")
        print(f"Theoretical EV per ticket: R{ev:.4f}")
        return

    # If simulate: run walk-forward simulation
    if args.simulate:
        simulate_walkforward(meta_rows, args)
        return

    # Otherwise produce one-off recommendations using entire dataset (or last WINDOW draws if provided)
    if args.window:
        draws_to_use = draws[-args.window:]
    else:
        draws_to_use = draws

    if args.method == "freq":
        number_scores = compute_frequency_scores(draws_to_use)
    elif args.method == "weighted":
        number_scores = compute_recency_weighted_scores(draws_to_use, decay=args.decay)
    elif args.method == "ml":
        try:
            number_scores = run_ml_predictor(draws_to_use, lookback=5, random_state=args.seed)
        except Exception as e:
            print("ML method failed:", e)
            return
    else:
        number_scores = compute_recency_weighted_scores(draws_to_use, decay=args.decay)

    pair_counts = compute_pair_scores(draws_to_use)
    cand_list = generate_candidate_lines(args.candidates, number_scores, pair_counts,
                                         pair_weight=args.pair_weight, rng=random)

    recommendations = cand_list[: args.lines]
    print(f"Top {len(recommendations)} recommended lines (method={args.method}):")
    for idx, (line, score) in enumerate(recommendations, start=1):
        print(f"{idx:2d}: {line}  score={score:.4f}")

    save_recommendations(recommendations, args.output, method=args.method)
    print(f"Saved recommendations to {args.output}")


if __name__ == "__main__":
    main()
