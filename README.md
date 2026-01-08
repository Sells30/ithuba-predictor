# Ithuba 5/36 Predictor (prototype) — with EV & Simulation

This project is a prototype tool to analyze Ithuba 5/36 draws (pick 5 from 1–36), produce recommended lines, compute expected value (EV) under configurable prize assumptions, and run a walk‑forward historical simulation.

Important: lotteries are designed to be random. This tool is for analysis/experimentation only — it cannot reliably predict winning numbers.

What's new
- Per-draw jackpot support: the input CSV may include a `jackpot` column (numeric, Rands).
- EV calculation: theoretical probabilities for matches (2–5) used to compute expected value per ticket.
- Walk-forward simulation (`--simulate`) that generates lines using only prior draws and computes realized payouts against historical draws.
- Parameters to model jackpot sharing: `--expected-jackpot-winners` and `--jackpot-mode` (see below).

Requirements
- Python 3.8+
- pip packages: pandas, numpy
- Optional for ML mode: scikit-learn

Install
```bash
python -m venv venv
source venv/bin/activate   # or venv\Scripts\activate on Windows
pip install -r requirements.txt
```

Usage examples
- Generate 5 recommended lines using the default weighted method:
```bash
python predict_5of36.py --input sample_draws.csv --method weighted --lines 5 --output recs.csv
```

- Compute theoretical EV using default prize table:
```bash
python predict_5of36.py --input sample_draws.csv --ev
```

- Run a walk-forward historical simulation using the weighted predictor, starting after 50 warm-up draws, and assume on average 2 jackpot winners per draw (so shared):
```bash
python predict_5of36.py --input sample_draws.csv --simulate --min-history 50 --lines 5 --expected-jackpot-winners 2.0 --jackpot-mode total --sim-output sim_results.csv
```

CLI flags (high level)
- `--input` / `-i`: CSV of past draws (required).
- `--method` / `-m`: `freq`, `weighted`, or `ml`. Default `weighted`.
- `--lines` / `-n`: number of recommended lines to output per run/draw.
- `--candidates` / `-c`: how many candidate lines to sample when selecting recommended lines.
- `--decay` / `-d`: decay factor for recency-weighted scores (default 0.95).
- `--pair-weight`: multiplier for pair co-occurrence bonus.
- `--window`: restrict scoring to last WINDOW draws.
- `--output` / `-o`: file to save one-off recommendations CSV.
- Prize / EV and simulation:
  - `--ev`: print theoretical EV given prize config and exit (no generation).
  - `--prize-match2`, `--prize-match3`, `--prize-match4`: specify prizes (numeric, Rands). Defaults provided below.
  - `--default-jackpot`: use this jackpot value (Rands) if CSV has no `jackpot` column (default placeholder R1,000,000).
  - `--ticket-cost`: cost per line (default R3).
  - `--simulate`: run walk-forward simulation over the CSV (generates sim summary and per-draw CSV if `--sim-output` specified).
  - `--min-history`: number of initial draws required before starting simulation (default 50).
  - `--expected-jackpot-winners`: float; used to estimate sharing when CSV `jackpot` column represents the total jackpot to be shared. Default 1.0 (no sharing).
  - `--jackpot-mode`: `total` (CSV `jackpot` is total payout for 5-match winners and will be divided by expected_jackpot_winners) or `per_winner` (CSV `jackpot` already represents per-winner payout). Default `total`.
  - `--sim-output`: path to save per-draw simulation results CSV.

Default prize table (used if not supplied)
- Match 5: use jackpot (per-draw or default)
- Match 4: R1,200
- Match 3: R100
- Match 2: R20
- Ticket cost: R3

Input CSV format
- The CSV accepts:
  - `numbers` column with a delimiter-separated list (e.g., `25 27 34 35 36` or `25,27,34,35,36`), OR
  - Five separate columns `n1..n5` (or similar).
- Optional columns:
  - `date` (YYYY-MM-DD or other parsable formats)
  - `draw_no` or `draw` (draw identifier)
  - `jackpot` (numeric; Rands). Accepts values like `307533.80` (no currency symbol). If input has R or commas, the loader tries to clean them.
- Example sample CSV `sample_draws.csv` included.

How the simulation works
- Walk-forward: for each historical draw at index t >= min_history:
  - Build a predictor using draws[0:t] only.
  - Generate `--lines` recommended lines for draw t (using the predictor and parameters).
  - Compare those lines to the actual draw t numbers and compute payouts:
    - For matches 2–4, use configured prizes.
    - For match 5, use the `jackpot` value from the CSV row. If `jackpot-mode` is `total`, we divide by `--expected-jackpot-winners` to estimate per-winner share.
  - Accumulate total spent and total payout and report ROI.

Notes & caveats
- The simulation models only payouts for the lines the tool generated (not the entire market). For modeling sharing precisely you would need the actual number of jackpot winners per draw; this tool lets you vary `--expected-jackpot-winners` to explore sharing scenarios.
- The simulation models only payouts for the lines the tool generated (not the entire market). For modeling sharing precisely you would need the actual number of jackpot winners per draw; this tool lets you vary `--expected-jackpot-winners` to explore sharing scenarios.
- ML mode is experimental and requires scikit-learn and more historical data to be meaningful.
- Always treat results as experimental; no guarantees.

If you'd like, next I can:
- Add an automated scraper to fetch official Ithuba history and jackpot numbers,
- Improve ML features or scoring,
- Add a Dockerfile or scheduler to run daily.
