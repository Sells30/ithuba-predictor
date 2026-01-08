#!/usr/bin/env python3
"""
Simple Flask web UI for the Ithuba predictor with Scraper and Predict-Next features.

Run:
    python web_app.py

Open http://127.0.0.1:5000/
"""
import os
import csv
import uuid
from glob import glob
from flask import Flask, request, render_template, redirect, url_for, send_from_directory, flash
from werkzeug.utils import secure_filename

# Import predictor functions from existing module
import predict_5of36 as predictor

# Import scraper functions (site-specific)
from scrape_lottery_coza import crawl_year_pages, merge_and_write

UPLOAD_FOLDER = "uploads"
OUTPUT_FOLDER = "outputs"
ALLOWED_EXTENSIONS = {"csv"}

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

app = Flask(__name__)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
app.config["OUTPUT_FOLDER"] = OUTPUT_FOLDER
app.secret_key = os.environ.get("FLASK_SECRET", str(uuid.uuid4()))

DEFAULT_YEAR_URLS = [
    "https://www.lottery.co.za/daily-lotto/results/2019",
    "https://www.lottery.co.za/daily-lotto/results/2020",
    "https://www.lottery.co.za/daily-lotto/results/2021",
    "https://www.lottery.co.za/daily-lotto/results/2022",
    "https://www.lottery.co.za/daily-lotto/results/2023",
    "https://www.lottery.co.za/daily-lotto/results/2024",
    "https://www.lottery.co.za/daily-lotto/results/2025",
]

def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS

def find_latest_dataset():
    """
    Return path to the latest CSV in OUTPUT_FOLDER (scraped results), or sample_draws.csv if none.
    """
    patterns = [
        os.path.join(app.config["OUTPUT_FOLDER"], "scraped_*.csv"),
        os.path.join(app.config["OUTPUT_FOLDER"], "lottery_coza_draws*.csv"),
        os.path.join(app.config["OUTPUT_FOLDER"], "*.csv"),
    ]
    candidates = []
    for p in patterns:
        candidates.extend(glob(p))
    if candidates:
        # pick newest by mtime
        candidates = sorted(candidates, key=lambda p: os.path.getmtime(p), reverse=True)
        return candidates[0]
    # fallback to repo sample
    if os.path.exists("sample_draws.csv"):
        return "sample_draws.csv"
    return None

@app.route("/", methods=["GET"])
def index():
    defaults = {
        "method": "weighted",
        "lines": 5,
        "candidates": 1000,
        "decay": 0.95,
        "pair_weight": 0.5,
        "window": "",
    }
    return render_template("index.html", defaults=defaults)

# --- existing generate/download/scrape routes unchanged (omitted here for brevity in this snippet) ---
# To keep the file concise here, assume the previous implementation of /generate, /download, /scrape, /run_scrape exists unchanged.
# (When saving this file replace the placeholder below with the generate/download/scrape/run_scrape implementations from your current web_app.py.)
#
# For clarity, the new routes /predict_next are added below.

@app.route("/predict_next", methods=["GET"])
def predict_next_form():
    # show form letting user pick method and parameters; source dataset will be auto-detected
    dataset = find_latest_dataset()
    return render_template("predict.html", dataset=dataset, defaults={
        "method": "weighted",
        "lines": 5,
        "candidates": 1000,
        "decay": 0.95,
        "pair_weight": 0.5,
        "window": ""
    })

@app.route("/predict_next", methods=["POST"])
def predict_next_run():
    # allow optional override upload
    file = request.files.get("file")
    if file and file.filename != "" and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        save_path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
        file.save(save_path)
        dataset_path = save_path
    else:
        dataset_path = find_latest_dataset()
    if not dataset_path:
        flash("No dataset found (no outputs and no sample_draws.csv). Please upload or run the scraper first.", "danger")
        return redirect(url_for("predict_next_form"))

    method = request.form.get("method", "weighted")
    try:
        lines = int(request.form.get("lines", 5))
    except Exception:
        lines = 5
    try:
        candidates = int(request.form.get("candidates", 1000))
    except Exception:
        candidates = 1000
    try:
        decay = float(request.form.get("decay", 0.95))
    except Exception:
        decay = 0.95
    try:
        pair_weight = float(request.form.get("pair_weight", 0.5))
    except Exception:
        pair_weight = 0.5
    window_raw = request.form.get("window", "")
    window = int(window_raw) if window_raw.strip().isdigit() else None

    # load dataset
    try:
        meta_rows = predictor.load_draws_with_meta(dataset_path)
        if not meta_rows:
            flash("No valid draws found in dataset.", "danger")
            return redirect(url_for("predict_next_form"))
        draws = predictor.extract_numbers_only(meta_rows)
    except Exception as e:
        flash(f"Failed to load dataset: {e}", "danger")
        return redirect(url_for("predict_next_form"))

    # determine draws to use
    draws_to_use = draws[-window:] if window else draws

    # compute per-number scores
    try:
        if method == "ml":
            number_scores = predictor.run_ml_predictor(draws_to_use, lookback=5, random_state=42)
            # ML returns probabilities 0..1; we may normalize later for display
            per_number = number_scores.copy()
        elif method == "freq":
            number_scores = predictor.compute_frequency_scores(draws_to_use, window=window)
            per_number = number_scores / (number_scores.sum() if number_scores.sum() > 0 else 1.0)
        else:  # weighted
            number_scores = predictor.compute_recency_weighted_scores(draws_to_use, decay=decay, window=window)
            per_number = number_scores / (number_scores.sum() if number_scores.sum() > 0 else 1.0)
    except Exception as e:
        flash(f"Failed to compute number scores: {e}", "danger")
        return redirect(url_for("predict_next_form"))

    # compute pair counts and generate candidate lines
    pair_counts = predictor.compute_pair_scores(draws_to_use, window=window)
    try:
        candidates_list = predictor.generate_candidate_lines(candidates, number_scores, pair_counts, pair_weight=pair_weight)
    except Exception as e:
        flash(f"Failed to generate candidate lines: {e}", "danger")
        return redirect(url_for("predict_next_form"))

    recommendations = candidates_list[:lines]

    # save recommendations CSV
    out_filename = f"pred_next_{uuid.uuid4().hex[:8]}.csv"
    out_path = os.path.join(app.config["OUTPUT_FOLDER"], out_filename)
    predictor.save_recommendations(recommendations, out_path, method=method)

    # prepare per-number display rows
    per_number_rows = []
    for i in range(1, predictor.NUM_MAX + 1):
        per_number_rows.append({
            "number": i,
            "score": float(number_scores[i-1]) if hasattr(number_scores, "__len__") else float(per_number[i-1]),
            "prob": float(per_number[i-1]) if len(per_number) == predictor.NUM_MAX else None
        })
    # sort numbers descending by score/prob for display
    per_number_rows_sorted = sorted(per_number_rows, key=lambda r: r["score"], reverse=True)

    recs_for_display = []
    for idx, (line, score) in enumerate(recommendations, start=1):
        recs_for_display.append({
            "line_id": idx,
            "n1": line[0],
            "n2": line[1],
            "n3": line[2],
            "n4": line[3],
            "n5": line[4],
            "score": score
        })

    return render_template("predict.html",
                           dataset=dataset_path,
                           per_number=per_number_rows_sorted,
                           recommendations=recs_for_display,
                           out_file=out_filename,
                           method=method)
