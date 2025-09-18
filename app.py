import os, uuid
from pathlib import Path
from flask import Flask, render_template, request, redirect, url_for, flash, send_file
import pandas as pd
import numpy as np

import matplotlib
matplotlib.use("Agg")  # headless backend for saving plots
import matplotlib.pyplot as plt

app = Flask(__name__)
app.config["SECRET_KEY"] = os.getenv("FLASK_SECRET_KEY", "dev-key")
app.config["MAX_CONTENT_LENGTH"] = 8 * 1024 * 1024  # 8MB max upload
app.config["OUTPUT_FOLDER"] = "outputs"
Path(app.config["OUTPUT_FOLDER"]).mkdir(exist_ok=True)

ALLOWED_CSV = {"csv"}


def allowed_csv(filename: str) -> bool:
	"""Check filename has a .csv extension."""
	return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_CSV


@app.get("/")
def index():
	return render_template("index.html")


@app.post("/analyze-csv")
def analyze_csv():
    f = request.files.get("file")
    if not f or not allowed_csv(f.filename):
        flash("Please upload a .csv file.")
        return redirect(url_for("index"))

    # Read CSV into a DataFrame
    df = pd.read_csv(f)

    # Parse dates if present (example: joined_date)
    if "joined_date" in df.columns:
        df["joined_date"] = pd.to_datetime(df["joined_date"], errors="coerce")

    # --- Separate summaries ---
    num_summary = df.describe(include=[np.number]).round(3)
    cat_summary = df.describe(exclude=[np.number])

    num_html = None
    cat_html = None
    if not num_summary.empty:
        num_html = num_summary.to_html(classes="dataframe grid", border=0)
    if not cat_summary.empty:
        cat_html = cat_summary.to_html(classes="dataframe grid", border=0)

    # Missing values table
    missing_html = (
        df.isna().sum().to_frame("missing")
        .to_html(classes="dataframe grid", border=0)
    )

    # --- Histogram: make it specifically for AGE ---
    hist_png = None
    if "age" in df.columns:
        # Coerce to numeric, drop NaNs
        age = pd.to_numeric(df["age"], errors="coerce").dropna()
        if not age.empty:
            plt.figure()
            # Bin in 5-year groups
            try:
                lo = int(age.min())
                hi = int(age.max())
                bin_start = (lo // 5) * 5
                bin_end = ((hi + 4) // 5) * 5 + 5
                bins = list(range(bin_start, bin_end, 5))
            except Exception:
                bins = 30  # fallback if something odd happens

            plt.hist(age, bins=bins)
            plt.xlabel("Age")
            plt.ylabel("Count")
            plt.title("Age distribution")
            out_path = Path(app.config["OUTPUT_FOLDER"]) / f"{uuid.uuid4()}_age_hist.png"
            plt.savefig(out_path, bbox_inches="tight")
            plt.close()
            hist_png = out_path.name

    return render_template(
        "csv_result.html",
        num_html=num_html,
        cat_html=cat_html,
        missing_html=missing_html,
        hist_png=hist_png,
    )



@app.get("/outputs/<name>")
def get_output(name):
	return send_file(Path(app.config["OUTPUT_FOLDER"]) / name, mimetype="image/png")


if __name__ == "__main__":
	app.run(debug=True)
