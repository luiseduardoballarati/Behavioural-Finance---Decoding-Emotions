# J. Powell Correlations — Do the Fed Chair's Emotions Move Markets?

An exploratory study correlating Jerome Powell's **emotional expressions** during the FOMC press conference of **December 18, 2024** with **second-by-second market data** across 20+ stocks and ETFs.

Emotion signals (facial expression, voice prosody, and language) were extracted from the press conference video using the [Hume AI](https://hume.ai) multimodal API. Market data covers per-second aggregates (close price, volume, number of transactions) for tickers including NVDA, AAPL, TSLA, MSFT, AMZN, GOOG, PLTR, SMCI, DJT, and broad-market/sector ETFs (VOO, IVV, GLD, IEF, IYR, IYW, IYT, IYH, IYF, IYE, IYC, NDAQ, NIM).

## The Question

When Powell's voice shows more *Awkwardness*, or his face shows more *Amusement* — does the market react in real time? This notebook builds the dataset and runs the statistical groundwork to find out.

## What the Notebook Does

### 1. Dataset creation
- Parses raw market JSON (per-ticker arrays of `{t, v, c, n}` records) into a tidy pandas DataFrame with full timestamp decomposition
- Loads Hume AI emotion output (face, prosody, and language scores per second of the speech) and aligns it to a wall-clock timeline starting at 14:30 EST
- Trims all tickers to the exact window of the press conference

### 2. Time-series alignment
- Reindexes every ticker to a **strict 1-second frequency** and forward-fills gaps, so emotion and market series can be joined on a common `time` key
- Engineers features per ticker: close/volume/transaction **ratios** (t vs. t−1), total and average amount transacted, and % price delta

### 3. Correlation analysis
- **Pearson, Spearman, and Kendall** correlations between every emotion dimension and market variables (`close`, `volume`, `transactions`)
- A custom `correlation_with_counts()` function that reports the **number of valid pairwise observations** behind each coefficient — because a 0.9 correlation on 15 points is worth less than a 0.7 on 1,000
- A significance-filtered variant (`select_significant_correlations()`) that keeps only correlations with p < α and scores them by strength × confidence
- Correlation heatmaps and ranked lists of the most market-associated emotions

### 4. Baseline regression
- A simple `LinearRegression` predicting NVDA close price from ~163 features (emotion scores + engineered market features), with train/test split, R², and RMSE
- Feature-weight visualization showing which emotion dimensions carry positive vs. negative coefficients

### 5. Diagnostics & future work
- A `check_linearity()` utility comparing Pearson vs. Spearman, with normality (Shapiro) and heteroscedasticity (Breusch–Pagan) tests to decide whether linear methods are appropriate per feature
- Scatter diagnostics (e.g., `prosody.Awkwardness` vs. close price) and LOWESS smoothing

## Tech Stack

- **Python** — pandas, NumPy
- **Stats** — SciPy, statsmodels (Shapiro, Breusch–Pagan, LOWESS, OLS)
- **ML** — scikit-learn (LinearRegression, train/test split, polynomial features)
- **Visualization** — Plotly Express, seaborn, matplotlib
- **Emotion data** — Hume AI multimodal API (face, prosody, language)

## Running It

1. Clone the repo and install dependencies:
   ```bash
   pip install pandas numpy scipy statsmodels scikit-learn plotly seaborn matplotlib
   ```
2. Provide the two input files (paths are set at the top of the notebook):
   - Per-second market JSON (per-ticker `{t, v, c, n}` aggregates)
   - Hume AI batch output JSON for the press conference video
3. Run the notebook top to bottom:
   ```bash
   jupyter notebook jpowell_correlations.ipynb
   ```

> **Note:** The raw data files are not included in the repo. Market data can be reproduced from any per-second aggregates provider; emotion data requires a Hume AI API key.

## Caveats

This is exploratory research, not a trading signal. Correlation on a single event window says nothing about causation, forward-filled series inflate autocorrelation, and an in-window R² of ~0.91 largely reflects shared time trends rather than predictive power. The interesting output is the *ranking* of which emotional dimensions co-move with market activity — a starting point for event-study designs across multiple FOMC conferences.

## Context

This project is part of a broader exploration of **emotion analytics in high-stakes settings** at [MoodMetrics AI](https://moodmetrics.ai) — applying multimodal emotion AI to founder pitch evaluation, advertising effectiveness, and market signals.

---

*Author: Luis Eduardo Ballarati — [Medium](https://medium.com/@luiseduardoballarati)*
