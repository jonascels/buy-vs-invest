"""
Historical test: is a variable mortgage cheaper than rolling 2-year fixes?

Replays UK history using Bank of England quoted rates (75% LTV, excl. fees) on
a repayment mortgage. All strategies face the same months, so comparisons are
paired. No Streamlit imports.

Strategies
  fix2  Rolling 2-yr fix: rate locked at the quoted fix, re-quoted every 24 months
  var2  Rolling 2-yr variable: margin over Bank Rate locked for 24 months, re-quoted
  life  Lifetime tracker: margin over Bank Rate locked for the whole period
  svr   Pays the lender's revert-to rate each month

Methods
  continuous  every possible start month, running through real history
  segments    random historical segments stitched together (block bootstrap)
"""

from __future__ import annotations

import numpy as np
import pandas as pd

import rates

CODES = {"IUMBV34": "fix2", "IUMBV48": "var2", "IUMBV24": "life",
         "IUMTLMV": "svr", "IUMABEDR": "bank"}
DEAL_MONTHS = 24
VARIABLE_TYPES = ["var2", "life", "svr"]
NAMES = {"fix2": "Rolling 2-yr fix", "var2": "Rolling 2-yr variable",
         "life": "Lifetime tracker", "svr": "Standard variable rate"}


def fetch_history() -> pd.DataFrame:
    return prepare(rates.fetch_series(list(CODES), start="01/Jan/1995"))


def prepare(raw: pd.DataFrame) -> pd.DataFrame:
    """Decimal rates and margins from raw BoE percentages, from when var2 starts."""
    df = raw.rename(columns=CODES)[list(CODES.values())] / 100
    for product in ("var2", "life"):  # zeros in these series are placeholders
        df[product] = df[product].where(df[product] > 0)
    df["var2_margin"] = df["var2"] - df["bank"]
    # Margins are only read at the start of a deal; carry over the odd gap
    df["life_margin"] = (df["life"] - df["bank"]).ffill()
    return df.dropna(subset=["var2", "fix2", "svr", "bank"])


def continuous_windows(n_months: int, years: int) -> np.ndarray:
    months = years * 12
    return np.array([np.arange(s, s + months) for s in range(n_months - months + 1)])


def segment_windows(n_months: int, years: int, block: int, n: int,
                    seed: int = 42) -> np.ndarray:
    rng = np.random.default_rng(seed)
    months = years * 12
    blocks = int(np.ceil(months / block))
    starts = rng.integers(0, n_months - block + 1, (n, blocks))
    return (starts[:, :, None] + np.arange(block)).reshape(n, -1)[:, :months]


def rate_paths(d: pd.DataFrame, idx: np.ndarray) -> dict:
    """Monthly rate paid by each strategy along month indices idx (paths, months)."""
    bank = d["bank"].values
    deal_start = (np.arange(idx.shape[1]) // DEAL_MONTHS) * DEAL_MONTHS
    start_idx = idx[:, deal_start]
    return {
        "fix2": d["fix2"].values[start_idx],
        "var2": bank[idx] + d["var2_margin"].values[start_idx],
        "life": bank[idx] + d["life_margin"].values[idx[:, 0]][:, None],
        "svr": d["svr"].values[idx],
    }


def mortgage_cost(rate_matrix: np.ndarray, loan: float, term_months: int) -> dict:
    """Interest, average rate and payment risk; payment re-sets as the rate moves."""
    n, months = rate_matrix.shape
    r = np.maximum(rate_matrix, 1e-4) / 12
    bal = np.full(n, float(loan))
    interest = np.zeros(n)
    bal_sum = np.zeros(n)
    payments = np.empty((n, months))
    for m in range(months):
        left = term_months - m
        pmt = bal * r[:, m] * (1 + r[:, m]) ** left / ((1 + r[:, m]) ** left - 1)
        i = bal * r[:, m]
        interest += i
        bal_sum += bal
        payments[:, m] = pmt
        bal -= pmt - i
    jump = (payments[:, 12:] - payments[:, :-12]).max(axis=1) if months > 12 else np.zeros(n)
    return {"interest": interest, "avg_rate": interest * 12 / bal_sum,
            "peak": payments.max(axis=1), "jump": jump}


def run_test(d: pd.DataFrame, years: int, method: str = "continuous", block: int = 24,
             n_paths: int = 5_000, loan: float = 300_000, term_years: int = 25,
             seed: int = 42) -> dict:
    """Costs per strategy plus savings of each variable type against the fix."""
    if method == "continuous":
        idx = continuous_windows(len(d), years)
    else:
        idx = segment_windows(len(d), years, block, n_paths, seed)
    costs = {k: mortgage_cost(v, loan, term_years * 12)
             for k, v in rate_paths(d, idx).items()}
    return {"costs": costs,
            "savings": {k: costs["fix2"]["interest"] - costs[k]["interest"]
                        for k in VARIABLE_TYPES},
            "starts": d.index[idx[:, 0]]}


def summary(result: dict) -> pd.DataFrame:
    fix = result["costs"]["fix2"]
    rows = []
    for k in VARIABLE_TYPES:
        s, c = result["savings"][k], result["costs"][k]
        rows.append({
            "key": k, "Variable type": NAMES[k],
            "Variable cheaper (%)": 100 * np.mean(s > 0),
            "Median saving": np.median(s),
            "10th %ile": np.percentile(s, 10), "90th %ile": np.percentile(s, 90),
            "Rate gap (pp)": 100 * np.median(fix["avg_rate"] - c["avg_rate"]),
            "Biggest 12-month rise": np.median(c["jump"]),
            "Fix: biggest 12-month rise": np.median(fix["jump"]),
        })
    return pd.DataFrame(rows)


def single_deals(d: pd.DataFrame) -> pd.DataFrame:
    """Each 2-year deal: variable's rate advantage and what drove it, in pp."""
    idx = continuous_windows(len(d), DEAL_MONTHS // 12)
    p = rate_paths(d, idx)
    gap = p["fix2"].mean(axis=1) - p["var2"].mean(axis=1)
    premium = (d["fix2"] - d["var2"]).values[idx[:, 0]]
    move = d["bank"].values[idx].mean(axis=1) - d["bank"].values[idx[:, 0]]
    bank = d["bank"].values
    slope = (bank[-1] - bank[0]) / (len(bank) - 1)
    return pd.DataFrame({
        "gap": 100 * gap, "premium": 100 * premium, "bank_move": 100 * move,
        # Average month of a 24-month deal is 11.5 months in
        "gap_no_drift": 100 * (gap + slope * 11.5),
    }, index=d.index[idx[:, 0]])
