"""
Live UK mortgage rates from the Bank of England Interactive Database (IADB).

No Streamlit imports, so it can be tested on its own. The BoE publishes
monthly "quoted" household mortgage rates (the average advertised rate,
excluding product fees) by fixed period and loan-to-value. They are updated
around the start of each month for the month before, so "live" here means
"latest official monthly figure", not today's best-buy deal.
"""

from __future__ import annotations

import io
import urllib.request
from dataclasses import dataclass
from datetime import date

import numpy as np
import pandas as pd

BOE_URL = (
    "https://www.bankofengland.co.uk/boeapps/database/_iadb-fromshowcolumns.asp"
    "?csv.x=yes&Datefrom={start}&Dateto=now&SeriesCodes={codes}"
    "&CSVF=TN&UsingCodes=Y&VPD=Y&VFD=N"
)
SOURCE_PAGE = "https://www.bankofengland.co.uk/boeapps/database/"


@dataclass(frozen=True)
class Series:
    code: str
    label: str
    fixed_years: int | None  # None for variable / revert-to rates
    ltv: int | None


SERIES = [
    Series("IUMBV34", "2-year fixed (75% LTV)", 2, 75),
    Series("IUMBV37", "3-year fixed (75% LTV)", 3, 75),
    Series("IUMBV42", "5-year fixed (75% LTV)", 5, 75),
    Series("IUMBV45", "10-year fixed (75% LTV)", 10, 75),
    Series("IUMZICQ", "2-year fixed (60% LTV)", 2, 60),
    Series("IUMZICR", "2-year fixed (85% LTV)", 2, 85),
    Series("IUMB482", "2-year fixed (90% LTV)", 2, 90),
    Series("IUMZO27", "5-year fixed (60% LTV)", 5, 60),
    Series("IUMZO28", "5-year fixed (90% LTV)", 5, 90),
    Series("IUMTLMV", "Standard variable / revert-to rate", None, None),
]
SERIES_BY_CODE = {s.code: s for s in SERIES}
FIXED_PERIODS = [2, 3, 5, 10]
# The headline series for each fixed period (75% LTV is the only LTV the BoE
# publishes for every period).
HEADLINE = {2: "IUMBV34", 3: "IUMBV37", 5: "IUMBV42", 10: "IUMBV45"}


def fetch_boe_rates(start: str = "01/Jan/2010", timeout: float = 15.0
                    ) -> pd.DataFrame:
    """Monthly rates (in %) indexed by month-end date, one column per code.

    Raises on network / parse errors; callers decide how to fall back.
    """
    codes = ",".join(s.code for s in SERIES)
    req = urllib.request.Request(
        BOE_URL.format(start=start, codes=codes),
        headers={"User-Agent": "Mozilla/5.0 (buy-vs-invest model)"},
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        text = resp.read().decode("utf-8", errors="replace")
    return parse_boe_csv(text)


def parse_boe_csv(text: str) -> pd.DataFrame:
    """Parse the IADB CSV (DATE + one column per series; '..' = missing)."""
    df = pd.read_csv(io.StringIO(text), na_values=["..", ""])
    if "DATE" not in df.columns:
        raise ValueError("Unexpected response from Bank of England database")
    df["DATE"] = pd.to_datetime(df["DATE"], format="%d %b %Y")
    df = df.set_index("DATE").sort_index().apply(pd.to_numeric, errors="coerce")
    return df.dropna(how="all")


def latest_table(df: pd.DataFrame) -> pd.DataFrame:
    """Latest value per series, with the change over the previous 12 months."""
    rows = []
    for s in SERIES:
        if s.code not in df:
            continue
        col = df[s.code].dropna()
        if col.empty:
            continue
        latest_date, latest = col.index[-1], col.iloc[-1]
        year_ago = col[col.index <= latest_date - pd.DateOffset(months=12)]
        rows.append({
            "Product": s.label,
            "Fixed (yrs)": s.fixed_years,
            "LTV (%)": s.ltv,
            "Rate (%)": latest,
            "12-month change (pp)": latest - year_ago.iloc[-1] if len(year_ago) else np.nan,
            "As of": latest_date.strftime("%b %Y"),
            "code": s.code,
        })
    return pd.DataFrame(rows)


def best_series(fixed_years: int, ltv_pct: float) -> str:
    """Pick the published series closest to a fixed period and LTV.

    Lenders price in LTV bands, so we use the cheapest published band the
    loan still fits under (e.g. 72% LTV -> the 75% band). Above the highest
    band available we use that highest band.
    """
    candidates = [s for s in SERIES if s.fixed_years == fixed_years]
    if not candidates:
        raise ValueError(f"No series for a {fixed_years}-year fix")
    candidates.sort(key=lambda s: s.ltv)
    for s in candidates:
        if ltv_pct <= s.ltv:
            return s.code
    return candidates[-1].code


def latest_rate(df: pd.DataFrame, code: str) -> tuple[float, date] | None:
    """(rate as a decimal, month) for a series, or None if not available."""
    if code not in df:
        return None
    col = df[code].dropna()
    if col.empty:
        return None
    return col.iloc[-1] / 100, col.index[-1].date()


def annual_change_std(df: pd.DataFrame, code: str, years: int = 15) -> float | None:
    """Std dev of 12-month changes in a rate (as a decimal), over `years`.

    Used as the default year-to-year volatility of mortgage rates in the
    Monte Carlo.
    """
    if code not in df:
        return None
    col = df[code].dropna()
    col = col[col.index >= col.index[-1] - pd.DateOffset(years=years)] if len(col) else col
    changes = (col - col.shift(12)).dropna()
    if len(changes) < 12:
        return None
    return float(changes.std() / 100)
