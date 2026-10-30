"""
Streamlit UI for the Mortgage lab tab: find the best deposit, compare repayment
with interest-only, and plot custom scenarios against each other.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

import rates as boe
from model import (INTEREST_ONLY, REPAYMENT, Scenario, ScenarioResult,
                   StochasticAssumptions, compare_scenarios, run_scenarios,
                   upfront_cash_needed)

# Fixed order, so a scenario keeps its colour as long as it keeps its row
SERIES_COLOURS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100",
                  "#e87ba4", "#008300", "#6250d6", "#e34948"]
MAX_SCENARIOS = len(SERIES_COLOURS)
TYPE_LABELS = {REPAYMENT: "Repayment", INTEREST_ONLY: "Interest-only"}
TYPE_BY_LABEL = {v: k for k, v in TYPE_LABELS.items()}
CHART_LAYOUT = dict(template="plotly_white", hovermode="x unified",
                    margin=dict(l=10, r=10, t=10, b=10),
                    legend=dict(orientation="h", y=1.12),
                    font=dict(size=12, color="#333333"),
                    paper_bgcolor="white", plot_bgcolor="#fafafa")


def gbp(x: float) -> str:
    return f"-£{-x:,.0f}" if x < 0 else f"£{x:,.0f}"


def rgba(hex_colour: str, alpha: float) -> str:
    r, g, b = (int(hex_colour[i:i + 2], 16) for i in (1, 3, 5))
    return f"rgba({r}, {g}, {b}, {alpha})"


def add_band(fig: go.Figure, x, median, low, high, name: str, colour: str,
             hover: str = "£%{y:,.0f}") -> None:
    """Median line with a filled 10th-90th percentile band."""
    fig.add_trace(go.Scatter(x=x, y=high, mode="lines", line=dict(width=0),
                             legendgroup=name, showlegend=False, hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=x, y=low, mode="lines", line=dict(width=0),
                             fill="tonexty", fillcolor=rgba(colour, 0.14),
                             legendgroup=name, showlegend=False, hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=x, y=median, mode="lines", name=name, legendgroup=name,
                             line=dict(color=colour, width=2),
                             hovertemplate=f"{name}: {hover}<extra></extra>"))


def wealth_matrix(r: ScenarioResult, real: bool) -> np.ndarray:
    return r.net_worth / r.inflation_index if real else r.net_worth


def rate_for(deposit: float, fixed_years: int, repayment_type: str, premium: float,
             live_df, use_live: bool, manual_rate: float) -> float:
    """Start rate for a deposit: the live LTV band if on, else the sidebar rate."""
    rate = manual_rate
    if use_live and live_df is not None:
        found = boe.latest_rate(live_df, boe.best_series(fixed_years, (1 - deposit) * 100))
        if found:
            rate = found[0]
    return rate + premium if repayment_type == INTEREST_ONLY else rate


def results_table(df: pd.DataFrame) -> pd.DataFrame:
    show = df.copy()
    for c in ("Cash upfront", "First payment", "Median net worth", "10th %ile",
              "90th %ile", "Median interest paid"):
        show[c] = show[c].map(gbp)
    show["Deposit (%)"] = show["Deposit (%)"].map("{:.0f}".format)
    show["Start rate (%)"] = show["Start rate (%)"].map("{:.2f}".format)
    show["Chance best (%)"] = show["Chance best (%)"].map("{:.0f}".format)
    show["Ran out of cash (%)"] = show["Ran out of cash (%)"].map("{:.0f}".format)
    return show


def warn_if_cash_ran_out(df: pd.DataFrame, horizon: int) -> None:
    short = df[df["Ran out of cash (%)"] > 0]
    if len(short):
        worst = short.loc[short["Ran out of cash (%)"].idxmax()]
        st.warning(
            f"⚠️ {len(short)} of {len(df)} options ran out of money within {horizon} "
            f"years on some paths (worst: {worst['Scenario']}, "
            f"{worst['Ran out of cash (%)']:.0f}% of paths). The shortfall is treated as "
            "extra debt at the mortgage rate. A real lender may not allow that, so "
            "treat these results with care.")


# ------------------------------------------------------------------ best deposit
def deposit_sweep(a: StochasticAssumptions, cash: float, budget: float, premium: float,
                  n_sims: int, horizon: int, fixed_years: int, term: int, real: bool,
                  live_df, use_live: bool) -> None:
    c1, c2 = st.columns([2, 1])
    low, high = c1.slider("Deposits to test (%)", 5, 100, (5, 60), 5)
    types = c2.multiselect("Mortgage types", list(TYPE_BY_LABEL),
                           default=list(TYPE_BY_LABEL))
    if not types:
        st.info("Pick at least one mortgage type.")
        return

    affordable = [d for d in range(low, high + 1, 5)
                  if upfront_cash_needed(a, d / 100) <= cash]
    if not affordable:
        st.warning(f"Your cash doesn't cover any of these deposits. Stamp duty and fees "
                   f"alone are {gbp(upfront_cash_needed(a, 0.0))}.")
        return
    if affordable[-1] < high:
        st.caption(f"With {gbp(cash)} available, the largest deposit tested is "
                   f"{affordable[-1]}%, after stamp duty and fees.")

    scenarios = [
        Scenario(f"{d}% {label}", d / 100,
                 rate_for(d / 100, fixed_years, TYPE_BY_LABEL[label], premium,
                          live_df, use_live, a.mortgage_rate),
                 TYPE_BY_LABEL[label], fixed_years, term)
        for label in types for d in affordable
    ]
    results = run_scenarios(a, scenarios, cash, budget, n_sims)
    table = compare_scenarios(results, horizon, real)

    top = table.loc[table["Median net worth"].idxmax()]
    safest = table.loc[table["10th %ile"].idxmax()]
    spread = table["Median net worth"].max() - table["Median net worth"].min()
    m1, m2, m3 = st.columns(3)
    m1.metric("Highest typical outcome", top["Scenario"], gbp(top["Median net worth"]),
              delta_color="off", help="Highest median net worth at this horizon.")
    m2.metric("Best in a bad scenario", safest["Scenario"], gbp(safest["10th %ile"]),
              delta_color="off",
              help="Highest 10th percentile: only 1 path in 10 ends up lower.")
    m3.metric("Median gap, best to worst", gbp(spread),
              help="If this is small next to the range on the chart, the deposit "
                   "choice matters less than how markets turn out.")
    warn_if_cash_ran_out(table, horizon)

    st.subheader(f"Net worth after {horizon} years, by deposit")
    fig = go.Figure()
    for i, label in enumerate(types):
        rows = table[table["Type"] == label]
        add_band(fig, rows["Deposit (%)"], rows["Median net worth"], rows["10th %ile"],
                 rows["90th %ile"], label, SERIES_COLOURS[i])
    fig.update_layout(height=420, xaxis_title="Deposit (%)",
                      yaxis_title="Net worth (£)", **CHART_LAYOUT)
    st.plotly_chart(fig, use_container_width=True)
    st.caption("Line = median. Shaded = 80% of simulated paths. Small jumps come from "
               "the cheaper rate when a loan drops into a lower LTV band.")

    # Paths share shocks, so a path-by-path difference strips out market noise.
    base = results[0]
    base_w = wealth_matrix(base, real)[:, horizon - 1]
    st.subheader(f"Gain or loss compared with {base.scenario.name}")
    fig = go.Figure()
    for i, label in enumerate(types):
        rows = [r for r in results if TYPE_LABELS[r.scenario.repayment_type] == label]
        diffs = np.array([wealth_matrix(r, real)[:, horizon - 1] - base_w for r in rows])
        add_band(fig, [r.scenario.deposit_pct * 100 for r in rows],
                 np.percentile(diffs, 50, axis=1), np.percentile(diffs, 10, axis=1),
                 np.percentile(diffs, 90, axis=1), label, SERIES_COLOURS[i])
    fig.add_hline(y=0, line_color="#999999", line_dash="dot")
    fig.update_layout(height=340, xaxis_title="Deposit (%)",
                      yaxis_title="Difference (£)", **CHART_LAYOUT)
    st.plotly_chart(fig, use_container_width=True)
    st.caption("Every option faces the same simulated markets on each path, so this "
               "shows the effect of the financing choice alone.")

    left, right = st.columns(2)
    for col, metric, title in ((left, "First payment", "First monthly payment"),
                               (right, "Median interest paid",
                                f"Interest paid over {horizon} years (median)")):
        with col:
            st.subheader(title)
            fig = go.Figure()
            for i, label in enumerate(types):
                rows = table[table["Type"] == label]
                fig.add_trace(go.Scatter(
                    x=rows["Deposit (%)"], y=rows[metric], name=label, mode="lines+markers",
                    line=dict(color=SERIES_COLOURS[i], width=2), marker=dict(size=8),
                    hovertemplate=f"{label}: £%{{y:,.0f}}<extra></extra>"))
            fig.update_layout(height=280, xaxis_title="Deposit (%)",
                              yaxis_title="£", **CHART_LAYOUT)
            st.plotly_chart(fig, use_container_width=True)

    st.subheader("All results")
    st.dataframe(results_table(table), use_container_width=True, hide_index=True)
    st.download_button("⬇️ Download results (CSV)", table.to_csv(index=False).encode(),
                       file_name="deposit_sweep.csv", mime="text/csv")


# ------------------------------------------------------------------ custom scenarios
DEFAULT_SCENARIOS = pd.DataFrame([
    {"Name": "20% repayment", "Deposit (%)": 20, "Type": "Repayment",
     "Fixed (yrs)": 5, "Term (yrs)": 30, "Own rate (%)": None},
    {"Name": "40% repayment", "Deposit (%)": 40, "Type": "Repayment",
     "Fixed (yrs)": 5, "Term (yrs)": 30, "Own rate (%)": None},
    {"Name": "20% interest-only", "Deposit (%)": 20, "Type": "Interest-only",
     "Fixed (yrs)": 5, "Term (yrs)": 30, "Own rate (%)": None},
    {"Name": "40% interest-only", "Deposit (%)": 40, "Type": "Interest-only",
     "Fixed (yrs)": 5, "Term (yrs)": 30, "Own rate (%)": None},
]).astype({"Own rate (%)": "float"})


def build_scenarios(edited: pd.DataFrame, a: StochasticAssumptions, cash: float,
                    premium: float, live_df, use_live: bool):
    """Turn editor rows into Scenarios; returns (scenarios, problems)."""
    scenarios, problems, names = [], [], set()
    for i, row in edited.iterrows():
        if pd.isna(row["Deposit (%)"]):
            continue
        deposit = float(row["Deposit (%)"]) / 100
        label = row["Type"] if row["Type"] in TYPE_BY_LABEL else "Repayment"
        typ = TYPE_BY_LABEL[label]
        fixed = int(row["Fixed (yrs)"]) if pd.notna(row["Fixed (yrs)"]) else 5
        term = int(row["Term (yrs)"]) if pd.notna(row["Term (yrs)"]) else 30
        name = str(row["Name"]).strip() if pd.notna(row["Name"]) else ""
        name = name or f"{deposit * 100:.0f}% {label.lower()}"
        while name in names:
            name += " (2)"
        names.add(name)
        needed = upfront_cash_needed(a, deposit)
        if needed > cash:
            problems.append(f"**{name}** needs {gbp(needed)} upfront, more than your "
                            f"{gbp(cash)}, so it's left out.")
            continue
        rate = float(row["Own rate (%)"]) / 100 if pd.notna(row["Own rate (%)"]) else \
            rate_for(deposit, fixed, typ, premium, live_df, use_live, a.mortgage_rate)
        scenarios.append(Scenario(name, deposit, rate, typ, fixed, term))
    if len(scenarios) > MAX_SCENARIOS:
        problems.append(f"Only the first {MAX_SCENARIOS} scenarios are plotted.")
        scenarios = scenarios[:MAX_SCENARIOS]
    return scenarios, problems


def custom_scenarios(a: StochasticAssumptions, cash: float, budget: float, premium: float,
                     n_sims: int, horizon: int, real: bool, live_df, use_live: bool) -> None:
    st.caption("Edit any cell, or add a row at the bottom of the table to create a new "
               "scenario (up to 8). The first row is the baseline the others are "
               "compared with. Leave **Own rate** blank to use the live rate plus any "
               "interest-only premium.")
    edited = st.data_editor(
        DEFAULT_SCENARIOS, key="lab_scenarios", num_rows="dynamic", hide_index=True,
        use_container_width=True,
        column_config={
            "Name": st.column_config.TextColumn(),
            "Deposit (%)": st.column_config.NumberColumn(min_value=0, max_value=100, step=1,
                                                         required=True, default=20),
            "Type": st.column_config.SelectboxColumn(options=list(TYPE_BY_LABEL),
                                                     required=True, default="Repayment"),
            "Fixed (yrs)": st.column_config.SelectboxColumn(options=boe.FIXED_PERIODS,
                                                            required=True, default=5),
            "Term (yrs)": st.column_config.NumberColumn(min_value=5, max_value=40, step=1,
                                                        required=True, default=30),
            "Own rate (%)": st.column_config.NumberColumn(min_value=0.1, max_value=15.0,
                                                          step=0.01, format="%.2f"),
        })
    scenarios, problems = build_scenarios(edited, a, cash, premium, live_df, use_live)
    for p in problems:
        st.warning(p)
    if not scenarios:
        st.info("Add a scenario you can afford to see results.")
        return

    results = run_scenarios(a, scenarios, cash, budget, n_sims)
    table = compare_scenarios(results, horizon, real)
    years = np.arange(1, results[0].net_worth.shape[1] + 1)
    warn_if_cash_ran_out(table, horizon)

    st.subheader(f"Who comes out ahead after {horizon} years?")
    cols = st.columns(len(results))
    for col, (_, row) in zip(cols, table.iterrows()):
        col.metric(row["Scenario"], gbp(row["Median net worth"]),
                   f"{row['Chance best (%)']:.0f}% chance best", delta_color="off")
        col.caption(f"{row['Start rate (%)']:.2f}% · {gbp(row['First payment'])}/month")

    st.subheader("Net worth over time")
    fig = go.Figure()
    for i, r in enumerate(results):
        w = wealth_matrix(r, real)
        add_band(fig, years, np.percentile(w, 50, axis=0), np.percentile(w, 10, axis=0),
                 np.percentile(w, 90, axis=0), r.scenario.name, SERIES_COLOURS[i])
    fig.update_layout(height=460, xaxis_title="Years", yaxis_title="Net worth (£)",
                      **CHART_LAYOUT)
    st.plotly_chart(fig, use_container_width=True)
    st.caption("Line = median. Shaded = 80% of simulated paths. Click a legend entry to "
               "hide a scenario.")

    if len(results) > 1:
        st.subheader(f"Gain or loss compared with {results[0].scenario.name}")
        base = wealth_matrix(results[0], real)
        fig = go.Figure()
        for i, r in enumerate(results[1:], start=1):
            d = wealth_matrix(r, real) - base
            add_band(fig, years, np.percentile(d, 50, axis=0), np.percentile(d, 10, axis=0),
                     np.percentile(d, 90, axis=0), r.scenario.name, SERIES_COLOURS[i])
        fig.add_hline(y=0, line_color="#999999", line_dash="dot")
        fig.update_layout(height=380, xaxis_title="Years", yaxis_title="Difference (£)",
                          **CHART_LAYOUT)
        st.plotly_chart(fig, use_container_width=True)
        st.caption("Every scenario faces the same simulated markets on each path, so this "
                   "shows the effect of the financing choice alone.")

    st.subheader(f"Spread of outcomes after {horizon} years")
    fig = go.Figure()
    for i, r in enumerate(results):
        fig.add_trace(go.Box(y=wealth_matrix(r, real)[:, horizon - 1] / 1e6, name=r.scenario.name,
                             marker=dict(color=SERIES_COLOURS[i]), boxmean=True, boxpoints=False))
    fig.update_layout(height=380, yaxis_title="Net worth (£ millions)", showlegend=False,
                      **{k: v for k, v in CHART_LAYOUT.items() if k not in ("hovermode", "legend")})
    st.plotly_chart(fig, use_container_width=True)

    left, right = st.columns(2)
    for col, attr, title in ((left, "payment", "Monthly payment (median)"),
                             (right, "balance", "Mortgage left (median)")):
        with col:
            st.subheader(title)
            fig = go.Figure()
            for i, r in enumerate(results):
                fig.add_trace(go.Scatter(
                    x=years, y=np.percentile(getattr(r, attr), 50, axis=0),
                    name=r.scenario.name, mode="lines",
                    line=dict(color=SERIES_COLOURS[i], width=2),
                    hovertemplate=f"{r.scenario.name}: £%{{y:,.0f}}<extra></extra>"))
            fig.update_layout(height=320, xaxis_title="Years", yaxis_title="£", **CHART_LAYOUT)
            st.plotly_chart(fig, use_container_width=True)

    st.subheader("All results")
    st.dataframe(results_table(table), use_container_width=True, hide_index=True)
    st.download_button("⬇️ Download results (CSV)", table.to_csv(index=False).encode(),
                       file_name="mortgage_scenarios.csv", mime="text/csv")


# ------------------------------------------------------------------ the tab
@st.fragment
def render_lab(a: StochasticAssumptions, fixed_years: int, term: int, real: bool,
               live_df, use_live: bool) -> None:
    """A fragment, so edits here don't re-run the Buy vs invest tab."""
    st.subheader("How should you finance this house?")
    st.caption(
        "Every option starts with the same cash and the same monthly budget. Cash not "
        "used upfront is invested on day one, and whatever the budget doesn't spend on "
        "housing is invested each month. Uses the house price and assumptions in the "
        "sidebar, except its deposit setting. Rent isn't part of this comparison.")
    c1, c2, c3, c4, c5 = st.columns(5)
    cash = c1.number_input("Cash you have (£)", 0, 5_000_000, 300_000, 10_000,
                           help="Everything available for the deposit, stamp duty and "
                                "fees. The rest gets invested.")
    budget = c2.number_input("Monthly budget (£)", 0, 50_000, 4_000, 100,
                             help="Total for mortgage, upkeep, insurance and council tax. "
                                  "Grows with inflation. The rest gets invested.")
    premium = c3.number_input("Interest-only extra rate (pp)", 0.0, 3.0, 0.25, 0.05,
                              help="Our assumption, not live data: the Bank of England "
                                   "doesn't publish interest-only rates, and they usually "
                                   "cost a little more. Check real quotes.") / 100
    horizon = c4.selectbox("Compare after", a.horizons, index=len(a.horizons) - 1,
                           format_func=lambda h: f"{h} years")
    n_sims = c5.select_slider("Simulations", [200, 500, 1000, 2000, 5000], 1000)

    tab_sweep, tab_custom = st.tabs(["🎯 Best deposit", "⚖️ Compare scenarios"])
    with tab_sweep:
        deposit_sweep(a, cash, budget, premium, n_sims, horizon, fixed_years, term, real,
                      live_df, use_live)
    with tab_custom:
        custom_scenarios(a, cash, budget, premium, n_sims, horizon, real, live_df, use_live)

    with st.expander("ℹ️ How the Mortgage lab works"):
        st.markdown(
            """
- **Interest-only:** you pay only interest, so the loan doesn't shrink. The full loan falls
  due at the end of the term and is paid from your investments, after capital gains tax.
- **Ran out of cash:** the budget and investments couldn't cover the costs on that path. The
  gap is treated as extra debt at the mortgage rate and paid off from later surpluses.
- **Chance best:** the share of simulated paths where that option ends with the most money.
- **Rates:** the live Bank of England rate for the loan's LTV band. It only publishes LTV bands
  for 2- and 5-year fixes, so 3- and 10-year fixes use the 75% LTV rate at every deposit.
  Nothing is published above 90% LTV, so a 5% deposit uses the 90% band and may look too cheap.
- **Not modelled:** lender rules. UK lenders often need a large deposit and a credible repayment
  plan for interest-only, and affordability checks can rule out some options. There are also
  no ISA or pension tax wrappers and no remortgage fees.
- **Not financial advice.** This is a tool for exploring scenarios.
            """
        )
