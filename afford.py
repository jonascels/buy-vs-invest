"""
Streamlit UI for the Affordability tab: a matrix of monthly payments by loan
size and rate, for several terms, checked against what the household can spend.
"""

from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

import rates as boe
from lab import CHART_LAYOUT, SERIES_COLOURS, gbp
from model import monthly_budget, mortgage_scenario, repayment_schedule, stamp_duty

TERMS = [10, 15, 20, 25, 30, 35, 40]
# Status colours with a symbol, so feasibility never relies on colour alone
STATUS = {"ok": ("#e3f4ea", "✓"), "tight": ("#fff1cc", "!"), "over": ("#fde2e1", "✗")}
TIGHT = 0.9  # within 10% of the budget counts as tight

DEFAULT_COSTS = pd.DataFrame([
    {"Cost": "Bills and living costs", "£ per month": 3_000, "Stops in (yrs)": None},
    {"Cost": "Childcare", "£ per month": 0, "Stops in (yrs)": None},
]).astype({"£ per month": "float", "Stops in (yrs)": "float"})


def status(payment: float, budget: float) -> str:
    if payment > budget:
        return "over"
    return "tight" if payment > budget * TIGHT else "ok"


def nice_range(price: float) -> tuple[int, int]:
    """Default loan range: about 30% to 80% of the price, in £50k steps."""
    lo = max(int(round(price * 0.3 / 50_000)) * 50_000, 50_000)
    hi = max(int(round(price * 0.8 / 50_000)) * 50_000, lo)
    return lo, min(hi, int(price))


def matrix(amounts: np.ndarray, rate_grid: np.ndarray, term: int, year: int,
           start_rate: float, price: float, upfront_fees: float, savings: float,
           budget: float, show: str):
    """One term's matrix as a pandas Styler."""
    s = mortgage_scenario(amounts[:, None], rate_grid[None, :], term, year,
                          start_rate if year else None)
    payment = s["payment"]
    values = {"Monthly payment": payment, "Total interest": s["total_interest"],
              "Total repaid": s["total_repaid"],
              "Headroom": budget - payment}[show]

    cash = price - amounts + upfront_fees
    short = cash > savings
    paid_off = year >= term
    rate_cols = [f"{r * 100:g}%" for r in rate_grid]

    # Build the displayed text directly, so each cell's symbol matches its own row
    info = {"Cash needed": [gbp(c) for c in cash],
            "LTV": [f"{a / price:.0%}" for a in amounts],
            "Share of savings": [f"{c / savings:.0%}" if savings else "–" for c in cash]}
    if year:
        info[f"Owed in year {year}"] = [gbp(b) for b in s["balance"][:, 0]]
    text = pd.DataFrame(info)
    colours = pd.DataFrame("", index=text.index, columns=text.columns)
    for j, col in enumerate(rate_cols):
        cells, styles = [], []
        for i in range(len(amounts)):
            if short[i]:
                cells.append(gbp(values[i, j]))
                styles.append("color: #9e9e9e; background-color: #f2f2f2")
            elif paid_off:
                cells.append("Paid off" if show == "Monthly payment" else gbp(values[i, j]))
                styles.append("color: #6b6b6b")
            else:
                key = status(payment[i, j], budget)
                cells.append(f"{gbp(values[i, j])} {STATUS[key][1]}")
                styles.append(f"background-color: {STATUS[key][0]}; color: #1f1f1f")
        text[col], colours[col] = cells, styles
    for c in info:
        colours.loc[short, c] = "color: #9e9e9e"
    text.index = colours.index = [gbp(a) for a in amounts]
    text.index.name = "Mortgage"
    return text.style.apply(lambda _: colours, axis=None), short.any()


@st.fragment
def render_affordability(price: float, ftb: bool, addl: bool, legal: float,
                         inflation: float, default_rate: float, live_df) -> None:
    """A fragment, so edits here don't re-run the other tabs."""
    st.subheader("What can we afford?")
    sdlt = stamp_duty(price, ftb, addl)
    st.caption(f"Monthly repayments for different mortgage sizes, rates and terms, checked "
               f"against what you can spend each month. Uses the house price in the sidebar "
               f"({gbp(price)}, stamp duty {gbp(sdlt)}).")

    left, right = st.columns([1, 1])
    with left:
        st.markdown("**Your monthly budget**")
        income = st.number_input("Combined monthly take-home income (£)", 0, 200_000,
                                 6_000, 100, key="aff_income")
        costs = st.data_editor(
            DEFAULT_COSTS, key="aff_costs", num_rows="dynamic", hide_index=True,
            width="stretch",
            column_config={
                "Cost": st.column_config.TextColumn(width="medium"),
                "£ per month": st.column_config.NumberColumn(min_value=0, step=50,
                                                             format="£%.0f", default=0,
                                                             width="small"),
                "Stops in (yrs)": st.column_config.NumberColumn(
                    min_value=0, max_value=40, step=1, width="small",
                    help="Leave blank if it never stops, e.g. 4 for childcare ending."),
            })
        cost_list = [(float(r["£ per month"]),
                      None if pd.isna(r["Stops in (yrs)"]) else float(r["Stops in (yrs)"]))
                     for _, r in costs.iterrows() if pd.notna(r["£ per month"])]
        st.caption("Include everything except the mortgage, such as bills, food, "
                   "childcare, council tax and upkeep.")
    with right:
        st.markdown("**The mortgages to compare**")
        savings = st.number_input("Savings for deposit, stamp duty and fees (£)", 0,
                                  20_000_000, int(min(price, 500_000)), 10_000,
                                  key="aff_savings")
        lo, hi = nice_range(price)
        amounts_range = st.slider("Mortgage sizes (£)", 0, int(price), (lo, hi), 10_000,
                                  key=f"aff_amounts_{int(price)}")
        step = st.select_slider("Gap between rows (£)", [10_000, 25_000, 50_000, 100_000],
                                100_000 if hi - lo >= 400_000 else 50_000, key="aff_step")
        rate_lo, rate_hi = st.slider("Interest rates (%)", 0.5, 12.0, (2.0, 8.0), 0.25,
                                     key="aff_rates")
        rate_step = st.select_slider("Gap between columns (%)", [0.25, 0.5, 1.0], 1.0,
                                     key="aff_rate_step")
        terms = st.multiselect("Mortgage terms (years)", TERMS, [20, 25, 30], key="aff_terms")

    if not terms:
        st.info("Pick at least one mortgage term.")
        return
    terms = sorted(terms)
    amounts = np.arange(amounts_range[0], amounts_range[1] + 1, step, dtype=float)
    amounts = amounts[amounts > 0]
    if len(amounts) == 0:
        st.info("Choose a mortgage range above £0.")
        return
    rate_grid = np.round(np.arange(rate_lo, rate_hi + 1e-9, rate_step), 4) / 100

    st.divider()
    c1, c2, c3 = st.columns([2, 1, 1])
    year = c1.select_slider(
        "Look at the matrix", [0, 5, 10, 15, 20, 25, 30],
        format_func=lambda y: "Today" if y == 0 else f"In {y} years", key="aff_year")
    show = c2.selectbox("Show", ["Monthly payment", "Total interest", "Total repaid",
                                 "Headroom"], key="aff_show",
                        help="Headroom = your monthly budget minus the payment.")
    start_rate = default_rate
    if year:
        start_rate = c3.number_input(
            f"Rate until year {year} (%)", 0.5, 12.0, round(default_rate * 100, 2), 0.05,
            key="aff_start_rate",
            help="What you pay until then. The columns then show the rate you "
                 "remortgage at.") / 100
    with st.expander("📈 Future years: pay rises and rising costs"):
        g1, g2 = st.columns(2)
        income_growth = g1.number_input("Pay rises (%/yr)", 0.0, 10.0, 2.5, 0.25,
                                        key="aff_income_growth") / 100
        cost_growth = g2.number_input("Costs rise (%/yr)", 0.0, 10.0,
                                      round(inflation * 100, 2), 0.25,
                                      key="aff_cost_growth") / 100

    budget_now = monthly_budget(income, cost_list, 0)
    budget = monthly_budget(income, cost_list, year, income_growth, cost_growth)
    m1, m2, m3 = st.columns(3)
    m1.metric("Available for the mortgage today", gbp(budget_now) + " /month")
    if year:
        m2.metric(f"Available in year {year}", gbp(budget) + " /month",
                  f"{gbp(budget - budget_now)} vs today", delta_color="normal")
    live = boe.latest_rate(live_df, boe.HEADLINE[5]) if live_df is not None else None
    if live:
        m3.metric("Today's average 5-year fix", f"{live[0] * 100:.2f}%",
                  help=f"Bank of England, 75% LTV, {live[1]:%b %Y}. Excludes fees.")

    if year:
        st.caption(f"In {year} years: you've paid {start_rate * 100:.2f}% until then, and the "
                   f"columns show the payment if you remortgage at that rate for the rest "
                   f"of the term. Totals cover the whole term.")
    st.caption(f"✓ fits your budget · ! within 10% of it · ✗ over it. Greyed rows need more "
               f"than your savings. Cash needed = deposit + stamp duty + {gbp(legal)} fees.")

    any_short = False
    for term in terms:
        st.markdown(f"#### {term}-year mortgage · {show.lower()}")
        styler, short = matrix(amounts, rate_grid, term, year, start_rate, price,
                               sdlt + legal, savings, budget, show)
        any_short |= short
        st.dataframe(styler, use_container_width=True)
    if any_short:
        st.caption("Greyed-out rows need more cash upfront than your savings.")

    # ---------------------------------------------------- one mortgage in detail
    st.divider()
    st.subheader("Look closer at one mortgage")
    d1, d2 = st.columns(2)
    amount = d1.selectbox("Mortgage", amounts, format_func=gbp,
                          index=int(np.argmin(np.abs(amounts - np.median(amounts)))),
                          key="aff_detail_amount")
    rate = d2.number_input("Interest rate (%)", 0.5, 12.0,
                           round(default_rate * 100, 2), 0.05, key="aff_detail_rate") / 100
    if year:
        st.caption(f"Pays {start_rate * 100:.2f}% for {year} years, then {rate * 100:.2f}%.")

    cols = st.columns(len(terms))
    schedules = {}
    this_year = date.today().year
    for col, term in zip(cols, terms):
        sched = repayment_schedule(amount, rate, term, year if year < term else 0,
                                   start_rate if year and year < term else None)
        schedules[term] = sched
        first = sched["payment"].iloc[0]
        later = sched["payment"].iloc[year * 12] if year and year < term else first
        col.markdown(f"**{term} years**")
        col.metric("Monthly payment", gbp(first),
                   f"{gbp(later - first)} from year {year}" if abs(later - first) >= 1 else None,
                   delta_color="inverse")
        col.metric("Total interest", gbp(sched["interest"].sum()))
        col.metric("Total repaid", gbp(sched["payment"].sum()))
        col.caption(f"Mortgage-free in {this_year + term} · "
                    f"{status(first, budget_now).replace('ok', 'fits today').replace('tight', 'tight today').replace('over', 'over budget today')}")

    years_axis = np.arange(0, max(terms) * 12) / 12
    budget_line = [monthly_budget(income, cost_list, y, income_growth, cost_growth)
                   for y in years_axis]

    st.markdown("**Monthly repayment over time**")
    fig = go.Figure()
    for i, term in enumerate(terms):
        s = schedules[term]
        fig.add_trace(go.Scatter(
            x=s["year"], y=s["payment"], name=f"{term} years", mode="lines",
            line=dict(color=SERIES_COLOURS[i], width=2, shape="hv"),
            hovertemplate=f"{term} yrs: £%{{y:,.0f}}<extra></extra>"))
    fig.add_trace(go.Scatter(
        x=years_axis, y=budget_line, name="Your budget", mode="lines",
        line=dict(color="#777777", width=2, dash="dot"),
        hovertemplate="Budget: £%{y:,.0f}<extra></extra>"))
    fig.update_layout(height=360, xaxis_title="Years from now",
                      yaxis_title="£ per month", **CHART_LAYOUT)
    st.plotly_chart(fig, use_container_width=True)
    st.caption("The dotted line is what you can spend on the mortgage, after pay rises, "
               "rising costs and any costs that stop.")

    left, right = st.columns(2)
    with left:
        st.markdown("**Mortgage left**")
        fig = go.Figure()
        for i, term in enumerate(terms):
            s = schedules[term]
            fig.add_trace(go.Scatter(
                x=s["year"], y=s["balance"], name=f"{term} years", mode="lines",
                line=dict(color=SERIES_COLOURS[i], width=2),
                hovertemplate=f"{term} yrs: £%{{y:,.0f}}<extra></extra>"))
        fig.update_layout(height=300, xaxis_title="Years from now", yaxis_title="£",
                          **CHART_LAYOUT)
        st.plotly_chart(fig, use_container_width=True)
    with right:
        term_pick = st.selectbox("Interest vs paying off the loan", terms,
                                 index=len(terms) // 2,
                                 format_func=lambda t: f"{t}-year mortgage",
                                 key="aff_split_term")
        s = schedules[term_pick]
        yearly = s.groupby(np.ceil(s["year"]).astype(int))[["interest", "principal"]].sum()
        fig = go.Figure()
        fig.add_trace(go.Bar(x=yearly.index, y=yearly["interest"], name="Interest",
                             marker_color=SERIES_COLOURS[1],
                             hovertemplate="Interest: £%{y:,.0f}<extra></extra>"))
        fig.add_trace(go.Bar(x=yearly.index, y=yearly["principal"], name="Paying off the loan",
                             marker_color=SERIES_COLOURS[0],
                             hovertemplate="Loan paid off: £%{y:,.0f}<extra></extra>"))
        fig.update_layout(height=300, barmode="stack", bargap=0.15, xaxis_title="Year",
                          yaxis_title="£ per year", **CHART_LAYOUT)
        st.plotly_chart(fig, use_container_width=True)

    with st.expander("ℹ️ How the affordability matrix works"):
        st.markdown(
            """
- **Monthly payment:** a standard repayment mortgage, the same formula lenders use.
- **In N years:** you pay the "rate until then" for N years. Each column then shows the
  payment if you remortgage at that rate for the rest of the term. If the rate doesn't
  change, the payment doesn't either.
- **Your budget:** take-home income minus your costs. Pay rises and rising costs only
  matter when you look ahead, and costs with a stop year drop out after it.
- **Cash needed:** deposit (price minus mortgage) + stamp duty + legal fees.
- **Not included:** product fees, lender affordability rules (often 4 to 5 times income)
  and overpayments. **Not financial advice.**
            """
        )
