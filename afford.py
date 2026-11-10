"""
Streamlit UI for the household's finances and the affordability matrix: monthly
payments by loan size and rate, for several terms, checked against the budget.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

import rates as boe
import settings
from model import budget_path, monthly_budget, mortgage_scenario, repayment_schedule
from ui import CHART_LAYOUT, SERIES_COLOURS, gbp

TERMS = [10, 15, 20, 25, 30, 35, 40]
# Status colours with a symbol, so feasibility never relies on colour alone
STATUS = {"ok": ("#e3f4ea", "✓"), "tight": ("#fff1cc", "!"), "over": ("#fde2e1", "✗")}
TIGHT = 0.9  # within 10% of the budget counts as tight

DEFAULT_COSTS = pd.DataFrame([
    {"Cost": "Bills and living costs", "£ per month": 3_500,
     "Starts in (yrs)": None, "Stops in (yrs)": None},
    {"Cost": "Childcare", "£ per month": 0, "Starts in (yrs)": None, "Stops in (yrs)": None},
]).astype({"£ per month": "float", "Starts in (yrs)": "float", "Stops in (yrs)": "float"})


@dataclass
class Finances:
    income: float
    costs: list
    savings: float
    income_growth: float
    cost_growth: float

    def budget(self, years: float) -> float:
        return monthly_budget(self.income, self.costs, years, self.income_growth,
                              self.cost_growth)

    def budget_path(self, months: int) -> np.ndarray:
        return budget_path(self.income, self.costs, months, self.income_growth,
                           self.cost_growth)


def status(payment: float, budget: float) -> str:
    if payment > budget:
        return "over"
    return "tight" if payment > budget * TIGHT else "ok"


def nice_range(price: float, upfront_fees: float, savings: float) -> tuple[int, int]:
    """Default loan range: from the smallest loan your savings allow (or 30% of
    the price) up to 90% of the price, in £50k steps."""
    smallest = max(price + upfront_fees - savings, price * 0.3, 50_000)
    lo = int(np.ceil(smallest / 50_000)) * 50_000
    hi = int(price * 0.9 // 50_000) * 50_000
    lo = min(lo, hi)
    return lo, max(hi, lo)


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


def finances_panel(inflation: float, savings: float) -> Finances:
    """Income, costs and savings shared by every section of the Mortgage tab."""
    left, right = st.columns([3, 2])
    with left:
        income = st.number_input("Combined monthly take-home income (£)", 0, 200_000,
                                 8_500, 100, key="fin_income")
        rows, editor_key = settings.table("fin_costs", DEFAULT_COSTS)
        costs = st.data_editor(
            rows, key=editor_key, num_rows="dynamic", hide_index=True,
            width="stretch",
            column_config={
                "Cost": st.column_config.TextColumn(width="medium"),
                "£ per month": st.column_config.NumberColumn(
                    min_value=0, step=50, format="£%.0f", default=0, width="small",
                    help="In today's money. It rises with costs each year."),
                "Starts in (yrs)": st.column_config.NumberColumn(
                    min_value=0, max_value=40, step=1, width="small",
                    help="Leave blank if it applies now, e.g. 5 for school fees starting."),
                "Stops in (yrs)": st.column_config.NumberColumn(
                    min_value=0, max_value=40, step=1, width="small",
                    help="Leave blank if it never stops, e.g. 4 for childcare ending."),
            })
        settings.remember_table("fin_costs", costs)
        st.caption("List everything except the mortgage: bills, food, childcare, council "
                   "tax, upkeep. Add rows for costs you can see coming.")
    with right:
        st.metric("Savings", gbp(savings), help="Set in the sidebar under Property.")
        g1, g2 = st.columns(2)
        income_growth = g1.number_input("Pay rises (%/yr)", 0.0, 10.0, 2.5, 0.25,
                                        key="fin_income_growth") / 100
        cost_growth = g2.number_input("Costs rise (%/yr)", 0.0, 10.0,
                                      round(inflation * 100, 2), 0.25,
                                      key="fin_cost_growth") / 100

    def years(v):
        return None if pd.isna(v) else float(v)
    cost_list = [(float(r["£ per month"]), years(r["Starts in (yrs)"]), years(r["Stops in (yrs)"]))
                 for _, r in costs.iterrows() if pd.notna(r["£ per month"])]
    fin = Finances(income, cost_list, savings, income_growth, cost_growth)
    with right:
        st.metric("Available for the mortgage today", gbp(fin.budget(0)) + " /month")
    return fin


def affordability_section(fin: Finances, price: float, sdlt: float, legal: float,
                          default_rate: float, live_df) -> None:
    st.caption(f"Monthly repayments for different mortgage sizes, rates and terms, checked "
               f"against your budget. House price {gbp(price)} from the sidebar, stamp "
               f"duty {gbp(sdlt)}.")
    left, right = st.columns(2)
    with left:
        lo, hi = nice_range(price, sdlt + legal, fin.savings)
        amounts_range = st.slider("Mortgage sizes (£)", 0, int(price), (lo, hi), 10_000,
                                  key=settings.dynamic_key("aff_amounts", (lo, hi)))
        step = st.select_slider("Gap between rows (£)", [10_000, 25_000, 50_000, 100_000],
                                100_000 if hi - lo >= 600_000 else 50_000, key="aff_step")
        terms = st.multiselect("Mortgage terms (years)", TERMS, [20, 25, 30], key="aff_terms")
    with right:
        rate_lo, rate_hi = st.slider("Interest rates (%)", 0.5, 12.0, (2.0, 8.0), 0.25,
                                     key="aff_rates")
        rate_step = st.select_slider("Gap between columns (%)", [0.25, 0.5, 1.0], 1.0,
                                     key="aff_rate_step")

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
    budget_now = fin.budget(0)
    budget = fin.budget(year)
    m1, m2 = st.columns(2)
    if year:
        m1.metric(f"Available for the mortgage in year {year}", gbp(budget) + " /month",
                  f"{gbp(budget - budget_now)} vs today", delta_color="normal", delta_arrow="off")
    live = boe.latest_rate(live_df, boe.HEADLINE[5]) if live_df is not None else None
    if live:
        m2.metric("Today's average 5-year fix", f"{live[0] * 100:.2f}%",
                  help=f"Bank of England, 75% LTV, {live[1]:%b %Y}. Excludes fees.")

    if year:
        st.caption(f"In {year} years: you've paid {start_rate * 100:.2f}% until then, and the "
                   f"columns show the payment if you remortgage at that rate for the rest "
                   f"of the term. Totals cover the whole term.")
    st.caption(f"✓ fits your budget · ! within 10% of it · ✗ over it. Greyed rows need more "
               f"than your savings. Cash needed = deposit + stamp duty + {gbp(legal)} fees.")

    for term in terms:
        st.markdown(f"#### {term}-year mortgage · {show.lower()}")
        styler, _ = matrix(amounts, rate_grid, term, year, start_rate, price,
                           sdlt + legal, fin.savings, budget, show)
        st.dataframe(styler, use_container_width=True)

    # ---------------------------------------------------- one mortgage in detail
    st.divider()
    st.subheader("Look closer at one mortgage")
    d1, d2 = st.columns(2)
    amount = d1.selectbox("Mortgage", amounts, format_func=gbp,
                          index=int(np.argmin(np.abs(amounts - np.median(amounts)))),
                          key="aff_detail_amount")
    rate = d2.number_input("Interest rate (%)", 0.5, 12.0,
                           round(default_rate * 100, 2), 0.05, key="aff_detail_rate") / 100
    o1, o2, o3 = st.columns(3)
    overpay = o1.number_input("Overpay each month (£)", 0, 50_000, 0, 50, key="aff_overpay",
                              help="Extra on top of the monthly payment. It clears the "
                                   "mortgage sooner and cuts the interest.")
    lump = o2.number_input("One-off lump sum (£)", 0, 5_000_000, 0, 1_000, key="aff_lump")
    lump_year = o3.number_input("Paid at the end of year", 1, 40, 5, 1, key="aff_lump_year")
    lumps = [(lump_year, lump)] if lump else []
    overpaying = overpay > 0 or lump > 0
    if year:
        st.caption(f"Pays {start_rate * 100:.2f}% for {year} years, then {rate * 100:.2f}%.")
    if overpaying:
        st.caption("Most fixed-rate deals let you overpay up to 10% of the balance each year "
                   "without an early repayment charge. Check your lender's limit.")

    cols = st.columns(len(terms))
    schedules = {}
    this_year = date.today().year
    for col, term in zip(cols, terms):
        switch = (year, start_rate) if year and year < term else (0, None)
        sched = repayment_schedule(amount, rate, term, *switch, overpay, lumps)
        plain = repayment_schedule(amount, rate, term, *switch) if overpaying else sched
        schedules[term] = sched
        first = sched["payment"].iloc[0]
        later = sched["payment"].iloc[year * 12] if year and year < term else first
        months_live = int((sched["balance"] > 1e-6).sum()) + 1
        free_in = min(months_live, term * 12) / 12
        col.markdown(f"**{term} years**")
        col.metric("Monthly payment", gbp(first + overpay),
                   f"{gbp(later - first)} from year {year}" if abs(later - first) >= 1 else None,
                   delta_color="inverse", delta_arrow="off",
                   help="Including your monthly overpayment." if overpay else None)
        saved = plain["interest"].sum() - sched["interest"].sum()
        col.metric("Total interest", gbp(sched["interest"].sum()),
                   f"{gbp(saved)} less" if saved >= 1 else None,
                   delta_color="normal", delta_arrow="off")
        col.metric("Total repaid", gbp((sched["payment"] + sched["overpayment"]).sum()))
        sooner = term - free_in
        col.caption(f"Mortgage-free in {this_year + int(np.ceil(free_in))}"
                    + (f" ({sooner:.1f} years sooner)" if sooner >= 0.1 else "") + " · "
                    f"{status(first + overpay, budget_now).replace('ok', 'fits today').replace('tight', 'tight today').replace('over', 'over budget today')}")

    years_axis = np.arange(0, max(terms) * 12) / 12
    budget_line = [fin.budget(y) for y in years_axis]

    st.markdown("**Monthly repayment over time**")
    fig = go.Figure()
    for i, term in enumerate(terms):
        s = schedules[term]
        fig.add_trace(go.Scatter(
            x=s["year"], y=s["payment"] + s["overpayment"], name=f"{term} years", mode="lines",
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
        s = s.assign(principal=s["principal"] + s["overpayment"])
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

    with st.expander("How the affordability matrix works"):
        st.markdown(
            """
- **Monthly payment:** a standard repayment mortgage, the same formula lenders use.
- **In N years:** you pay the "rate until then" for N years. Each column then shows the
  payment if you remortgage at that rate for the rest of the term. If the rate doesn't
  change, the payment doesn't either.
- **Your budget:** take-home income minus your costs. Pay rises and rising costs only
  matter when you look ahead. Costs join in their start year and drop out after their
  stop year.
- **Cash needed:** deposit (price minus mortgage) + stamp duty + legal fees.
- **Overpayments** (detail view only): paid on top of the monthly payment, they clear the
  loan sooner. The matrix shows the normal payment. At a remortgage the new payment is
  worked out on the smaller balance, so it drops.
- **Not included:** product fees and lender affordability rules (often 4 to 5 times income).
  **Not financial advice.**
            """
        )
