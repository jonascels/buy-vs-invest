"""
Streamlit UI for the Mortgage tab: your finances, what you can afford, which
deposit and mortgage type work best over time, and variable vs fixed history.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

import history
import rates as boe
import settings
from afford import Finances, affordability_section, finances_panel
from model import (INTEREST_ONLY, REPAYMENT, Scenario, ScenarioResult,
                   StochasticAssumptions, compare_scenarios, run_scenarios,
                   stamp_duty, upfront_cash_needed)
from ui import BOX_LAYOUT, CHART_LAYOUT, SERIES_COLOURS, add_band, gbp

N_SIMS = 1_000
MAX_SCENARIOS = len(SERIES_COLOURS)
TYPE_LABELS = {REPAYMENT: "Repayment", INTEREST_ONLY: "Interest-only"}
TYPE_BY_LABEL = {v: k for k, v in TYPE_LABELS.items()}


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


def table_expander(table: pd.DataFrame, file_name: str) -> None:
    with st.expander("All results"):
        st.dataframe(results_table(table), use_container_width=True, hide_index=True)
        st.download_button("Download results (CSV)", table.to_csv(index=False).encode(),
                           file_name=file_name, mime="text/csv", key=f"dl_{file_name}")


def warn_if_cash_ran_out(df: pd.DataFrame, horizon: int) -> None:
    short = df[df["Ran out of cash (%)"] > 0]
    if len(short):
        worst = short.loc[short["Ran out of cash (%)"].idxmax()]
        st.warning(
            f"{len(short)} of {len(df)} options ran out of money within {horizon} years on "
            f"some paths (worst: {worst['Scenario']}, {worst['Ran out of cash (%)']:.0f}% of "
            "paths). The shortfall is treated as extra debt at the mortgage rate. A real "
            "lender may not allow that, so treat these results with care.")


# ------------------------------------------------------------------ compare mortgages
DEFAULT_SCENARIOS = pd.DataFrame([
    {"Name": "20% repayment", "Deposit (%)": 20, "Type": "Repayment",
     "Fixed (yrs)": 5, "Term (yrs)": 30, "Own rate (%)": None},
    {"Name": "30% repayment", "Deposit (%)": 30, "Type": "Repayment",
     "Fixed (yrs)": 5, "Term (yrs)": 30, "Own rate (%)": None},
    {"Name": "20% interest-only", "Deposit (%)": 20, "Type": "Interest-only",
     "Fixed (yrs)": 5, "Term (yrs)": 30, "Own rate (%)": None},
]).astype({"Own rate (%)": "float"})


def build_scenarios(edited: pd.DataFrame, a: StochasticAssumptions, cash: float,
                    premium: float, live_df, use_live: bool):
    """Turn editor rows into Scenarios; returns (scenarios, problems)."""
    scenarios, problems, names = [], [], set()
    for _, row in edited.iterrows():
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
                            f"{gbp(cash)} savings, so it's left out.")
            continue
        rate = float(row["Own rate (%)"]) / 100 if pd.notna(row["Own rate (%)"]) else \
            rate_for(deposit, fixed, typ, premium, live_df, use_live, a.mortgage_rate)
        scenarios.append(Scenario(name, deposit, rate, typ, fixed, term))
    if len(scenarios) > MAX_SCENARIOS:
        problems.append(f"Only the first {MAX_SCENARIOS} scenarios are plotted.")
        scenarios = scenarios[:MAX_SCENARIOS]
    return scenarios, problems


def compare_section(a: StochasticAssumptions, fin: Finances, fixed_years: int, term: int,
                    real: bool, live_df, use_live: bool) -> None:
    st.caption(
        "Which deposit and mortgage type leave you better off over time? Every option starts "
        "with your savings and your monthly budget above. Savings not used upfront are "
        "invested, and whatever the budget doesn't spend on the mortgage is invested each "
        "month. 1,000 simulated futures for markets and rates.")
    c1, c2 = st.columns(2)
    horizon = c1.selectbox("Compare after", a.horizons, index=len(a.horizons) - 1,
                           format_func=lambda h: f"{h} years", key="cmp_horizon")
    premium = c2.number_input(
        "Interest-only extra rate (pp)", 0.0, 3.0, 0.25, 0.05, key="cmp_premium",
        help="Our assumption, not live data: the Bank of England doesn't publish "
             "interest-only rates, and they usually cost a little more.") / 100
    budgets = fin.budget_path(max(a.horizons) * 12)
    cash = fin.savings

    def run(scenarios):
        return run_scenarios(a, scenarios, cash, budgets, N_SIMS,
                             include_running_costs=False)

    # ---- which deposit
    st.subheader("Which deposit works best?")
    affordable = [d for d in range(5, 101, 5) if upfront_cash_needed(a, d / 100) <= cash]
    if not affordable:
        st.warning(f"Your savings don't cover the smallest deposit. Stamp duty and fees "
                   f"alone are {gbp(upfront_cash_needed(a, 0.0))}.")
    else:
        st.caption(f"Tests every deposit from 5% to {affordable[-1]}% (the most your savings "
                   f"allow), with a {fixed_years}-year fix and a {term}-year term from the "
                   "sidebar, for both repayment and interest-only.")
        sweep = [Scenario(f"{d}% {label.lower()}", d / 100,
                          rate_for(d / 100, fixed_years, TYPE_BY_LABEL[label], premium,
                                   live_df, use_live, a.mortgage_rate),
                          TYPE_BY_LABEL[label], fixed_years, term)
                 for label in TYPE_BY_LABEL for d in affordable]
        results = run(sweep)
        table = compare_scenarios(results, horizon, real)
        top = table.loc[table["Median net worth"].idxmax()]
        safest = table.loc[table["10th %ile"].idxmax()]
        spread = table["Median net worth"].max() - table["Median net worth"].min()
        m1, m2, m3 = st.columns(3)
        m1.metric("Highest typical outcome", top["Scenario"], gbp(top["Median net worth"]),
                  delta_color="off", delta_arrow="off", help="Highest median net worth at this horizon.")
        m2.metric("Best in a bad scenario", safest["Scenario"], gbp(safest["10th %ile"]),
                  delta_color="off", delta_arrow="off",
                  help="Highest 10th percentile: only 1 path in 10 ends up lower.")
        m3.metric("Median gap, best to worst", gbp(spread),
                  help="If this is small next to the shaded range on the chart, the deposit "
                       "matters less than how markets turn out.")
        warn_if_cash_ran_out(table, horizon)

        # Paths share shocks, so a path-by-path difference strips out market noise
        base = results[0]
        base_w = wealth_matrix(base, real)[:, horizon - 1]
        fig = go.Figure()
        for i, label in enumerate(TYPE_BY_LABEL):
            rows = [r for r in results if TYPE_LABELS[r.scenario.repayment_type] == label]
            diffs = np.array([wealth_matrix(r, real)[:, horizon - 1] - base_w for r in rows])
            add_band(fig, [r.scenario.deposit_pct * 100 for r in rows],
                     np.percentile(diffs, 50, axis=1), np.percentile(diffs, 10, axis=1),
                     np.percentile(diffs, 90, axis=1), label, SERIES_COLOURS[i])
        fig.add_hline(y=0, line_color="#999999", line_dash="dot")
        fig.update_layout(height=360, xaxis_title="Deposit (%)",
                          yaxis_title=f"Better or worse off than {base.scenario.name} (£)",
                          **CHART_LAYOUT)
        st.plotly_chart(fig, use_container_width=True)
        st.caption(f"Net worth after {horizon} years compared with {base.scenario.name}. "
                   "Line = median, shaded = 80% of simulated futures. Every option faces the "
                   "same futures, so this shows the effect of the choice alone. Small jumps "
                   "come from cheaper rates at lower loan-to-value.")
        table_expander(table, "deposit_comparison.csv")

    # ---- own scenarios
    st.subheader("Compare your own options")
    st.caption("Edit any cell, or add a row to create an option (up to 8). Leave **Own rate** "
               "blank to use the live rate. The first row is the baseline.")
    rows, editor_key = settings.table("cmp_scenarios", DEFAULT_SCENARIOS)
    edited = st.data_editor(
        rows, key=editor_key, num_rows="dynamic", hide_index=True,
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
    settings.remember_table("cmp_scenarios", edited)
    scenarios, problems = build_scenarios(edited, a, cash, premium, live_df, use_live)
    for p in problems:
        st.warning(p)
    if not scenarios:
        st.info("Add an option your savings can cover to see results.")
        return

    results = run(scenarios)
    table = compare_scenarios(results, horizon, real)
    years = np.arange(1, results[0].net_worth.shape[1] + 1)
    warn_if_cash_ran_out(table, horizon)

    cols = st.columns(len(results))
    for col, (_, row) in zip(cols, table.iterrows()):
        col.metric(row["Scenario"], gbp(row["Median net worth"]),
                   f"{row['Chance best (%)']:.0f}% chance best", delta_color="off", delta_arrow="off")
        col.caption(f"{row['Start rate (%)']:.2f}% · {gbp(row['First payment'])}/month")

    left, right = st.columns([3, 2])
    with left:
        fig = go.Figure()
        for i, r in enumerate(results):
            w = wealth_matrix(r, real)
            add_band(fig, years, np.percentile(w, 50, axis=0), np.percentile(w, 10, axis=0),
                     np.percentile(w, 90, axis=0), r.scenario.name, SERIES_COLOURS[i])
        fig.update_layout(height=400, xaxis_title="Years", yaxis_title="Net worth (£)",
                          **CHART_LAYOUT)
        st.plotly_chart(fig, use_container_width=True)
        st.caption("Net worth over time. Line = median, shaded = 80% of simulated futures.")
    with right:
        fig = go.Figure()
        for i, r in enumerate(results):
            fig.add_trace(go.Box(y=wealth_matrix(r, real)[:, horizon - 1] / 1e6,
                                 name=r.scenario.name, marker=dict(color=SERIES_COLOURS[i]),
                                 boxmean=True, boxpoints=False))
        fig.update_layout(height=400, yaxis_title=f"Net worth after {horizon} years (£m)",
                          showlegend=False, **BOX_LAYOUT)
        st.plotly_chart(fig, use_container_width=True)
        st.caption("Spread of outcomes. Box = middle 50%, line = median, dashed = mean.")
    table_expander(table, "mortgage_options.csv")


# ------------------------------------------------------------------ variable vs fixed
@st.cache_data(ttl=6 * 3600, show_spinner="Fetching rate history…")
def load_history():
    """Bank of England rate history, or None if unreachable."""
    try:
        return history.fetch_history()
    except Exception:
        return None


@st.cache_data(show_spinner="Replaying rate history…")
def cached_test(d: pd.DataFrame, years: int, method: str, block: int, loan: float,
                term: int) -> dict:
    return history.run_test(d, years, method, block, 5_000, loan, term)


def variable_vs_fixed(price: float) -> None:
    st.caption(
        "Is a variable mortgage cheaper than a series of 2-year fixes? This replays real "
        "Bank of England rates. Your finances above don't apply here. Every "
        "option borrows the same amount over the same months, and savings are in interest "
        "paid, compared with rolling 2-year fixes.")
    d = load_history()
    if d is None:
        st.warning("Couldn't reach the Bank of England database just now, so this test "
                   "can't run. Try again later.")
        return

    c1, c2, c3, c4 = st.columns(4)
    loan = c1.number_input("Loan (£)", 50_000, 5_000_000,
                           int(round(price * 0.75, -4)), 10_000, key="hist_loan")
    term = c2.slider("Mortgage term (yrs)", 10, 40, 25, 1, key="hist_term")
    choices = [y for y in (5, 10, 15, 20, 25) if y <= term]
    years = c3.selectbox("Compare over", choices, index=choices.index(min(10, term)),
                         format_func=lambda y: f"{y} years", key="hist_years")
    method = c4.radio(
        "History", ("Continuous", "2-year segments", "5-year segments"), key="hist_method",
        help="Continuous uses every possible start month in real order. Segments stitch "
             "random pieces of history together, 5,000 times.")
    kw = {"Continuous": ("continuous", 24), "2-year segments": ("segments", 24),
          "5-year segments": ("segments", 60)}[method]

    result = cached_test(d, years, kw[0], kw[1], loan, term)
    table = history.summary(result)
    span = f"{d.index[0]:%b %Y} to {d.index[-1]:%b %Y}"

    st.subheader(f"How often was variable cheaper than rolling 2-year fixes, "
                 f"over {years} years?")
    cols = st.columns(len(table))
    for col, (_, row) in zip(cols, table.iterrows()):
        col.metric(row["Variable type"], f"{row['Variable cheaper (%)']:.0f}% of the time",
                   f"Median {gbp(row['Median saving'])} saved"
                   if row["Median saving"] >= 0 else f"Median {gbp(-row['Median saving'])} extra",
                   delta_color="off", delta_arrow="off")
    if kw[0] == "continuous":
        windows = len(result["starts"])
        separate = len(d) // (years * 12)
        st.caption(f"{windows} start months from {span}. They overlap heavily: the data "
                   f"only holds {separate} fully separate {years}-year "
                   f"period{'s' if separate != 1 else ''}, so these percentages can look "
                   "more certain than they are.")
    else:
        st.caption(f"5,000 stitched histories drawn from {span}. Each repeats the same "
                   "history in a different order, so it can't show a different rate "
                   "environment.")

    st.subheader("Spread of savings")
    fig = go.Figure()
    for i, key in enumerate(history.VARIABLE_TYPES):
        fig.add_trace(go.Box(x=result["savings"][key], name=history.NAMES[key],
                             marker=dict(color=SERIES_COLOURS[i]), boxmean=True,
                             boxpoints=False, orientation="h"))
    fig.add_vline(x=0, line_color="#999999", line_dash="dot")
    fig.update_layout(height=300, xaxis_title="Interest saved compared with fixes (£)",
                      showlegend=False,
                      **BOX_LAYOUT)
    st.plotly_chart(fig, use_container_width=True)
    st.caption("Right of the dotted line, variable was cheaper. Box = middle 50% of "
               "outcomes. Line inside = median. Dashed line = mean.")

    if kw[0] == "continuous":
        st.subheader("Does the start date matter?")
        fig = go.Figure()
        for i, key in enumerate(history.VARIABLE_TYPES):
            fig.add_trace(go.Scatter(
                x=result["starts"], y=result["savings"][key], name=history.NAMES[key],
                mode="lines", line=dict(color=SERIES_COLOURS[i], width=2),
                hovertemplate=f"{history.NAMES[key]}: £%{{y:,.0f}}<extra></extra>"))
        fig.add_hline(y=0, line_color="#999999", line_dash="dot")
        fig.update_layout(height=340, xaxis_title="Month the mortgage started",
                          yaxis_title="Interest saved (£)", **CHART_LAYOUT)
        st.plotly_chart(fig, use_container_width=True)

    deals = history.single_deals(d)
    st.subheader("What drove the difference on single 2-year deals?")
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Variable cheaper", f"{100 * (deals['gap'] > 0).mean():.0f}% of deals")
    m2.metric("Average rate advantage", f"{deals['gap'].mean():+.2f}pp",
              help="Average rate on rolling 2-year fixes minus the variable deal.")
    m3.metric("Lower starting rate", f"{deals['premium'].mean():+.2f}pp",
              help="How much lower the variable deal's rate was on day one.")
    m4.metric("Without falling rates", f"{deals['gap_no_drift'].mean():+.2f}pp",
              help="The advantage after removing Bank Rate's overall fall across the data. "
                   f"Variable would win {100 * (deals['gap_no_drift'] > 0).mean():.0f}% of deals.")

    st.subheader("The rates behind this test")
    fig = go.Figure()
    for key, colour, dash in (("fix2", SERIES_COLOURS[3], None), ("var2", SERIES_COLOURS[0], None),
                              ("life", SERIES_COLOURS[1], None), ("svr", SERIES_COLOURS[2], None),
                              ("bank", "#777777", "dot")):
        name = history.NAMES.get(key, "Bank Rate")
        fig.add_trace(go.Scatter(
            x=d.index, y=d[key] * 100, name=name, mode="lines", connectgaps=False,
            line=dict(color=colour, width=2, dash=dash),
            hovertemplate=f"{name}: %{{y:.2f}}%<extra></extra>"))
    fig.update_layout(height=340, yaxis_title="Rate (%)", **CHART_LAYOUT)
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("All results")
    rows = [history.summary(cached_test(d, y, kw[0], kw[1], loan, term)).assign(Years=y)
            for y in choices]
    show = pd.concat(rows).drop(columns="key")
    csv = show.to_csv(index=False).encode()
    show = show[["Years"] + [c for c in show.columns if c != "Years"]]
    for c in ("Median saving", "10th %ile", "90th %ile", "Biggest 12-month rise",
              "Fix: biggest 12-month rise"):
        show[c] = show[c].map(gbp)
    show["Variable cheaper (%)"] = show["Variable cheaper (%)"].map("{:.0f}".format)
    show["Rate gap (pp)"] = show["Rate gap (pp)"].map("{:+.2f}".format)
    st.dataframe(show, use_container_width=True, hide_index=True)
    st.download_button("Download results (CSV)", csv,
                       file_name="variable_vs_fixed.csv", mime="text/csv")

    with st.expander("How to read this test"):
        st.markdown(
            f"""
- **Data:** Bank of England average quoted rates at 75% LTV, {span}. Variable deal data doesn't
  go back further, so a 30- or 40-year test isn't possible.
- **Rolling 2-year variable:** the margin over Bank Rate is locked for 2 years, then re-quoted.
  This is a stand-in: some older deals in this series may have been discounts off the lender's
  standard rate rather than true trackers.
- **Lifetime tracker:** the margin is locked for the whole period. Margins have varied a lot over
  time, so older start dates may reflect deals that aren't on offer today.
- **Biggest 12-month rise:** the largest increase in the monthly payment compared with a year
  earlier, as the median across paths. With segments this is overstated, because rates can
  leap where two pieces of history join. Use Continuous to judge payment shocks.
- **Not included:** product fees, early repayment charges and credit differences between borrowers.
- **Past rates don't predict future rates,** and most of this period had falling rates.
  **Not financial advice.**
            """
        )


# ------------------------------------------------------------------ the tab
@st.fragment
def render_mortgage_tab(a: StochasticAssumptions, ftb: bool, addl: bool, legal: float,
                        fixed_years: int, term: int, real: bool, live_df,
                        use_live: bool) -> None:
    """A fragment, so edits here don't re-run the Buy vs invest tab."""
    st.subheader("Your finances")
    fin = finances_panel(a.inflation)

    afford_tab, compare_tab, history_tab = st.tabs(
        ["What can we afford?", "Compare mortgages", "Variable or fixed?"])
    with afford_tab:
        affordability_section(fin, a.house_price,
                              stamp_duty(a.house_price, ftb, addl), legal,
                              a.mortgage_rate, live_df)
    with compare_tab:
        compare_section(a, fin, fixed_years, term, real, live_df, use_live)
        with st.expander("How the comparison works"):
            st.markdown(
                """
- **Same starting point:** every option uses your savings and the monthly budget from
  your finances. Your costs list already covers upkeep and council tax, so they aren't
  counted again here. Costs that start or stop change the budget in those years.
- **Interest-only:** you pay only interest, so the loan doesn't shrink. The full loan falls
  due at the end of the term and is paid from your investments, after capital gains tax.
- **Ran out of cash:** the budget and investments couldn't cover the mortgage on that path.
  The gap is treated as extra debt at the mortgage rate and paid off from later surpluses.
- **Chance best:** the share of simulated futures where that option ends with the most money.
- **Rates:** the live Bank of England rate for the loan's LTV band. It only publishes LTV bands
  for 2- and 5-year fixes, and nothing above 90% LTV, so a 5% deposit may look too cheap.
- **Not modelled:** lender rules. UK lenders often need a large deposit and a repayment plan
  for interest-only, and affordability checks can rule out some options. No remortgage fees.
- **Not financial advice.** This is a tool for exploring scenarios.
                """
            )
    with history_tab:
        variable_vs_fixed(a.house_price)
    settings.refresh_snapshot()
