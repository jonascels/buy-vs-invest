"""
Streamlit interface for the Buy vs. Invest model.

Run with:  streamlit run app.py
"""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from model import Assumptions, run_model, stamp_duty, summarise

st.set_page_config(page_title="Buy vs Invest", page_icon="🏠", layout="wide")

# ----- light theming -----
st.markdown(
    """
    <style>
      .stApp { background: #0e1117; }
      .metric-good { color: #2dd4bf; }
      h1, h2, h3 { letter-spacing: -0.02em; }
      div[data-testid="stMetricValue"] { font-size: 1.4rem; }
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("🏠  Buy a home vs. keep investing")
st.caption(
    "Compares terminal net worth of buying with a mortgage against renting and "
    "investing the deposit plus any monthly cash-flow difference. All figures GBP."
)

# ------------------------------------------------------------------ sidebar
sb = st.sidebar
sb.header("Assumptions")

with sb.expander("🏡  Property", expanded=True):
    house_price = st.slider("House price (£)", 150_000, 2_500_000, 600_000, 10_000)
    deposit_pct = st.slider("Deposit (%)", 5, 60, 20, 1) / 100
    mortgage_rate = st.slider("Mortgage rate (%)", 0.5, 9.0, 4.5, 0.1) / 100
    term = st.slider("Mortgage term (yrs)", 10, 40, 30, 1)
    appreciation = st.slider("House appreciation (%/yr)", -2.0, 10.0, 3.0, 0.25) / 100
    ftb = st.checkbox("First-time buyer (SDLT relief)", value=False)
    addl = st.checkbox("Additional property (+5% SDLT)", value=False)

with sb.expander("🔧  Running & transaction costs"):
    maintenance = st.slider("Maintenance (%/yr of value)", 0.0, 3.0, 1.0, 0.1) / 100
    insurance = st.slider("Insurance (%/yr of value)", 0.0, 1.0, 0.15, 0.05) / 100
    service_charge = st.number_input("Service charge / ground rent (£/yr)", 0, 20_000, 0, 250)
    council_tax = st.number_input("Council tax (£/yr)", 0, 8_000, 2_200, 100)
    legal = st.number_input("Legal + survey (£, one-off)", 0, 15_000, 2_500, 250)
    selling_cost = st.slider("Selling costs (% of value)", 0.0, 4.0, 1.75, 0.25) / 100

with sb.expander("🏘️  Rent (the alternative)"):
    rent = st.slider("Monthly rent (£)", 500, 8_000, 2_200, 50)
    rent_growth = st.slider("Rent growth (%/yr)", 0.0, 8.0, 3.0, 0.25) / 100

with sb.expander("📈  Investment", expanded=True):
    inv_return = st.slider("Investment return (%/yr)", 0.0, 12.0, 7.0, 0.25) / 100
    inv_fee = st.slider("Fund fees / OCF (%/yr)", 0.0, 1.5, 0.16, 0.01) / 100
    cgt = st.slider("Capital gains tax on exit (%)", 0, 40, 20, 1) / 100

with sb.expander("🌍  Economy"):
    inflation = st.slider("Inflation (%/yr)", 0.0, 8.0, 2.5, 0.25) / 100

real_terms = sb.toggle("Show in real (today's £) terms", value=False)

# ------------------------------------------------------------------ run
a = Assumptions(
    house_price=house_price, deposit_pct=deposit_pct, mortgage_rate=mortgage_rate,
    mortgage_term_years=term, house_appreciation=appreciation,
    first_time_buyer=ftb, additional_property=addl,
    maintenance_pct=maintenance, insurance_pct=insurance,
    service_charge=service_charge, council_tax=council_tax,
    legal_survey_fixed=legal, selling_cost_pct=selling_cost,
    monthly_rent=rent, rent_growth=rent_growth,
    investment_return=inv_return, investment_fee=inv_fee, capital_gains_tax=cgt,
    inflation=inflation,
)

df = run_model(a)
summary = summarise(df, a)

buy_col = "buy_net_worth_real" if real_terms else "buy_net_worth"
inv_col = "invest_net_worth_real" if real_terms else "invest_net_worth"
suffix = " (real)" if real_terms else ""


def gbp(x: float) -> str:
    return f"£{x:,.0f}"


# ------------------------------------------------------------------ top cards
deposit = df.attrs["deposit"]
c1, c2, c3, c4 = st.columns(4)
c1.metric("Deposit", gbp(deposit))
c2.metric("Stamp duty (SDLT)", gbp(df.attrs["sdlt"]))
c3.metric("Loan", gbp(df.attrs["loan"]))
c4.metric("Cash needed upfront", gbp(df.attrs["upfront_purchase_costs"]))

st.divider()

# ------------------------------------------------------------------ horizon cards
st.subheader("Verdict by horizon")
cols = st.columns(len(a.horizons))
for col, (_, r) in zip(cols, summary.iterrows()):
    buy = r["Buy (real)" if real_terms else "Buy net worth"]
    inv = r["Invest (real)" if real_terms else "Invest net worth"]
    winner = "Buy" if buy >= inv else "Invest"
    gap = abs(buy - inv)
    col.markdown(f"**{int(r['Horizon (yrs)'])} years**")
    col.metric(
        f"🏆 {winner}",
        gbp(max(buy, inv)),
        delta=f"+{gbp(gap)} vs other",
    )
    col.caption(f"Buy {gbp(buy)} · Invest {gbp(inv)}")

st.divider()

# ------------------------------------------------------------------ main chart
st.subheader(f"Net worth over time{suffix}")
fig = go.Figure()
fig.add_trace(go.Scatter(
    x=df["year"], y=df[buy_col], name="Buy", mode="lines",
    line=dict(color="#2dd4bf", width=3),
    hovertemplate="Yr %{x:.1f}<br>Buy £%{y:,.0f}<extra></extra>"))
fig.add_trace(go.Scatter(
    x=df["year"], y=df[inv_col], name="Invest", mode="lines",
    line=dict(color="#f59e0b", width=3),
    hovertemplate="Yr %{x:.1f}<br>Invest £%{y:,.0f}<extra></extra>"))
# shade who's ahead
fig.add_trace(go.Scatter(
    x=df["year"], y=np.where(df[buy_col] >= df[inv_col], df[buy_col], df[inv_col]),
    fill=None, mode="lines", line=dict(width=0), showlegend=False,
    hoverinfo="skip"))
for h in a.horizons:
    fig.add_vline(x=h, line_dash="dot", line_color="#475569", opacity=0.5)
fig.update_layout(
    template="plotly_dark", height=460, hovermode="x unified",
    margin=dict(l=10, r=10, t=10, b=10),
    legend=dict(orientation="h", y=1.05),
    yaxis_title="Net worth (£)", xaxis_title="Years",
    paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
st.plotly_chart(fig, use_container_width=True)

# ------------------------------------------------------------------ components
left, right = st.columns(2)

with left:
    st.subheader("Buy: equity vs debt")
    f2 = go.Figure()
    f2.add_trace(go.Scatter(x=df["year"], y=df["house_value"], name="House value",
                            line=dict(color="#60a5fa")))
    f2.add_trace(go.Scatter(x=df["year"], y=df["home_equity"], name="Your equity",
                            fill="tozeroy", line=dict(color="#2dd4bf")))
    f2.add_trace(go.Scatter(x=df["year"], y=df["loan_balance"], name="Mortgage left",
                            line=dict(color="#ef4444", dash="dot")))
    f2.update_layout(template="plotly_dark", height=320,
                     margin=dict(l=10, r=10, t=10, b=10),
                     legend=dict(orientation="h", y=1.1),
                     paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
    st.plotly_chart(f2, use_container_width=True)

with right:
    st.subheader("Monthly outgoings: buy vs rent")
    f3 = go.Figure()
    f3.add_trace(go.Scatter(x=df["year"], y=df["buy_outgoing"], name="Buy /mo",
                            line=dict(color="#2dd4bf")))
    f3.add_trace(go.Scatter(x=df["year"], y=df["rent_outgoing"], name="Rent /mo",
                            line=dict(color="#f59e0b")))
    f3.update_layout(template="plotly_dark", height=320,
                     margin=dict(l=10, r=10, t=10, b=10),
                     legend=dict(orientation="h", y=1.1),
                     paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
    st.plotly_chart(f3, use_container_width=True)

# ------------------------------------------------------------------ table + export
st.subheader("Summary table")
show = summary.copy()
money_cols = ["Buy net worth", "Invest net worth", "Difference (Buy − Invest)",
              "Buy (real)", "Invest (real)"]
for c in money_cols:
    show[c] = show[c].map(gbp)
st.dataframe(show, use_container_width=True, hide_index=True)

st.download_button(
    "⬇️  Download full monthly projection (CSV)",
    df.to_csv(index=False).encode(),
    file_name="buy_vs_invest_projection.csv", mime="text/csv")

with st.expander("ℹ️  How this works & caveats"):
    st.markdown(
        """
- **Buy net worth** = home value − remaining mortgage − selling costs, net of CGT
  on any buyer-side invested surplus.
- **Invest net worth** = the renter invests the buyer's full upfront cash (deposit
  + SDLT + legal) and every month invests the difference when buying costs more
  than renting; shown net of CGT on gains at exit.
- Both renter and owner pay council tax. Rent and house prices grow at their own
  rates; investment return is net of fund fees.
- **Not advice.** A single deterministic path hides sequence-of-returns risk —
  try low/median/high return scenarios. SDLT bands are England & N. Ireland
  (Apr 2025); Scotland (LBTT) and Wales (LTT) differ.
        """
    )
