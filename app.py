"""
Streamlit interface for the Buy vs. Invest model, with deterministic + MC modes.

Run with:  streamlit run app.py
"""

import time

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

import rates as boe
from lab import render_lab
from model import (Assumptions, StochasticAssumptions, monthly_mortgage_payment,
                   run_model, run_monte_carlo, stamp_duty, summarise)

st.set_page_config(page_title="Buy vs Invest", page_icon="🏠", layout="wide")

# Light theme - configured via .streamlit/config.toml
st.markdown(
    """
    <style>
      /* Minimal CSS - most theming done via config.toml */
      h1, h2, h3, h4, h5, h6 {
        letter-spacing: -0.02em;
      }
      
      /* Sidebar */
      [data-testid="stSidebar"] {
        background-color: #f5f5f5;
      }
      
      /* Plotly charts background */
      .plotly-graph-div {
        background-color: white !important;
      }
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("🏠 Buy a home vs. keep investing")
st.caption(
    "Compares terminal net worth of buying with a mortgage against renting and "
    "investing the deposit plus any monthly cash-flow difference. All figures GBP."
)



@st.cache_data(ttl=6 * 3600, show_spinner="Fetching live mortgage rates…")
def load_live_rates():
    """Bank of England monthly mortgage rates, or None if unreachable."""
    try:
        return boe.fetch_boe_rates()
    except Exception:
        return None


live_df = load_live_rates()

# ------------------------------------------------------------------ sidebar
sb = st.sidebar
sb.header("Assumptions")

# Toggle between deterministic and Monte Carlo
mode = sb.radio("📊 Analysis mode", ("Deterministic", "Monte Carlo"), horizontal=False,
                help="Applies to the Buy vs invest tab. The Mortgage lab always uses Monte Carlo.")

with sb.expander("🏡 Property", expanded=True):
    house_price = st.slider("House price (£)", 150_000, 2_500_000, 600_000, 10_000)
    deposit_pct = st.slider("Deposit (%)", 5, 100, 20, 1,
                            help="Set to 100% for a cash purchase with no mortgage") / 100
    cash_purchase = deposit_pct >= 1.0
    if cash_purchase:
        st.caption("💷 Cash purchase: no mortgage, so rate and term are ignored.")
    fixed_years = st.selectbox(
        "Fixed-rate period", boe.FIXED_PERIODS, index=2,
        format_func=lambda y: f"{y}-year fix", disabled=cash_purchase,
        help="How long the rate is locked. In Monte Carlo mode you remortgage "
             "at the market rate at the end of each fixed period.")
    ltv_pct = (1 - deposit_pct) * 100
    live_code = boe.best_series(fixed_years, ltv_pct)
    live = boe.latest_rate(live_df, live_code) if live_df is not None else None
    use_live = st.toggle("Use live Bank of England rate", value=live is not None,
                         disabled=cash_purchase or live is None)
    if use_live and live is not None:
        live_rate, live_month = live
        st.caption(f"📡 {boe.SERIES_BY_CODE[live_code].label}: "
                   f"**{live_rate * 100:.2f}%** ({live_month:%b %Y} average, "
                   f"excludes product fees)")
        default_rate = min(max(round(live_rate * 100, 2), 0.5), 9.0)
    else:
        if live is None and not cash_purchase:
            st.caption("Live rates unavailable right now, so set the rate manually.")
        default_rate = 4.5
    mortgage_rate = st.slider("Mortgage rate (%)", 0.5, 9.0, default_rate, 0.01,
                              disabled=cash_purchase or use_live) / 100
    term = st.slider("Mortgage term (yrs)", 10, 40, 30, 1, disabled=cash_purchase)
    appreciation = st.slider("House appreciation (%/yr)", -2.0, 10.0, 3.0, 0.25) / 100
    ftb = st.checkbox("First-time buyer (SDLT relief)", value=False)
    addl = st.checkbox("Additional property (+5% SDLT)", value=False)

with sb.expander("🔧 Running & transaction costs"):
    maintenance = st.slider("Maintenance (%/yr of value)", 0.0, 3.0, 1.0, 0.1) / 100
    insurance = st.slider("Insurance (%/yr of value)", 0.0, 1.0, 0.15, 0.05) / 100
    service_charge = st.number_input("Service charge / ground rent (£/yr)", 0, 20_000, 0, 250)
    council_tax = st.number_input("Council tax (£/yr)", 0, 8_000, 2_200, 100)
    legal = st.number_input("Legal + survey (£, one-off)", 0, 15_000, 2_500, 250)
    selling_cost = st.slider("Selling costs (% of value)", 0.0, 4.0, 1.75, 0.25) / 100

with sb.expander("🏘️ Rent (the alternative)"):
    rent = st.slider("Monthly rent (£)", 500, 8_000, 2_200, 50)
    rent_growth = st.slider("Rent growth (%/yr)", 0.0, 8.0, 3.0, 0.25) / 100

with sb.expander("📈 Investment", expanded=True):
    inv_return = st.slider("Investment return (%/yr)", 0.0, 12.0, 7.0, 0.25) / 100
    inv_fee = st.slider("Fund fees / OCF (%/yr)", 0.0, 1.5, 0.16, 0.01) / 100
    cgt = st.slider("Capital gains tax on exit (%)", 0, 40, 20, 1) / 100

with sb.expander("🌍 Economy"):
    inflation = st.slider("Inflation (%/yr)", 0.0, 8.0, 2.5, 0.25) / 100

real_terms = sb.toggle("📌 Show in real (today's £) terms", value=False)

# Monte Carlo options
if mode == "Monte Carlo":
    sb.divider()
    sb.subheader("🎲 Monte Carlo options")
    n_sims = sb.slider("Number of simulations", 10, 1_000, 100, 10,
                       help="More sims = more accurate but slower")
    sb.write(f"⏱️ ~{max(n_sims * 0.012, 0.1):.1f}s estimated runtime")

sb.divider()
sb.subheader("🎲 Simulation risk")
sb.caption("Used by Monte Carlo mode and the Mortgage lab.")
with sb.expander("📊 Volatility assumptions", expanded=False):
    inv_vol = st.slider("Investment volatility (%/yr)", 1.0, 30.0, 12.0, 0.5) / 100
    house_vol = st.slider("House price volatility (%/yr)", 1.0, 20.0, 8.0, 0.5) / 100
    rent_vol = st.slider("Rent growth volatility (%/yr)", 0.0, 5.0, 2.0, 0.1) / 100

with sb.expander("🏦 Mortgage rate risk", expanded=False):
    hist_vol = boe.annual_change_std(live_df, boe.HEADLINE[fixed_years]) \
        if live_df is not None else None
    rate_vol = st.slider(
        "Mortgage rate volatility (pp/yr)", 0.0, 3.0,
        round(hist_vol * 100, 1) if hist_vol else 1.0, 0.1,
        disabled=cash_purchase,
        help="Typical size of a one-year move in market mortgage rates. "
             "Default comes from the last 15 years of Bank of England data."
    ) / 100
    long_run_rate = st.slider(
        "Long-run mortgage rate (%)", 1.0, 9.0,
        min(max(round(mortgage_rate * 100, 1), 1.0), 9.0), 0.1,
        disabled=cash_purchase,
        help="Rates drift back towards this level over time. "
             "Defaults to today's rate (no expected rise or fall).") / 100
    if hist_vol and not cash_purchase:
        st.caption(f"Historical 1-year moves: ±{hist_vol * 100:.1f}pp "
                   f"(1 std dev, {fixed_years}-year fix).")

tab_main, tab_lab = st.tabs(["🏠 Buy vs invest", "🔬 Mortgage lab"])

with tab_main:
    # ------------------------------------------------------------------ live rates
    FIX_COLOURS = {2: "#8e24aa", 3: "#00897b", 5: "#3949ab", 10: "#6d4c41"}

    with st.expander("📡 Live UK mortgage rates: compare fixed periods",
                     expanded=False):
        if live_df is None:
            st.warning("Couldn't reach the Bank of England database just now. "
                       "The model still works with a manual rate.")
        else:
            loan_amt = house_price * (1 - deposit_pct)
            st.caption(
                f"Average quoted rates from the Bank of England, 75% LTV "
                f"(your loan is {ltv_pct:.0f}% LTV). Payments are for your "
                f"£{loan_amt:,.0f} loan over {term} years. Rates exclude product fees, "
                f"so the best deals on the market can be lower.")
            cols = st.columns(len(boe.FIXED_PERIODS))
            rows = []
            latest = boe.latest_table(live_df).set_index("code")
            for col, fy in zip(cols, boe.FIXED_PERIODS):
                code = boe.HEADLINE[fy]
                if code not in latest.index:
                    continue
                r = latest.loc[code]
                rate = r["Rate (%)"] / 100
                pmt = monthly_mortgage_payment(loan_amt, rate, term)
                change = r["12-month change (pp)"]
                col.metric(f"{fy}-year fix", f"{rate * 100:.2f}%",
                           delta=None if pd.isna(change) else f"{change:+.2f}pp vs last year",
                           delta_color="inverse")
                col.caption(f"£{pmt:,.0f}/month · {r['As of']}")
                rows.append({
                    "Fixed period": f"{fy} years",
                    "Rate (%)": f"{rate * 100:.2f}",
                    "Monthly payment": f"£{pmt:,.0f}",
                    "Paid during the fix": f"£{pmt * 12 * fy:,.0f}",
                    "Certainty": f"Payment locked for {fy} years",
                })
            st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

            hist = live_df[live_df.index >= live_df.index[-1] - pd.DateOffset(years=10)]
            fig_live = go.Figure()
            for fy in boe.FIXED_PERIODS:
                code = boe.HEADLINE[fy]
                if code in hist:
                    fig_live.add_trace(go.Scatter(
                        x=hist.index, y=hist[code], name=f"{fy}-year fix",
                        mode="lines", line=dict(color=FIX_COLOURS[fy], width=2),
                        hovertemplate=f"{fy}-yr fix: %{{y:.2f}}%<extra></extra>"))
            if "IUMTLMV" in hist:
                fig_live.add_trace(go.Scatter(
                    x=hist.index, y=hist["IUMTLMV"], name="Standard variable rate",
                    mode="lines", line=dict(color="#9e9e9e", width=2, dash="dot"),
                    hovertemplate="SVR: %{y:.2f}%<extra></extra>"))
            fig_live.update_layout(
                template="plotly_white", height=320, hovermode="x unified",
                margin=dict(l=10, r=10, t=10, b=10),
                legend=dict(orientation="h", y=1.1), yaxis_title="Rate (%)",
                font=dict(size=11, color="#333333"),
                paper_bgcolor="white", plot_bgcolor="#fafafa")
            st.plotly_chart(fig_live, use_container_width=True)

            with st.popover("All published rates (by LTV)"):
                show_lt = latest.reset_index(drop=True).copy()
                show_lt["Rate (%)"] = show_lt["Rate (%)"].map("{:.2f}".format)
                show_lt["12-month change (pp)"] = show_lt["12-month change (pp)"].map(
                    lambda x: "" if pd.isna(x) else f"{x:+.2f}")
                st.dataframe(show_lt, hide_index=True)
            st.caption(f"Source: [Bank of England Database]({boe.SOURCE_PAGE}) · "
                       "refreshed every 6 hours.")

    # ------------------------------------------------------------------ run
    if mode == "Deterministic":
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
    
        # ---- top cards
        deposit = df.attrs["deposit"]
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Deposit", gbp(deposit))
        c2.metric("Stamp duty (SDLT)", gbp(df.attrs["sdlt"]))
        c3.metric("Loan", gbp(df.attrs["loan"]))
        c4.metric("Cash needed upfront", gbp(df.attrs["upfront_purchase_costs"]))
    
        st.divider()
    
        # ---- horizon cards
        st.subheader("Verdict by horizon")
        cols = st.columns(len(a.horizons))
        for col, (_, r) in zip(cols, summary.iterrows()):
            buy = r["Buy (real)" if real_terms else "Buy net worth"]
            inv = r["Invest (real)" if real_terms else "Invest net worth"]
            winner = "🏠 Buy" if buy >= inv else "📈 Invest"
            gap = abs(buy - inv)
            col.markdown(f"**{int(r['Horizon (yrs)'])} years**")
            col.metric(
                f"🏆 {winner}",
                gbp(max(buy, inv)),
                delta=f"+{gbp(gap)} vs other",
            )
            col.caption(f"Buy {gbp(buy)} · Invest {gbp(inv)}")
    
        st.divider()
    
        # ---- main chart
        st.subheader(f"Net worth over time{suffix}")
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=df["year"], y=df[buy_col], name="Buy", mode="lines",
            line=dict(color="#1e88e5", width=3),
            hovertemplate="Yr %{x:.1f}<br>Buy £%{y:,.0f}<extra></extra>"))
        fig.add_trace(go.Scatter(
            x=df["year"], y=df[inv_col], name="Invest", mode="lines",
            line=dict(color="#f57c00", width=3),
            hovertemplate="Yr %{x:.1f}<br>Invest £%{y:,.0f}<extra></extra>"))
        for h in a.horizons:
            fig.add_vline(x=h, line_dash="dot", line_color="#999999", opacity=0.5)
        fig.update_layout(
            template="plotly_white", height=460, hovermode="x unified",
            margin=dict(l=10, r=10, t=10, b=10),
            legend=dict(orientation="h", y=1.05),
            yaxis_title="Net worth (£)", xaxis_title="Years",
            font=dict(size=12, color="#333333"),
            paper_bgcolor="white", plot_bgcolor="#fafafa")
        st.plotly_chart(fig, use_container_width=True)
    
        # ---- components
        left, right = st.columns(2)
    
        with left:
            st.subheader("Buy: equity vs debt")
            f2 = go.Figure()
            f2.add_trace(go.Scatter(x=df["year"], y=df["house_value"], name="House value",
                                    line=dict(color="#4caf50")))
            f2.add_trace(go.Scatter(x=df["year"], y=df["home_equity"], name="Your equity",
                                    fill="tozeroy", line=dict(color="#1e88e5")))
            f2.add_trace(go.Scatter(x=df["year"], y=df["loan_balance"], name="Mortgage left",
                                    line=dict(color="#d32f2f", dash="dot")))
            f2.update_layout(template="plotly_white", height=320,
                             margin=dict(l=10, r=10, t=10, b=10),
                             legend=dict(orientation="h", y=1.1),
                             font=dict(size=11, color="#333333"),
                             paper_bgcolor="white", plot_bgcolor="#fafafa")
            st.plotly_chart(f2, use_container_width=True)
    
        with right:
            st.subheader("Monthly outgoings: buy vs rent")
            f3 = go.Figure()
            f3.add_trace(go.Scatter(x=df["year"], y=df["buy_outgoing"], name="Buy /mo",
                                    line=dict(color="#1e88e5")))
            f3.add_trace(go.Scatter(x=df["year"], y=df["rent_outgoing"], name="Rent /mo",
                                    line=dict(color="#f57c00")))
            f3.update_layout(template="plotly_white", height=320,
                             margin=dict(l=10, r=10, t=10, b=10),
                             legend=dict(orientation="h", y=1.1),
                             font=dict(size=11, color="#333333"),
                             paper_bgcolor="white", plot_bgcolor="#fafafa")
            st.plotly_chart(f3, use_container_width=True)
    
        # ---- table
        st.subheader("Summary table")
        show = summary.copy()
        money_cols = ["Buy net worth", "Invest net worth", "Difference (Buy − Invest)",
                      "Buy (real)", "Invest (real)"]
        for c in money_cols:
            show[c] = show[c].map(gbp)
        st.dataframe(show, use_container_width=True, hide_index=True)
    
        st.download_button(
            "⬇️ Download full monthly projection (CSV)",
            df.to_csv(index=False).encode(),
            file_name="buy_vs_invest_deterministic.csv", mime="text/csv")


    else:  # Monte Carlo mode
        a = StochasticAssumptions(
            house_price=house_price, deposit_pct=deposit_pct, mortgage_rate=mortgage_rate,
            mortgage_term_years=term, house_appreciation=appreciation,
            house_appreciation_std=house_vol,
            first_time_buyer=ftb, additional_property=addl,
            maintenance_pct=maintenance, insurance_pct=insurance,
            service_charge=service_charge, council_tax=council_tax,
            legal_survey_fixed=legal, selling_cost_pct=selling_cost,
            monthly_rent=rent, rent_growth=rent_growth, rent_growth_std=rent_vol,
            investment_return=inv_return, investment_fee=inv_fee,
            investment_return_std=inv_vol,
            mortgage_rate_std=rate_vol, fixed_period_years=fixed_years,
            mortgage_rate_long_run=long_run_rate,
            capital_gains_tax=cgt,
            inflation=inflation,
        )
    
        # Run MC with progress indicator
        with st.spinner(f"Running {n_sims} Monte Carlo simulations..."):
            t0 = time.time()
            pct_df, win_df, buy_out, inv_out = run_monte_carlo(a, n_sims=n_sims)
            t1 = time.time()
    
        st.success(f"✓ Completed in {t1-t0:.2f}s")
    
        def gbp(x: float) -> str:
            return f"£{x:,.0f}"
    
        deposit = a.house_price * a.deposit_pct
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Deposit", gbp(deposit))
        c2.metric("Stamp duty (SDLT)", gbp(stamp_duty(a.house_price, a.first_time_buyer, a.additional_property)))
        c3.metric("Loan", gbp(a.house_price - deposit))
        c4.metric("Sims run", f"{n_sims}")
    
        st.divider()
    
        # Win rates
        st.subheader("Win probability by horizon")
        cols = st.columns(len(a.horizons))
        for col, (_, r) in zip(cols, win_df.iterrows()):
            h = int(r["Horizon (yrs)"])
            buy_pct = r["Buy wins (%)"]
            inv_pct = r["Invest wins (%)"]
            winner = "🏠 Buy" if buy_pct >= 50 else "📈 Invest"
            col.markdown(f"**{h} years**")
            col.metric(f"{winner}", f"{max(buy_pct, inv_pct):.0f}%")
            col.caption(f"Buy {buy_pct:.0f}% · Invest {inv_pct:.0f}%")
    
        st.divider()
    
        # Percentile distribution with filled uncertainty ranges
        st.subheader("Net worth distribution with uncertainty ranges")
    
        fig_pct = go.Figure()
    
        # Buy outcomes with filled uncertainty range (10th-90th percentile)
        buy_90 = pct_df["Buy 90th %ile"].values
        buy_10 = pct_df["Buy 10th %ile"].values
        buy_med = pct_df["Buy median"].values
        horizons = pct_df["Horizon (yrs)"].values
    
        # Add filled area between 10th and 90th percentile for Buy
        fig_pct.add_trace(go.Scatter(
            x=horizons, y=buy_90, 
            name="Buy (90th %ile)",
            line=dict(color="rgba(30, 136, 229, 0)", width=0),
            hovertemplate="Yr %{x}<br>Buy 90th: £%{y:,.0f}<extra></extra>"
        ))
        fig_pct.add_trace(go.Scatter(
            x=horizons, y=buy_10,
            name="Buy (10th %ile)",
            line=dict(color="rgba(30, 136, 229, 0)", width=0),
            fillcolor="rgba(30, 136, 229, 0.2)",
            fill="tonexty",
            hovertemplate="Yr %{x}<br>Buy 10th: £%{y:,.0f}<extra></extra>"
        ))
    
        # Buy median line
        fig_pct.add_trace(go.Scatter(
            x=horizons, y=buy_med,
            name="Buy (median)",
            line=dict(color="#1e88e5", width=3),
            hovertemplate="Yr %{x}<br>Buy median: £%{y:,.0f}<extra></extra>"
        ))
    
        # Invest outcomes with filled uncertainty range
        inv_90 = pct_df["Invest 90th %ile"].values
        inv_10 = pct_df["Invest 10th %ile"].values
        inv_med = pct_df["Invest median"].values
    
        # Add filled area between 10th and 90th percentile for Invest
        fig_pct.add_trace(go.Scatter(
            x=horizons, y=inv_90,
            name="Invest (90th %ile)",
            line=dict(color="rgba(245, 124, 0, 0)", width=0),
            hovertemplate="Yr %{x}<br>Invest 90th: £%{y:,.0f}<extra></extra>"
        ))
        fig_pct.add_trace(go.Scatter(
            x=horizons, y=inv_10,
            name="Invest (10th %ile)",
            line=dict(color="rgba(245, 124, 0, 0)", width=0),
            fillcolor="rgba(245, 124, 0, 0.2)",
            fill="tonexty",
            hovertemplate="Yr %{x}<br>Invest 10th: £%{y:,.0f}<extra></extra>"
        ))
    
        # Invest median line
        fig_pct.add_trace(go.Scatter(
            x=horizons, y=inv_med,
            name="Invest (median)",
            line=dict(color="#f57c00", width=3),
            hovertemplate="Yr %{x}<br>Invest median: £%{y:,.0f}<extra></extra>"
        ))
    
        fig_pct.update_layout(
            template="plotly_white", height=480, hovermode="x unified",
            margin=dict(l=10, r=10, t=10, b=10),
            legend=dict(orientation="v", x=1.02, y=1, bgcolor="rgba(255,255,255,0.8)"),
            xaxis_title="Years", yaxis_title="Net worth (£)",
            font=dict(size=12, color="#333333"),
            paper_bgcolor="white", plot_bgcolor="#fafafa")
        st.plotly_chart(fig_pct, use_container_width=True)
    
        st.caption("💡 Blue shaded area = range of Buy outcomes (10th–90th percentile). Orange shaded area = range of Invest outcomes. Bold lines = median (50th percentile).")
    
        st.divider()
    
        # Box and whisker plot
        st.subheader("Distribution comparison at each horizon")
        fig_fan = go.Figure()
    
        for h in a.horizons:
            buy_vals = np.array(buy_out[h]) / 1e6  # convert to millions for readability
            inv_vals = np.array(inv_out[h]) / 1e6
        
            fig_fan.add_trace(go.Box(
                y=buy_vals, name=f"Buy {h}y", marker=dict(color="#1e88e5"),
                boxmean='sd'))
            fig_fan.add_trace(go.Box(
                y=inv_vals, name=f"Invest {h}y", marker=dict(color="#f57c00"),
                boxmean='sd'))
    
        fig_fan.update_layout(
            template="plotly_white", height=420,
            margin=dict(l=10, r=10, t=10, b=10),
            yaxis_title="Net worth (£ millions)", boxmode="group",
            font=dict(size=12, color="#333333"),
            paper_bgcolor="white", plot_bgcolor="#fafafa")
        st.plotly_chart(fig_fan, use_container_width=True)
    
        st.caption("📊 Box = 25th to 75th percentile (interquartile range). Whisker = full range. Line inside box = median. Cross = mean.")

        if not cash_purchase:
            st.divider()
            st.subheader("Mortgage rate risk")
            rb = pct_df.attrs["rates"]
            left, right = st.columns(2)
            with left:
                f_rate = go.Figure()
                f_rate.add_trace(go.Scatter(
                    x=rb["Year"], y=rb["Paid rate p90"] * 100, name="90th %ile",
                    line=dict(width=0, shape="hv"), showlegend=False,
                    hovertemplate="Yr %{x}<br>90th: %{y:.2f}%<extra></extra>"))
                f_rate.add_trace(go.Scatter(
                    x=rb["Year"], y=rb["Paid rate p10"] * 100, name="10th–90th %ile",
                    line=dict(width=0, shape="hv"), fill="tonexty",
                    fillcolor="rgba(30, 136, 229, 0.2)",
                    hovertemplate="Yr %{x}<br>10th: %{y:.2f}%<extra></extra>"))
                f_rate.add_trace(go.Scatter(
                    x=rb["Year"], y=rb["Paid rate p50"] * 100, name="Median",
                    line=dict(color="#1e88e5", width=3, shape="hv"),
                    hovertemplate="Yr %{x}<br>Median: %{y:.2f}%<extra></extra>"))
                f_rate.update_layout(
                    template="plotly_white", height=320, hovermode="x unified",
                    title=dict(text="Rate you pay", font=dict(size=13)),
                    margin=dict(l=10, r=10, t=40, b=10),
                    legend=dict(orientation="h", y=-0.2),
                    xaxis_title="Years", yaxis_title="Rate (%)",
                    font=dict(size=11, color="#333333"),
                    paper_bgcolor="white", plot_bgcolor="#fafafa")
                st.plotly_chart(f_rate, use_container_width=True)
            with right:
                f_pmt = go.Figure()
                f_pmt.add_trace(go.Scatter(
                    x=rb["Year"], y=rb["Payment p90"], name="90th %ile",
                    line=dict(width=0, shape="hv"), showlegend=False,
                    hovertemplate="Yr %{x}<br>90th: £%{y:,.0f}<extra></extra>"))
                f_pmt.add_trace(go.Scatter(
                    x=rb["Year"], y=rb["Payment p10"], name="10th–90th %ile",
                    line=dict(width=0, shape="hv"), fill="tonexty",
                    fillcolor="rgba(30, 136, 229, 0.2)",
                    hovertemplate="Yr %{x}<br>10th: £%{y:,.0f}<extra></extra>"))
                f_pmt.add_trace(go.Scatter(
                    x=rb["Year"], y=rb["Payment p50"], name="Median",
                    line=dict(color="#1e88e5", width=3, shape="hv"),
                    hovertemplate="Yr %{x}<br>Median: £%{y:,.0f}<extra></extra>"))
                f_pmt.update_layout(
                    template="plotly_white", height=320, hovermode="x unified",
                    title=dict(text="Monthly mortgage payment", font=dict(size=13)),
                    margin=dict(l=10, r=10, t=40, b=10),
                    legend=dict(orientation="h", y=-0.2),
                    xaxis_title="Years", yaxis_title="£ per month",
                    font=dict(size=11, color="#333333"),
                    paper_bgcolor="white", plot_bgcolor="#fafafa")
                st.plotly_chart(f_pmt, use_container_width=True)
            st.caption(f"🏦 Your rate is locked for {fixed_years} years at a time, then "
                       "you remortgage at whatever the market rate is. The steps show "
                       "each remortgage. Shaded = 80% of simulated paths.")
    
        # Percentile table
        st.divider()
        st.subheader("Percentile summary table")
        show_pct = pct_df.copy()
        show_pct.attrs = {}
        for col in show_pct.columns:
            if col != "Horizon (yrs)":
                show_pct[col] = show_pct[col].map(gbp)
        st.dataframe(show_pct, use_container_width=True, hide_index=True)
    
        st.subheader("Win rates summary")
        show_win = win_df.copy()
        for col in show_win.columns:
            if col != "Horizon (yrs)":
                show_win[col] = show_win[col].apply(lambda x: f"{x:.1f}%")
        st.dataframe(show_win, use_container_width=True, hide_index=True)
    
        # Export options
        csv_pct = pct_df.to_csv(index=False)
        csv_win = win_df.to_csv(index=False)
        combined = f"# Percentiles\n{csv_pct}\n\n# Win rates\n{csv_win}"
        st.download_button(
            "⬇️ Download MC summary (CSV)",
            combined.encode(),
            file_name="buy_vs_invest_monte_carlo.csv", mime="text/csv")

# ------------------------------------------------------------------ mortgage lab
with tab_lab:
    render_lab(
        StochasticAssumptions(
            house_price=house_price, deposit_pct=deposit_pct, mortgage_rate=mortgage_rate,
            mortgage_term_years=term, house_appreciation=appreciation,
            house_appreciation_std=house_vol,
            first_time_buyer=ftb, additional_property=addl,
            maintenance_pct=maintenance, insurance_pct=insurance,
            service_charge=service_charge, council_tax=council_tax,
            legal_survey_fixed=legal, selling_cost_pct=selling_cost,
            investment_return=inv_return, investment_fee=inv_fee,
            investment_return_std=inv_vol,
            mortgage_rate_std=rate_vol, fixed_period_years=fixed_years,
            mortgage_rate_long_run=long_run_rate,
            capital_gains_tax=cgt, inflation=inflation,
        ),
        fixed_years, term, real_terms, live_df, use_live,
    )

# ---- footer / caveats
st.divider()
with st.expander("ℹ️ How this works & caveats"):
    st.markdown(
        """
**Deterministic mode:**
- Single fixed path: house grows at 3%/yr, returns at 7%/yr, etc.
- Buy net worth = home equity − selling costs, net of CGT.
- Invest net worth = portfolio minus CGT on gains at exit.

**Monte Carlo mode:**
- Runs N independent paths with returns sampled from normal distributions.
- House price, investment returns, and rent growth each vary year-to-year.
- Mortgage rates move year-to-year too (a random walk that drifts back to a
  long-run level). Your rate is locked for the fixed period, then you
  remortgage at the market rate for the remaining term.
- **Uncertainty ranges** (filled area) show 10th to 90th percentile outcomes (80% of paths).
- **Median** (bold line) shows the middle outcome.
- **Box plots** show the full distribution at 5-year intervals.
- **Win probability** shows % of simulations where each strategy wins.
- Useful for stress-testing and understanding range of outcomes.

**Live mortgage rates:**
- Monthly average quoted rates from the Bank of England, by fixed period and
  loan-to-value. They exclude product fees and lag by about a month.
- In deterministic mode the chosen rate is assumed to apply for the whole term.

**Both modes:**
- Both renter and owner pay council tax. Rent and house prices grow at their own rates.
- Investment return is net of fund fees (~0.16% OCF).
- SDLT bands are England & N. Ireland (Apr 2025); Scotland (LBTT) and Wales (LTT) differ.
- Tax treatment is simplified; seek professional advice before major decisions.
- **Not financial advice.** This is an educational tool for scenario planning.
        """
    )
