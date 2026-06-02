"""
Buy vs. Invest financial model with Monte Carlo.

Pure-Python engine (no Streamlit dependency) so it can be unit-tested and
reused. All money is in nominal GBP unless a field name says "real".

The central question: if you have a deposit + monthly budget, are you better
off (a) buying a home with a mortgage, or (b) renting and investing the
deposit plus the monthly cash-flow difference? We compare terminal net worth
across several horizons.

Monte Carlo mode: run stochastic paths with normally-distributed returns.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Tuple

import numpy as np
import pandas as pd


# --------------------------------------------------------------------------- #
# UK Stamp Duty Land Tax (SDLT) — residential, England & N. Ireland
# --------------------------------------------------------------------------- #
def stamp_duty(price: float, first_time_buyer: bool = False,
               additional_property: bool = False) -> float:
    """SDLT on a residential purchase using the bands in force from Apr 2025.

    Standard bands:  0% to 125k, 2% 125k-250k, 5% 250k-925k,
                     10% 925k-1.5m, 12% above 1.5m.
    First-time buyer relief: 0% to 300k, 5% 300k-500k (no relief above 500k).
    Additional property (second home / BTL): +5% surcharge on every band.
    """
    if first_time_buyer and price <= 500_000:
        bands = [(0, 300_000, 0.00), (300_000, 500_000, 0.05)]
    else:
        bands = [
            (0, 125_000, 0.00),
            (125_000, 250_000, 0.02),
            (250_000, 925_000, 0.05),
            (925_000, 1_500_000, 0.10),
            (1_500_000, float("inf"), 0.12),
        ]

    surcharge = 0.05 if additional_property else 0.0
    tax = 0.0
    for lo, hi, rate in bands:
        if price > lo:
            taxable = min(price, hi) - lo
            tax += taxable * (rate + surcharge)
        else:
            break
    # surcharge also applies to the 0% portion for additional properties
    if additional_property:
        tax += min(price, 125_000) * 0.0  # band already covers it above
    return round(tax, 2)


def monthly_mortgage_payment(principal: float, annual_rate: float,
                             term_years: int) -> float:
    """Standard repayment (amortising) monthly payment."""
    if principal <= 0:
        return 0.0
    r = annual_rate / 12.0
    n = term_years * 12
    if r == 0:
        return principal / n
    return principal * r * (1 + r) ** n / ((1 + r) ** n - 1)


@dataclass
class Assumptions:
    # Property
    house_price: float = 600_000.0
    deposit_pct: float = 0.20            # share of price paid as deposit
    mortgage_rate: float = 0.045         # annual nominal
    mortgage_term_years: int = 30
    house_appreciation: float = 0.03     # annual nominal capital growth
    first_time_buyer: bool = False
    additional_property: bool = False

    # Recurring property costs (annual, as % of current house value unless noted)
    maintenance_pct: float = 0.01        # upkeep / repairs
    insurance_pct: float = 0.0015        # buildings insurance
    service_charge: float = 0.0          # flat annual £ (leasehold/ground rent)
    council_tax: float = 2_200.0         # flat annual £ (paid by owner OR renter)

    # One-off purchase costs (beyond stamp duty)
    legal_survey_fixed: float = 2_500.0  # solicitor + survey + searches
    selling_cost_pct: float = 0.0175     # estate agent + legal when sold

    # Rent (the alternative)
    monthly_rent: float = 2_200.0
    rent_growth: float = 0.03            # annual

    # Investment alternative
    investment_return: float = 0.07      # annual nominal (e.g. global equities)
    investment_fee: float = 0.0016       # annual OCF drag (your ~0.16% blend)
    capital_gains_tax: float = 0.20      # CGT on gains when liquidated

    # Economy
    inflation: float = 0.025             # annual CPI, for "real" conversions

    # Horizons to report
    horizons: List[int] = field(default_factory=lambda: [5, 10, 15, 20, 25, 30])


def _amortisation_schedule(principal: float, annual_rate: float,
                           term_years: int, months: int) -> pd.DataFrame:
    """Month-by-month balance, interest and principal for `months` months."""
    pmt = monthly_mortgage_payment(principal, annual_rate, term_years)
    r = annual_rate / 12.0
    rows = []
    balance = principal
    for m in range(1, months + 1):
        interest = balance * r
        principal_paid = min(pmt - interest, balance)
        balance = max(balance - principal_paid, 0.0)
        rows.append((m, pmt if balance > 0 or principal_paid > 0 else 0.0,
                     interest, principal_paid, balance))
    return pd.DataFrame(
        rows, columns=["month", "payment", "interest", "principal", "balance"]
    )


def run_model(a: Assumptions) -> pd.DataFrame:
    """Simulate both paths month-by-month over the longest horizon.

    Returns a tidy monthly DataFrame with the net-worth of each strategy and
    the components behind them.
    """
    max_years = max(a.horizons)
    months = max_years * 12

    deposit = a.house_price * a.deposit_pct
    loan = a.house_price - deposit
    sdlt = stamp_duty(a.house_price, a.first_time_buyer, a.additional_property)
    upfront_purchase_costs = deposit + sdlt + a.legal_survey_fixed

    sched = _amortisation_schedule(loan, a.mortgage_rate,
                                   a.mortgage_term_years, months)

    # Monthly growth factors
    g_house = (1 + a.house_appreciation) ** (1 / 12) - 1
    g_rent = (1 + a.rent_growth) ** (1 / 12) - 1
    net_inv_return = a.investment_return - a.investment_fee
    g_inv = (1 + net_inv_return) ** (1 / 12) - 1
    g_infl = (1 + a.inflation) ** (1 / 12) - 1

    # --- BUY path state ---
    house_value = a.house_price
    # --- INVEST path state ---
    # The renter invests the money the buyer sank upfront (deposit + fees),
    # then each month invests any positive difference between the buyer's
    # housing outgoings and their own rent.
    portfolio = upfront_purchase_costs
    invested_basis = upfront_purchase_costs  # cost basis for CGT
    rent = a.monthly_rent

    records = []
    for m in range(1, months + 1):
        # ----- grow asset values for the month -----
        house_value *= (1 + g_house)
        row = sched.loc[m - 1]
        mortgage_pmt = row["payment"]
        loan_balance = row["balance"]

        # property running costs this month
        maint = house_value * a.maintenance_pct / 12
        ins = house_value * a.insurance_pct / 12
        svc = a.service_charge / 12
        ctax = a.council_tax / 12
        buy_monthly_outgoing = mortgage_pmt + maint + ins + svc + ctax

        # renter's outgoing
        rent_monthly_outgoing = rent + ctax  # renters also pay council tax

        # cash-flow difference the renter can invest (if buying costs more)
        diff = buy_monthly_outgoing - rent_monthly_outgoing
        contribution = max(diff, 0.0)
        # if renting costs MORE than buying, the buyer invests the surplus;
        # handled symmetrically by tracking a buyer-side side-portfolio:
        buyer_side_contribution = max(-diff, 0.0)

        # grow portfolios then add contributions (end-of-month)
        portfolio *= (1 + g_inv)
        portfolio += contribution
        invested_basis += contribution

        records.append({
            "month": m,
            "year": m / 12.0,
            "house_value": house_value,
            "loan_balance": loan_balance,
            "home_equity": house_value - loan_balance,
            "mortgage_payment": mortgage_pmt,
            "maintenance": maint,
            "insurance": ins,
            "buy_outgoing": buy_monthly_outgoing,
            "rent": rent,
            "rent_outgoing": rent_monthly_outgoing,
            "renter_contribution": contribution,
            "buyer_surplus": buyer_side_contribution,
            "portfolio_value": portfolio,
            "invested_basis": invested_basis,
            "inflation_index": (1 + g_infl) ** m,
        })

        rent *= (1 + g_rent)

    df = pd.DataFrame(records)

    # ---- net worth (after liquidation costs & taxes) at every month ----
    # BUY net worth = home equity net of selling costs, plus any buyer-side
    #                 surplus that was invested (we approximate by reinvesting
    #                 buyer_surplus into the same portfolio after the renter's
    #                 contributions net to zero — handled below).
    df["selling_cost"] = df["house_value"] * a.selling_cost_pct
    df["buy_net_worth"] = df["home_equity"] - df["selling_cost"]

    # buyer-side surplus portfolio (when renting would have cost more)
    bs = 0.0
    bs_basis = 0.0
    buyer_port = []
    for _, r in df.iterrows():
        bs *= (1 + g_inv)
        bs += r["buyer_surplus"]
        bs_basis += r["buyer_surplus"]
        buyer_port.append((bs, bs_basis))
    bs_vals = pd.DataFrame(buyer_port, columns=["buyer_port", "buyer_basis"])
    df["buyer_port"] = bs_vals["buyer_port"].values
    df["buyer_basis"] = bs_vals["buyer_basis"].values
    buyer_gain = (df["buyer_port"] - df["buyer_basis"]).clip(lower=0)
    df["buy_net_worth"] += df["buyer_port"] - buyer_gain * a.capital_gains_tax

    # INVEST net worth = portfolio minus CGT on gains
    gain = (df["portfolio_value"] - df["invested_basis"]).clip(lower=0)
    df["invest_net_worth"] = df["portfolio_value"] - gain * a.capital_gains_tax

    # real (today's money) versions
    df["buy_net_worth_real"] = df["buy_net_worth"] / df["inflation_index"]
    df["invest_net_worth_real"] = df["invest_net_worth"] / df["inflation_index"]

    df.attrs["sdlt"] = sdlt
    df.attrs["upfront_purchase_costs"] = upfront_purchase_costs
    df.attrs["loan"] = loan
    df.attrs["deposit"] = deposit
    return df


def summarise(df: pd.DataFrame, a: Assumptions) -> pd.DataFrame:
    """One row per horizon with the headline comparison."""
    rows = []
    for h in a.horizons:
        snap = df.iloc[min(h * 12, len(df)) - 1]
        buy = snap["buy_net_worth"]
        inv = snap["invest_net_worth"]
        rows.append({
            "Horizon (yrs)": h,
            "Buy net worth": buy,
            "Invest net worth": inv,
            "Difference (Buy − Invest)": buy - inv,
            "Winner": "Buy" if buy >= inv else "Invest",
            "Buy (real)": snap["buy_net_worth_real"],
            "Invest (real)": snap["invest_net_worth_real"],
        })
    return pd.DataFrame(rows)


# ========================================================================== #
# Monte Carlo: stochastic simulation
# ========================================================================== #

@dataclass
class StochasticAssumptions(Assumptions):
    """Like Assumptions, but with standard deviations for stochastic variables."""
    investment_return_std: float = 0.12  # annual volatility, e.g. equities
    house_appreciation_std: float = 0.08  # annual volatility in house prices
    rent_growth_std: float = 0.02        # annual volatility in rent


def run_model_stochastic(a: StochasticAssumptions, seed: int | None = None
                        ) -> pd.DataFrame:
    """Simulate one stochastic path with normally-distributed returns.
    
    Returns monthly_df with house_value, portfolio_value, etc.
    """
    if seed is not None:
        np.random.seed(seed)
    
    max_years = max(a.horizons)
    months = max_years * 12
    
    # Draw annual shocks for each path
    inv_shocks = np.random.normal(a.investment_return, a.investment_return_std, max_years)
    house_shocks = np.random.normal(a.house_appreciation, a.house_appreciation_std, max_years)
    rent_shocks = np.random.normal(a.rent_growth, a.rent_growth_std, max_years)
    
    # Expand annual shocks to monthly (constant within each year)
    inv_monthly = np.repeat(inv_shocks, 12) / 12
    house_monthly = np.repeat(house_shocks, 12) / 12
    rent_monthly = np.repeat(rent_shocks, 12) / 12
    
    deposit = a.house_price * a.deposit_pct
    loan = a.house_price - deposit
    sdlt = stamp_duty(a.house_price, a.first_time_buyer, a.additional_property)
    upfront_purchase_costs = deposit + sdlt + a.legal_survey_fixed
    
    sched = _amortisation_schedule(loan, a.mortgage_rate,
                                   a.mortgage_term_years, months)
    
    # Monthly inflation (constant, not stochastic)
    g_infl = (1 + a.inflation) ** (1 / 12) - 1
    
    # --- BUY path state ---
    house_value = a.house_price
    # --- INVEST path state ---
    portfolio = upfront_purchase_costs
    invested_basis = upfront_purchase_costs
    rent = a.monthly_rent
    
    records = []
    for m in range(1, months + 1):
        # Stochastic growth for this month
        g_house_m = house_monthly[m - 1]
        g_rent_m = rent_monthly[m - 1]
        g_inv_m = inv_monthly[m - 1]
        
        house_value *= (1 + g_house_m)
        
        row = sched.loc[m - 1]
        mortgage_pmt = row["payment"]
        loan_balance = row["balance"]
        
        # property running costs
        maint = house_value * a.maintenance_pct / 12
        ins = house_value * a.insurance_pct / 12
        svc = a.service_charge / 12
        ctax = a.council_tax / 12
        buy_monthly_outgoing = mortgage_pmt + maint + ins + svc + ctax
        
        # renter's outgoing
        rent_monthly_outgoing = rent + ctax
        
        # cash-flow difference the renter can invest
        diff = buy_monthly_outgoing - rent_monthly_outgoing
        contribution = max(diff, 0.0)
        buyer_side_contribution = max(-diff, 0.0)
        
        # grow portfolios
        portfolio *= (1 + g_inv_m)
        portfolio += contribution
        invested_basis += contribution
        
        records.append({
            "month": m,
            "year": m / 12.0,
            "house_value": house_value,
            "loan_balance": loan_balance,
            "home_equity": house_value - loan_balance,
            "mortgage_payment": mortgage_pmt,
            "maintenance": maint,
            "insurance": ins,
            "buy_outgoing": buy_monthly_outgoing,
            "rent": rent,
            "rent_outgoing": rent_monthly_outgoing,
            "renter_contribution": contribution,
            "buyer_surplus": buyer_side_contribution,
            "portfolio_value": portfolio,
            "invested_basis": invested_basis,
            "inflation_index": (1 + g_infl) ** m,
        })
        
        rent *= (1 + g_rent_m)
    
    df = pd.DataFrame(records)
    
    # Calculate net worths
    df["selling_cost"] = df["house_value"] * a.selling_cost_pct
    df["buy_net_worth"] = df["home_equity"] - df["selling_cost"]
    
    bs = 0.0
    bs_basis = 0.0
    buyer_port = []
    for _, r in df.iterrows():
        bs *= (1 + inv_monthly[int(r["month"]) - 1])
        bs += r["buyer_surplus"]
        bs_basis += r["buyer_surplus"]
        buyer_port.append((bs, bs_basis))
    bs_vals = pd.DataFrame(buyer_port, columns=["buyer_port", "buyer_basis"])
    df["buyer_port"] = bs_vals["buyer_port"].values
    df["buyer_basis"] = bs_vals["buyer_basis"].values
    buyer_gain = (df["buyer_port"] - df["buyer_basis"]).clip(lower=0)
    df["buy_net_worth"] += df["buyer_port"] - buyer_gain * a.capital_gains_tax
    
    gain = (df["portfolio_value"] - df["invested_basis"]).clip(lower=0)
    df["invest_net_worth"] = df["portfolio_value"] - gain * a.capital_gains_tax
    
    df["buy_net_worth_real"] = df["buy_net_worth"] / df["inflation_index"]
    df["invest_net_worth_real"] = df["invest_net_worth"] / df["inflation_index"]
    
    return df


def run_monte_carlo(a: StochasticAssumptions, n_sims: int = 100
                   ) -> Tuple[pd.DataFrame, pd.DataFrame, dict, dict]:
    """Run n_sims stochastic paths.
    
    Returns (percentile_bounds, win_summary, buy_outcomes, invest_outcomes).
    - percentile_bounds: 10th/50th/90th percentiles by horizon
    - win_summary: % of paths where buy/invest wins
    - buy_outcomes, invest_outcomes: dicts of lists for each horizon
    """
    buy_outcomes = {h: [] for h in a.horizons}
    invest_outcomes = {h: [] for h in a.horizons}
    
    for sim in range(n_sims):
        df = run_model_stochastic(a, seed=None)
        for h in a.horizons:
            idx = min(h * 12, len(df)) - 1
            buy_outcomes[h].append(df.iloc[idx]["buy_net_worth"])
            invest_outcomes[h].append(df.iloc[idx]["invest_net_worth"])
    
    # Percentile summary
    percentile_rows = []
    for h in a.horizons:
        buy_arr = np.array(buy_outcomes[h])
        inv_arr = np.array(invest_outcomes[h])
        percentile_rows.append({
            "Horizon (yrs)": h,
            "Buy 10th %ile": np.percentile(buy_arr, 10),
            "Buy median": np.percentile(buy_arr, 50),
            "Buy 90th %ile": np.percentile(buy_arr, 90),
            "Invest 10th %ile": np.percentile(inv_arr, 10),
            "Invest median": np.percentile(inv_arr, 50),
            "Invest 90th %ile": np.percentile(inv_arr, 90),
        })
    
    # Win rate summary
    win_rows = []
    for h in a.horizons:
        buy_arr = np.array(buy_outcomes[h])
        inv_arr = np.array(invest_outcomes[h])
        buy_win_pct = 100 * np.mean(buy_arr >= inv_arr)
        win_rows.append({
            "Horizon (yrs)": h,
            "Buy wins (%)": buy_win_pct,
            "Invest wins (%)": 100 - buy_win_pct,
        })
    
    return (pd.DataFrame(percentile_rows), pd.DataFrame(win_rows),
            buy_outcomes, invest_outcomes)
