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

    # Mortgage rate risk: the rate is fixed for `fixed_period_years`, then the
    # loan is remortgaged at whatever the market rate is by then. The market
    # rate follows a mean-reverting random walk with annual steps.
    mortgage_rate_std: float = 0.01      # std dev of 1-year change (1.0pp)
    fixed_period_years: int = 5          # remortgage every N years
    mortgage_rate_long_run: float | None = None  # None -> today's rate
    mortgage_rate_reversion: float = 0.2  # share of gap to long run closed per yr
    mortgage_rate_floor: float = 0.005


def market_rate_path(a: StochasticAssumptions, shocks: np.ndarray) -> np.ndarray:
    """Market mortgage rate at the start of each year (index 0 = today).

    shocks[y] is the random part of the move into year y (shocks[0] unused).
    """
    long_run = a.mortgage_rate if a.mortgage_rate_long_run is None \
        else a.mortgage_rate_long_run
    rates = np.empty(len(shocks))
    rates[0] = a.mortgage_rate
    for y in range(1, len(shocks)):
        r = rates[y - 1] + a.mortgage_rate_reversion * (long_run - rates[y - 1])
        rates[y] = max(r + shocks[y], a.mortgage_rate_floor)
    return rates


def paid_rate_by_month(market_rates: np.ndarray, fixed_period_years: int
                       ) -> np.ndarray:
    """The rate the borrower actually pays each month.

    It is locked at the market rate at the start of each fixed period and
    held until the next remortgage.
    """
    years = len(market_rates)
    fp = max(int(fixed_period_years), 1)
    locked = np.array([market_rates[(y // fp) * fp] for y in range(years)])
    return np.repeat(locked, 12)


def _variable_rate_schedule(principal: float, monthly_rates: np.ndarray,
                            term_years: int) -> Tuple[np.ndarray, np.ndarray]:
    """Payments and end-of-month balances when the rate can change.

    Whenever the rate changes the payment is recalculated so the remaining
    balance still clears by the end of the original term. With a constant
    rate this matches `_amortisation_schedule`.
    """
    months = len(monthly_rates)
    term_months = term_years * 12
    payments = np.zeros(months)
    balances = np.zeros(months)
    balance = principal
    pmt = 0.0
    prev_rate = None
    for m in range(months):
        rate = monthly_rates[m]
        if balance <= 0 or m >= term_months:
            balances[m] = max(balance, 0.0)
            continue
        if rate != prev_rate:
            pmt = _payment_for_months(balance, rate, term_months - m)
            prev_rate = rate
        interest = balance * rate / 12
        principal_paid = min(pmt - interest, balance)
        payments[m] = interest + principal_paid
        balance = max(balance - principal_paid, 0.0)
        balances[m] = balance
    return payments, balances


def _payment_for_months(principal: float, annual_rate: float, n: int) -> float:
    r = annual_rate / 12.0
    if r == 0:
        return principal / n
    return principal * r * (1 + r) ** n / ((1 + r) ** n - 1)


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
    inv_shocks = np.random.normal(a.investment_return - a.investment_fee,
                                  a.investment_return_std, max_years)
    house_shocks = np.random.normal(a.house_appreciation, a.house_appreciation_std, max_years)
    rent_shocks = np.random.normal(a.rent_growth, a.rent_growth_std, max_years)
    rate_shocks = np.random.normal(0.0, a.mortgage_rate_std, max_years)

    # Expand annual shocks to monthly (constant within each year)
    inv_monthly = np.repeat(inv_shocks, 12) / 12
    house_monthly = np.repeat(house_shocks, 12) / 12
    rent_monthly = np.repeat(rent_shocks, 12) / 12

    deposit = a.house_price * a.deposit_pct
    loan = a.house_price - deposit
    sdlt = stamp_duty(a.house_price, a.first_time_buyer, a.additional_property)
    upfront_purchase_costs = deposit + sdlt + a.legal_survey_fixed

    # Mortgage: market rate path, locked for each fixed period
    market_rates = market_rate_path(a, rate_shocks)
    paid_rates = paid_rate_by_month(market_rates, a.fixed_period_years)
    payments, balances = _variable_rate_schedule(loan, paid_rates,
                                                 a.mortgage_term_years)

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
        
        mortgage_pmt = payments[m - 1]
        loan_balance = balances[m - 1]
        
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
            "mortgage_rate": paid_rates[m - 1] if loan > 0 else 0.0,
            "market_mortgage_rate": market_rates[(m - 1) // 12],
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

    percentile_bounds.attrs["rates"] holds yearly 10th/50th/90th percentiles of
    the market mortgage rate, the rate actually paid and the monthly payment.
    """
    buy_outcomes = {h: [] for h in a.horizons}
    invest_outcomes = {h: [] for h in a.horizons}
    yearly = slice(0, None, 12)  # first month of each year
    market_paths, paid_paths, pmt_paths = [], [], []
    
    for sim in range(n_sims):
        df = run_model_stochastic(a, seed=None)
        market_paths.append(df["market_mortgage_rate"].values[yearly])
        paid_paths.append(df["mortgage_rate"].values[yearly])
        pmt_paths.append(df["mortgage_payment"].values[yearly])
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
    
    pct_df = pd.DataFrame(percentile_rows)
    rates = {"Year": np.arange(max(a.horizons))}
    for name, paths in (("Market rate", market_paths), ("Paid rate", paid_paths),
                        ("Payment", pmt_paths)):
        arr = np.array(paths)
        for q in (10, 50, 90):
            rates[f"{name} p{q}"] = np.percentile(arr, q, axis=0)
    pct_df.attrs["rates"] = pd.DataFrame(rates)

    return (pct_df, pd.DataFrame(win_rows), buy_outcomes, invest_outcomes)
