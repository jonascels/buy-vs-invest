"""Sanity tests for the engine. Run: pytest -q"""
import math
import numpy as np
from model import (Assumptions, StochasticAssumptions, monthly_mortgage_payment,
                   run_model, run_monte_carlo, stamp_duty, summarise)


def test_stamp_duty_standard():
    assert math.isclose(stamp_duty(600_000), 20_000, abs_tol=1)


def test_stamp_duty_first_time_buyer():
    assert math.isclose(stamp_duty(400_000, first_time_buyer=True), 5_000, abs_tol=1)


def test_mortgage_payment_known():
    assert 2_400 < monthly_mortgage_payment(480_000, 0.045, 30) < 2_460


def test_cash_purchase_has_no_mortgage():
    a = Assumptions(deposit_pct=1.0)
    df = run_model(a)
    assert df.attrs["loan"] == 0
    assert df["mortgage_payment"].max() == 0
    assert df["loan_balance"].max() == 0
    assert len(summarise(df, a)) == len(a.horizons)


def test_cash_purchase_monte_carlo_runs():
    np.random.seed(0)
    pct, wins, _, _ = run_monte_carlo(StochasticAssumptions(deposit_pct=1.0), n_sims=10)
    assert len(wins) == len(StochasticAssumptions().horizons)


def test_mc_applies_fees():
    # With zero volatility, MC must match deterministic-style fee drag:
    # higher fees -> lower median invest outcome.
    base = dict(investment_return_std=0.0, house_appreciation_std=0.0, rent_growth_std=0.0)
    lo, *_ = run_monte_carlo(StochasticAssumptions(investment_fee=0.0, **base), n_sims=2)
    hi, *_ = run_monte_carlo(StochasticAssumptions(investment_fee=0.01, **base), n_sims=2)
    assert hi["Invest median"].iloc[-1] < lo["Invest median"].iloc[-1]
