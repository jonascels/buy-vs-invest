"""Sanity tests for the engine. Run: pytest -q"""
import math
import numpy as np
import rates
from model import (Assumptions, StochasticAssumptions, _variable_rate_schedule,
                   monthly_mortgage_payment, paid_rate_by_month, run_model,
                   run_model_stochastic, run_monte_carlo, stamp_duty, summarise)

NO_VOL = dict(investment_return_std=0.0, house_appreciation_std=0.0,
              rent_growth_std=0.0, mortgage_rate_std=0.0)


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
    lo, *_ = run_monte_carlo(StochasticAssumptions(investment_fee=0.0, **NO_VOL), n_sims=2)
    hi, *_ = run_monte_carlo(StochasticAssumptions(investment_fee=0.01, **NO_VOL), n_sims=2)
    assert hi["Invest median"].iloc[-1] < lo["Invest median"].iloc[-1]


# ---- mortgage rate risk ---------------------------------------------------

def test_zero_rate_vol_matches_fixed_mortgage():
    det = run_model(Assumptions())
    mc = run_model_stochastic(StochasticAssumptions(**NO_VOL))
    assert np.allclose(mc["mortgage_payment"], det["mortgage_payment"])
    assert np.allclose(mc["loan_balance"], det["loan_balance"])


def test_rate_locked_within_fixed_period():
    market = np.array([0.04, 0.06, 0.02, 0.05, 0.03, 0.07])
    paid = paid_rate_by_month(market, 2)
    assert len(paid) == 72
    assert set(paid[:24]) == {0.04}
    assert set(paid[24:48]) == {0.02}
    assert set(paid[48:]) == {0.03}


def test_variable_schedule_clears_loan_and_reprices():
    monthly = np.repeat([0.04] * 5 + [0.07] * 25, 12)
    pmts, bals = _variable_rate_schedule(300_000, monthly, 30)
    assert bals[-1] < 1e-6
    assert pmts[60] > pmts[59]  # payment rises at the remortgage


def test_higher_long_run_rate_hurts_buyer():
    base = dict(NO_VOL, fixed_period_years=2)
    lo, *_ = run_monte_carlo(StochasticAssumptions(mortgage_rate_long_run=0.03, **base), 2)
    hi, *_ = run_monte_carlo(StochasticAssumptions(mortgage_rate_long_run=0.08, **base), 2)
    assert hi["Buy median"].iloc[-1] < lo["Buy median"].iloc[-1]
    assert hi.attrs["rates"]["Paid rate p50"].iloc[-1] > 0.07


# ---- live rates (no network: parse a canned response) ---------------------

BOE_SAMPLE = """DATE,IUMBV34,IUMBV42,IUMZICQ,IUMB482,IUMTLMV
31 Aug 2025,4.10,4.20,3.90,4.60,7.00
30 Sep 2025,4.00,4.10,3.80,4.50,7.00
31 Aug 2026,4.92,4.78,4.79,5.16,6.58
30 Sep 2026,5.13,4.98,5.01,5.37,..
"""


def test_parse_and_latest():
    df = rates.parse_boe_csv(BOE_SAMPLE)
    assert df.index[-1].month == 9
    r, when = rates.latest_rate(df, "IUMBV34")
    assert math.isclose(r, 0.0513) and when.year == 2026
    # missing latest value falls back to the previous month
    assert math.isclose(rates.latest_rate(df, "IUMTLMV")[0], 0.0658)
    t = rates.latest_table(df).set_index("code")
    assert math.isclose(t.loc["IUMBV34", "12-month change (pp)"], 1.13)


def test_best_series_picks_ltv_band():
    assert rates.best_series(2, 55) == "IUMZICQ"   # 60% band
    assert rates.best_series(2, 72) == "IUMBV34"   # 75% band
    assert rates.best_series(2, 88) == "IUMB482"   # 90% band
    assert rates.best_series(2, 95) == "IUMB482"   # above top band
    assert rates.best_series(3, 50) == "IUMBV37"   # only 75% published
