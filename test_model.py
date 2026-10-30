"""Sanity tests for the engine. Run: pytest -q"""
import math
import numpy as np
import pytest
import rates
from model import (INTEREST_ONLY, REPAYMENT, Assumptions, Scenario, StochasticAssumptions,
                   _variable_rate_schedule, compare_scenarios, draw_shocks,
                   monthly_mortgage_payment, paid_rate_by_month, run_model,
                   run_model_stochastic, run_monte_carlo, run_scenarios,
                   simulate_scenario, stamp_duty, summarise, upfront_cash_needed)

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


# ---- scenario lab ---------------------------------------------------------

LAB = dict(NO_VOL, fixed_period_years=5, mortgage_rate=0.05)


def _one(s, cash=800_000, budget=6_000, n=3, **kw):
    a = StochasticAssumptions(**{**LAB, **kw})
    return run_scenarios(a, [s], cash, budget, n_sims=n)[0]


def test_scenario_repayment_matches_schedule():
    r = _one(Scenario("r", 0.2, 0.05))
    det = run_model(Assumptions(mortgage_rate=0.05))
    assert math.isclose(r.first_payment, det["mortgage_payment"].iloc[0], rel_tol=1e-9)
    assert np.allclose(r.balance[0], det["loan_balance"].values[11::12])


def test_interest_only_keeps_balance_then_falls_due():
    r = _one(Scenario("io", 0.2, 0.05, INTEREST_ONLY, term_years=20))
    assert np.allclose(r.balance[0, :19], 480_000)
    assert r.balance[0, 19] == 0  # repaid at term end
    assert math.isclose(r.first_payment, 480_000 * 0.05 / 12)
    assert np.allclose(r.interest_paid[0, 19], 480_000 * 0.05 * 20)


def test_interest_only_funding_gap_when_cash_is_short():
    a = StochasticAssumptions(**LAB)
    s = Scenario("io", 0.2, 0.05, INTEREST_ONLY, term_years=20)
    upfront = upfront_cash_needed(a, 0.2)
    r = simulate_scenario(a, s, upfront, 2_800, draw_shocks(a, 2))
    assert r.funding_gap[0, 19] > 0


def test_money_is_conserved_with_no_growth():
    # No growth, no tax, no selling costs: net worth = cash + budget paid in
    # minus everything spent (interest, running costs, SDLT, legal).
    kw = dict(investment_return=0.0, investment_fee=0.0, house_appreciation=0.0,
              inflation=0.0, capital_gains_tax=0.0, selling_cost_pct=0.0,
              maintenance_pct=0.0, insurance_pct=0.0, council_tax=0.0)
    for typ in (REPAYMENT, INTEREST_ONLY):
        r = _one(Scenario("x", 0.2, 0.05, typ), cash=300_000, budget=4_000, **kw)
        months = 30 * 12
        expected = 300_000 + 4_000 * months - r.interest_paid[0, -1] \
            - (20_000 + 2_500)
        assert math.isclose(r.net_worth[0, -1], expected, rel_tol=1e-9)


def test_scenarios_share_shocks_and_best_chance_sums():
    a = StochasticAssumptions(mortgage_rate=0.05)
    res = run_scenarios(a, [Scenario("a", 0.2, 0.05), Scenario("b", 0.4, 0.05),
                            Scenario("c", 0.2, 0.05)], 400_000, 5_000, n_sims=50)
    assert np.array_equal(res[0].net_worth, res[2].net_worth)
    assert math.isclose(compare_scenarios(res, 10)["Chance best (%)"].sum(), 100)


def test_scenario_refuses_unaffordable_deposit():
    with pytest.raises(ValueError):
        _one(Scenario("x", 0.9, 0.05), cash=100_000)


def test_app_runs_with_plausible_lab_results():
    # Guards against unit slips between the UI and the engine, such as a
    # percentage-point input being passed on as a decimal rate.
    AppTest = pytest.importorskip("streamlit.testing.v1").AppTest
    at = AppTest.from_file("app.py", default_timeout=60).run()
    assert not at.exception
    gap = next(m for m in at.metric if m.label == "Median gap, best to worst")
    assert float(gap.value.replace("£", "").replace(",", "")) < 2_000_000
    at.sidebar.radio[0].set_value("Monte Carlo").run()
    assert not at.exception
