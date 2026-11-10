"""Sanity tests for the engine. Run: pytest -q"""
import math
import numpy as np
import pandas as pd
import pytest
import history
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


# ---- variable vs fixed history ---------------------------------------------

def _history(n=240, fix=5.0, var=4.0, bank=3.0, svr=6.0):
    dates = pd.date_range("2000-01-31", periods=n, freq="ME")
    col = lambda v: np.full(n, v, dtype=float) if np.isscalar(v) else np.asarray(v, float)
    return history.prepare(pd.DataFrame({
        "IUMBV34": col(fix), "IUMBV48": col(var), "IUMBV24": col(var),
        "IUMTLMV": col(svr), "IUMABEDR": col(bank)}, index=dates))


def test_history_prepare_drops_placeholder_zeros():
    d = _history(var=[0.0] * 12 + [4.0] * 228)
    assert len(d) == 228 and d["var2_margin"].iloc[0] == pytest.approx(0.01)
    life = _history(n=4, var=4.0)
    raw = pd.DataFrame({"IUMBV34": 5.0, "IUMBV48": 4.0, "IUMBV24": [4.0, 0.0, 4.0, 4.0],
                        "IUMTLMV": 6.0, "IUMABEDR": 3.0}, index=life.index)
    assert np.allclose(history.prepare(raw)["life_margin"], 0.01)


def test_history_constant_rate_matches_standard_mortgage():
    c = history.mortgage_cost(np.full((1, 300), 0.05), 300_000, 300)
    total_paid = monthly_mortgage_payment(300_000, 0.05, 25) * 300
    assert c["interest"][0] == pytest.approx(total_paid - 300_000)
    assert c["jump"][0] == pytest.approx(0, abs=1e-6)


def test_history_cheaper_variable_saves_money():
    r = history.run_test(_history(), 10)
    assert (r["savings"]["var2"] > 0).all()
    assert (r["savings"]["svr"] < 0).all()
    assert len(r["starts"]) == 240 - 120 + 1


def test_history_fix_locks_and_variable_follows_bank_rate():
    bank = [2.0] * 12 + [6.0] * 12 + [6.0] * 12
    d = _history(n=36, fix=3.0, var=3.0, bank=bank, svr=5.0)
    p = history.rate_paths(d, np.arange(36)[None, :])
    assert (p["fix2"][0, :24] == 0.03).all()
    assert p["var2"][0, 0] == pytest.approx(0.03)   # margin 1pp
    assert p["var2"][0, 12] == pytest.approx(0.07)  # Bank Rate up 4pp
    assert p["var2"][0, 24] == pytest.approx(0.03)  # new deal, margin -3pp
    assert p["life"][0, 24] == pytest.approx(0.07)  # keeps its first margin


def test_history_segments_stay_in_range():
    idx = history.segment_windows(100, 10, 24, 50)
    assert idx.shape == (50, 120) and idx.min() >= 0 and idx.max() < 100
    assert (np.diff(idx[0, :24]) == 1).all()


def test_history_single_deals_decompose():
    bank = np.linspace(6.0, 2.0, 120)
    d = _history(n=120, fix=5.0, var=bank + 1, bank=bank)
    deals = history.single_deals(d)
    assert (deals["bank_move"] < 0).all()
    assert deals["gap_no_drift"].mean() < deals["gap"].mean()


# ---- affordability ----------------------------------------------------------

from model import (balance_after, budget_path, monthly_budget,  # noqa: E402
                   mortgage_scenario, repayment_schedule)


def test_scenario_matches_spreadsheet_figures():
    # From the owner's spreadsheet: £400k at 2% over 20 years is £2,024/month
    s = mortgage_scenario(400_000, 0.02, 20)
    assert round(float(s["payment"])) == 2024
    expected = monthly_mortgage_payment(400_000, 0.02, 20) * 240 - 400_000
    assert float(s["total_interest"]) == pytest.approx(expected)


def test_scenario_matrix_broadcasts():
    amounts = np.array([[400_000], [500_000]])
    rates_ = np.array([[0.02, 0.05]])
    s = mortgage_scenario(amounts, rates_, 25)
    assert s["payment"].shape == (2, 2)
    assert s["payment"][1, 1] == pytest.approx(monthly_mortgage_payment(500_000, 0.05, 25))


def test_switch_matches_schedule_and_same_rate_changes_nothing():
    s = mortgage_scenario(500_000, 0.07, 25, switch_year=5, start_rate=0.04)
    sched = repayment_schedule(500_000, 0.07, 25, switch_year=5, start_rate=0.04)
    assert float(s["payment"]) == pytest.approx(sched["payment"].iloc[60])
    assert float(s["first_payment"]) == pytest.approx(sched["payment"].iloc[0])
    assert float(s["total_interest"]) == pytest.approx(sched["interest"].sum())
    assert float(s["balance"]) == pytest.approx(sched["balance"].iloc[59])
    same = mortgage_scenario(500_000, 0.04, 25, switch_year=5, start_rate=0.04)
    assert float(same["payment"]) == pytest.approx(float(same["first_payment"]))


def test_switch_after_term_is_paid_off():
    s = mortgage_scenario(300_000, 0.05, 20, switch_year=20, start_rate=0.05)
    assert float(s["payment"]) == 0 and float(s["balance"]) == pytest.approx(0, abs=1e-6)
    assert balance_after(300_000, 0.05, 20, 240) == pytest.approx(0, abs=1e-6)


def test_monthly_budget_costs_start_and_stop():
    costs = [(1_500, None, 4), (6_000, None, None), (800, 3, 6)]
    assert monthly_budget(11_000, costs, 0) == 3_500
    assert monthly_budget(11_000, costs, 3) == 2_700   # school fees start, childcare on
    assert monthly_budget(11_000, costs, 5) == 4_200   # childcare stopped
    assert monthly_budget(11_000, costs, 6) == 5_000   # both stopped
    assert monthly_budget(11_000, [(6_000, None, None)], 1, 0.10, 0.0) == pytest.approx(6_100)


def test_budget_path_feeds_scenarios():
    path = budget_path(9_000, [(4_000, None, None), (1_000, 2, None)], 360)
    assert path[0] == 5_000 and path[24] == 4_000 and len(path) == 360
    a = StochasticAssumptions(**LAB)
    s = Scenario("x", 0.2, 0.05)
    flat = run_scenarios(a, [s], 800_000, np.full(360, 5_000.0), n_sims=2,
                         include_running_costs=False)[0]
    lower = run_scenarios(a, [s], 800_000, path, n_sims=2, include_running_costs=False)[0]
    assert lower.net_worth[0, -1] < flat.net_worth[0, -1]
    # Without running costs the same budget leaves more to invest
    with_costs = run_scenarios(a, [s], 800_000, np.full(360, 5_000.0), n_sims=2)[0]
    assert with_costs.net_worth[0, -1] < flat.net_worth[0, -1]
    with pytest.raises(ValueError):
        run_scenarios(a, [s], 800_000, path[:12], n_sims=2)


# ---- save and load settings -------------------------------------------------

def test_settings_round_trip():
    import json
    import settings
    AppTest = pytest.importorskip("streamlit.testing.v1").AppTest
    at = AppTest.from_file("app.py", default_timeout=60).run()
    at.slider(key="sb_house_price").set_value(800_000).run()
    at.number_input(key="fin_income").set_value(9_100).run()
    at.multiselect(key="aff_terms").set_value([15, 25]).run()
    assert not at.exception
    payload = json.loads(settings.to_json(at.session_state["_settings_snapshot"]))
    assert payload["widgets"]["sb_house_price"] == 800_000
    assert payload["tables"]["fin_costs"][0]["Cost"] == "Bills and living costs"
    assert "sb_long_run" in payload["widgets"]
    payload["tables"]["fin_costs"][0]["£ per month"] = 2_750.0
    payload["widgets"]["sb_long_run"] = 6.5

    fresh = AppTest.from_file("app.py", default_timeout=60)
    restored = settings.apply(payload, fresh.session_state)
    fresh.run()
    assert not fresh.exception and restored > 20
    assert fresh.slider(key="sb_house_price").value == 800_000
    assert fresh.number_input(key="fin_income").value == 9_100
    assert fresh.multiselect(key="aff_terms").value == [15, 25]
    assert fresh.session_state["_settings_table_values"]["fin_costs"][0]["£ per month"] == 2_750
    long_run = [s for s in fresh.slider if s.label == "Long-run mortgage rate (%)"][0]
    assert long_run.value == 6.5


def test_settings_rejects_other_files():
    import settings
    with pytest.raises(ValueError):
        settings.apply({"app": "something-else"}, {})
    state = {}
    assert settings.apply({"app": "buy-vs-invest", "widgets": {"evil": 1, "sb_ftb": True}},
                          state) == 1
    assert "evil" not in state and state["sb_ftb"] is True
