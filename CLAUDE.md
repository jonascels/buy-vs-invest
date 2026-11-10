# CLAUDE.md — handover for Claude Code

Context for continuing work on this repo. Read this first.

## What this is
A UK-focused **buy a home vs. rent and invest** model with a Streamlit UI.
Owner: Jonas (GitHub `jonascels`). Intended audience includes a non-technical
family member, so the deployed app must stay simple to use.

## Files
| File | Role |
|---|---|
| `model.py` | Pure-Python engine, no Streamlit imports. Deterministic + Monte Carlo. |
| `app.py` | Streamlit UI: sidebar inputs, Plotly charts, CSV export. |
| `afford.py` | Streamlit UI for the Affordability tab (payment matrix by loan, rate and term). |
| `lab.py` | Streamlit UI for the Mortgage lab tab (deposit sweep, custom scenarios). |
| `history.py` | Historical variable vs fixed test (no Streamlit). `analysis/variable_vs_fixed.py` prints it. |
| `rates.py` | Live UK mortgage rates from the Bank of England database (no Streamlit). |
| `test_model.py` | Pytest sanity tests. Run `python -m pytest -q` in the env before committing. |
| `.streamlit/config.toml` | Light theme. Theming lives here, **not** in CSS. |
| `environment.yml` | Conda env `buy-vs-invest` (py3.12, numpy, pandas, pytest, streamlit, plotly). |
| `.claude/launch.json` | Lets Claude Code's browser pane start the app. |

## Run
```bash
conda activate buy-vs-invest
streamlit run app.py
python -m pytest -q
```

## Model logic (key assumptions)
- **Buy path**: amortising repayment mortgage; net worth = house value − loan
  balance − selling costs, plus any invested surplus (when owning is cheaper
  than renting) net of CGT.
- **Invest path**: renter invests the buyer's full upfront cash (deposit + SDLT
  + legal) on day one, then invests the monthly difference whenever buying
  costs more than renting. Net of CGT at exit.
- Both pay council tax. Investment return is net of fund fees (OCF).
- **SDLT**: England & NI bands from Apr 2025, first-time-buyer relief,
  +5% additional-property surcharge. Scotland/Wales not modelled.
- **Cash purchase**: deposit slider goes to 100%. Loan = 0, so mortgage
  payment and balance are 0; rate/term sliders are disabled in the UI.
- **Live rates**: `rates.py` pulls monthly average quoted rates (excl. fees)
  from the BoE IADB CSV endpoint: 2/3/5/10-yr fixes at 75% LTV, plus 60/85/90%
  LTV bands for 2-yr and 60/90% for 5-yr, and the revert-to rate (SVR). The
  app picks the band the user's LTV fits under (`best_series`), caches the
  fetch for 6h, and falls back to a manual slider if the BoE is unreachable.
  Deterministic mode applies the chosen rate for the whole term.
- **Horizons**: `[5, 10, 15, 20, 25, 30]` years (set in `Assumptions.horizons`).

## Monte Carlo
- `StochasticAssumptions` adds annual std devs for investment return, house
  appreciation and rent growth.
- Each path draws one normal shock per year, held constant across its 12
  months. Shocks are independent (no correlation between house prices and
  equities). Inflation is fixed.
- **Mortgage rate risk**: market rate follows a mean-reverting annual random
  walk (`mortgage_rate_std`, default from 15y of BoE 12-month changes;
  `mortgage_rate_long_run`, default = today's rate; reversion 0.2/yr; floor
  0.5%). The paid rate is locked for `fixed_period_years`, then the loan is
  remortgaged at the market rate and the payment recalculated over the
  remaining term (`_variable_rate_schedule`). With zero rate vol it matches
  the deterministic fixed-rate schedule exactly (tested).
- Rate/payment percentile bands are returned in `pct_df.attrs["rates"]`.
- User picks sim count (10–1,000); ~10 ms per sim.
- Outputs: win-probability cards, filled 10th–90th percentile bands with
  median lines, box plots per horizon, percentile + win-rate tables.

## Affordability (second tab)
- Replicates the owner's Excel: monthly payment matrix with mortgage sizes as rows
  and rates as columns, one matrix per term (default 20/25/30), plus cash needed,
  LTV and share of savings per row. Cash needed = deposit + SDLT + legal fees.
- Budget = take-home income minus listed costs; costs can stop after N years
  (e.g. childcare). Cells: ✓ fits, ! within 10%, ✗ over; greyed rows need more
  cash than savings.
- "In N years": pays a starting rate for N years, then remortgages at the column
  rate (`mortgage_scenario`, closed form). Same rate means the same payment.
  Budget projects pay rises and cost growth.
- Detail view: headline cards per term (payment, total interest, total repaid,
  mortgage-free year), payment over time against the budget, balance left and
  interest vs principal by year.
- Engine: `mortgage_scenario`, `repayment_schedule`, `monthly_budget` in `model.py`.

## Mortgage lab (third tab)
- Engine: `Scenario`, `simulate_scenario`, `run_scenarios`, `compare_scenarios` in
  `model.py`. NumPy-vectorised across sims, about 10 ms per 1,000 paths per
  scenario, so the tab recalculates on every edit with no run button.
- **Fair comparison**: every scenario gets the same cash pot and monthly budget
  (budget grows with inflation). Unspent cash is invested on day one; monthly
  surplus is invested. Scenarios share shocks (`draw_shocks`), so path-by-path
  differences and "chance best" are meaningful. Rent isn't in this comparison.
- **Interest-only**: balance stays flat and falls due at the end of the term,
  paid from investments net of CGT. If cash runs out, the shortfall becomes a
  funding-gap debt at the paid mortgage rate, cleared by later surpluses and
  reported as "Ran out of cash (%)". Rate = live rate + a user premium
  (default 0.25pp, our assumption: the BoE doesn't publish interest-only rates).
- Each deposit uses its own LTV band rate, which only varies for 2- and 5-yr
  fixes. No series above 90% LTV, so 5% deposits may look too cheap.
- `render_lab` is an `@st.fragment`, so lab edits don't re-run the first tab.
- Scenarios last for the browser session; there's no save or load yet.
- Volatility sliders are always in the sidebar because both tabs use them.
- Lab colours are the dataviz skill's reference categorical palette, in order.

## Variable vs fixed (third section of the Mortgage lab)
- Tests whether variable beats rolling 2-year fixes, replaying BoE quoted rates at
  75% LTV, excl. fees. Variable deal data starts Apr 1998 (zeros in the variable
  and lifetime series are placeholders), so 30-40 years isn't possible.
- Variable types: rolling 2-yr variable (`IUMBV48`, margin over Bank Rate locked
  24 months, then re-quoted; a proxy, as some older deals may be discounts off
  SVR), lifetime tracker (`IUMBV24`, margin locked throughout) and SVR (`IUMTLMV`).
- Methods: continuous (every start month) or segments (block bootstrap, 2- or
  5-year pieces). Continuous windows overlap heavily, so few are independent.
  Segment payment rises are overstated because rates leap where pieces join.
- Results are read from data at run time, so don't hardcode conclusions in the UI.

## UI decisions the owner has made (don't undo without asking)
- Light theme via `config.toml`. Heavy `!important` CSS overrides broke table
  rendering; keep CSS minimal.
- Keep **box-and-whisker** plots for distributions (histograms were tried and
  rejected).
- Keep the **filled uncertainty band** chart.
- Buy = blue `#1e88e5`, Invest = orange `#f57c00`.

## Known limitations / ideas
- `run_model` and `run_model_stochastic` duplicate most of their loop. A
  refactor to one engine taking per-month growth arrays would remove that.
- Simulation loops are pure Python with `iterrows`; vectorising across sims
  with NumPy would make 1,000+ sims near-instant.
- No correlation between asset shocks (e.g. rates vs house prices).
- Lender rules (max LTV for interest-only, affordability) aren't enforced.
- No remortgage fees, overpayments, ISA/pension tax wrappers, or rental income.
- MC grows assets with simple `rate/12` monthly steps while deterministic
  compounds, so zero-vol MC net worth sits slightly below deterministic.
- `use_container_width` is deprecated in current Streamlit (log warnings).
- CGT applied as a flat rate on gains; no annual allowance.
- Mortgage term shorter than horizon is handled (payments stop at payoff).

## Deployment
Intended for Streamlit Community Cloud from this repo (`app.py`, branch
`main`). Repo can stay private if the app's viewer list is set in Cloud
settings.

## Changelog
- **2026-11-10** — Affordability tab: payment matrix by mortgage size, rate and
  term, budget check, future-year view and repayment over time.
- **2026-11-01** — Variable vs fixed historical test added to the Mortgage lab.
- **2026-10-30** — Mortgage lab tab: best-deposit sweep, repayment vs
  interest-only, and custom scenarios plotted against each other.
- **2026-10-24** — Live BoE mortgage rates (sidebar toggle + fixed-period
  comparison panel); stochastic mortgage rates with remortgaging every fixed
  period in MC, plus rate/payment band charts; stale zip removed.
- **2026-10-23** — Deposit up to 100% (cash purchase); MC now subtracts fund
  fees (previously ignored, overstating MC invest outcomes); theme config moved
  to `.streamlit/config.toml` so it actually applies; `.gitignore` added;
  `.DS_Store`/`__pycache__` untracked; tests added.
