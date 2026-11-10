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
| `mortgage.py` | Streamlit UI for the Mortgage tab: finances, compare mortgages, variable or fixed. |
| `afford.py` | Finances panel and the affordability matrix (used by `mortgage.py`). |
| `settings.py` | Save and load all inputs as a JSON file (sidebar panel). |
| `ui.py` | Shared chart helpers, colours and `gbp` formatting. |
| `history.py` | Historical variable vs fixed test (no Streamlit). `analysis/variable_vs_fixed.py` prints it. |
| `rates.py` | Live UK mortgage rates from the Bank of England database (no Streamlit). |
| `test_model.py` | Pytest sanity tests. Run `python -m pytest -q` in the env before committing. |
| `.streamlit/config.toml` | Light theme (theming lives here, **not** in CSS); silences the widget-state duplication warning that loading settings would trigger. |
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
- **Savings** (`Assumptions.savings`, sidebar "Savings and investments", key
  `sb_savings`, shared with the Mortgage tab): savings left after the upfront
  purchase cash are invested on both paths at the same return (`_add_savings_pot`,
  net of CGT). `buy_net_worth` and `invest_net_worth` include the pot, so the gap
  and winner are unchanged; `buy_home_net_worth` keeps the home-only figure, shown
  as a third dotted curve ("Buy, home only"). Short savings show a warning.
  Old settings files with `fin_savings` load into `sb_savings`.
- **Overpayments** (`Assumptions.overpay_monthly`, `overpay_lumps` as (years, £);
  sidebar "Overpayments"): `overpayment_plan` builds the monthly extra (a lump
  "in year Y" lands at month Y*12). All schedules (`_amortisation_schedule`,
  `_variable_rate_schedule`, `repayment_schedule`) take it off the balance after
  the regular payment; the contractual payment is unchanged until payoff, except
  at a remortgage, where it's recalculated on the smaller balance. Overpayments
  count as buyer outgoings, so the renter invests the same extra (fair). The
  tab shows years saved and interest saved, and warns above 10% of the loan in
  year one. The Mortgage tab detail view has its own monthly and lump-sum
  inputs; the matrix and the Compare mortgages engine don't model overpayments.
- **Rent step-ups** (`Assumptions.rent_steps`, sidebar "Bigger home later?" table):
  (years from now, extra £/month at today's prices). Rent each month =
  `rent_today_prices(a, year)` x cumulative rent growth, so a step is indexed
  from today. Applies in deterministic and Monte Carlo, so it cuts what the
  renter can invest. The Mortgage tab doesn't use rent.
- **Rent vs mortgage chart** (deterministic mode): rent with steps, mortgage
  repayment, and repayment + upkeep/insurance; totals over a chosen horizon;
  cumulative "money you don't get back" (rent vs interest + upkeep + insurance
  + service + SDLT + legal). Replaced the old "Monthly outgoings" chart.
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

## Mortgage tab (second tab)
One tab, a `@st.fragment` (`render_mortgage_tab`), so edits don't re-run the first
tab. A shared **Your finances** panel (`finances_panel` -> `Finances`) feeds all
three sections: take-home income, a costs table with optional start and stop
years (e.g. school fees starting in 5, childcare stopping in 4), savings, pay
rises and cost growth. Budget = income minus active costs (`monthly_budget`,
`budget_path`). Costs are in today's money and grow from today.

**What can we afford?** (`affordability_section`)
- Replicates the owner's Excel: payment matrix with mortgage sizes as rows and
  rates as columns, one per term (default 20/25/30), plus cash needed, LTV and
  share of savings. Cash needed = deposit + SDLT + legal fees. Default rows start
  at the smallest loan the savings allow.
- Cells: ✓ fits, ! within 10%, ✗ over (kept as plain symbols so status isn't
  colour-only); greyed rows need more cash than savings.
- "In N years": pays a starting rate for N years, then remortgages at the column
  rate (`mortgage_scenario`, closed form). Same rate means the same payment.
- Detail view: payment, total interest, total repaid, mortgage-free year per
  term; payment over time against the budget; balance; interest vs principal.

**Compare mortgages** (`compare_section`)
- Engine: `Scenario`, `simulate_scenario`, `run_scenarios`, `compare_scenarios` in
  `model.py`, NumPy-vectorised, fixed at 1,000 sims.
- Uses the finances: cash = savings, monthly budget = `budget_path` (so costs
  that start or stop change it), with `include_running_costs=False` because the
  costs list already covers upkeep and council tax.
- Deposit sweep: every deposit from 5% to the most savings allow, repayment and
  interest-only, sidebar fix and term; cards + one gain/loss chart vs the
  smallest repayment deposit. Then an editable table of own options (up to 8):
  cards, net worth bands over time and box plots. Tables sit in expanders.
- Interest-only: balance flat, due at term end from investments net of CGT.
  Shortfalls become funding-gap debt ("Ran out of cash %"). Rate = live rate +
  a premium (default 0.25pp, our assumption).
- Scenarios share shocks (`draw_shocks`), so "chance best" is path by path.
- Each deposit uses its own LTV band rate (only 2- and 5-yr fixes vary; nothing
  above 90% LTV).

**Variable or fixed?** (`variable_vs_fixed`, method in `history.py`)
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
- Its own loan and term inputs; the finances don't apply.

## Saving settings (`settings.py`)
- "Save settings" at the top of the sidebar downloads a JSON file of every input
  plus the costs and scenario tables; "Load settings" restores it. Nothing is
  stored on a server.
- **Every new input needs a stable `key` added to `STATIC_KEYS`**, or it won't be
  saved. Inputs whose default follows other inputs use `settings.dynamic_key`
  (long-run rate, rate volatility, mortgage size range). Editable tables use
  `settings.table` + `settings.remember_table`.
- The download builds its file on click from a snapshot refreshed at the end of
  every run and every Mortgage tab fragment run (`refresh_snapshot`), so edits
  made only inside the fragment are included.
- Loading runs in the uploader's `on_change` callback, writing into session state
  before widgets are created; editor keys get a version bump so tables reload.
- A live mortgage rate isn't saved (it's re-fetched); a manual rate is.

## UI decisions the owner has made (don't undo without asking)
- Light theme via `config.toml`. Heavy `!important` CSS overrides broke table
  rendering; keep CSS minimal.
- Keep **box-and-whisker** plots for distributions (histograms were tried and
  rejected).
- Keep the **filled uncertainty band** chart.
- Buy = blue `#1e88e5`, Invest = orange `#f57c00`.
- No emojis or icons in the UI (labels, tabs, captions, page icon). Metric deltas
  use `delta_arrow="off"`.
- Mortgage tab colours are the dataviz skill's reference categorical palette, in
  order (`ui.SERIES_COLOURS`).
- Default house price £1M.

## Known limitations / ideas
- `run_model` and `run_model_stochastic` duplicate most of their loop. A
  refactor to one engine taking per-month growth arrays would remove that.
- Simulation loops are pure Python with `iterrows`; vectorising across sims
  with NumPy would make 1,000+ sims near-instant.
- No correlation between asset shocks (e.g. rates vs house prices).
- Lender rules (max LTV for interest-only, affordability) aren't enforced.
- No remortgage fees, ISA/pension tax wrappers, or rental income.
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
- **2026-11-10** — Mortgage overpayments (monthly and lump sums) in Buy vs invest
  and the Mortgage tab detail view.
- **2026-11-10** — Savings beyond the purchase invested on both paths; third
  "home only" curve; savings input moved to the sidebar for both tabs.
- **2026-11-10** — Rent vs mortgage trend chart; rent step-ups for a bigger home.
- **2026-11-10** — Save and load settings as a file.
- **2026-11-10** — Affordability and Mortgage lab merged into one Mortgage tab with
  shared finances (costs can start and stop); comparison simplified and now uses
  the same budget; emojis removed; default house price £1M.
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
