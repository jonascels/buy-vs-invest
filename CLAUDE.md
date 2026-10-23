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
| `test_model.py` | Pytest sanity tests. Run `pytest -q` before committing. |
| `.streamlit/config.toml` | Light theme. Theming lives here, **not** in CSS. |
| `environment.yml` | Conda env `buy-vs-invest` (py3.12, numpy, pandas, streamlit, plotly). |
| `buy-vs-invest-with-mc.zip` | Stale snapshot from an early build. Safe to delete. |

## Run
```bash
conda activate buy-vs-invest
streamlit run app.py
pytest -q
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
- **Horizons**: `[5, 10, 15, 20, 25, 30]` years (set in `Assumptions.horizons`).

## Monte Carlo
- `StochasticAssumptions` adds annual std devs for investment return, house
  appreciation and rent growth.
- Each path draws one normal shock per year, held constant across its 12
  months. Shocks are independent (no correlation between house prices and
  equities). Inflation and mortgage rate are fixed.
- User picks sim count (10–1,000); ~25 ms per sim.
- Outputs: win-probability cards, filled 10th–90th percentile bands with
  median lines, box plots per horizon, percentile + win-rate tables.

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
- No correlation between asset shocks; no stochastic mortgage rate (fixed-rate
  remortgage risk is ignored).
- No remortgaging, overpayments, ISA/pension tax wrappers, or rental income.
- CGT applied as a flat rate on gains; no annual allowance.
- Mortgage term shorter than horizon is handled (payments stop at payoff).

## Deployment
Intended for Streamlit Community Cloud from this repo (`app.py`, branch
`main`). Repo can stay private if the app's viewer list is set in Cloud
settings.

## Changelog
- **2026-10-23** — Deposit up to 100% (cash purchase); MC now subtracts fund
  fees (previously ignored, overstating MC invest outcomes); theme config moved
  to `.streamlit/config.toml` so it actually applies; `.gitignore` added;
  `.DS_Store`/`__pycache__` untracked; tests added.
