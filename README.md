# 🏠 Buy vs Invest

An interactive financial model that compares **buying a home with a mortgage**
against **renting and investing** the deposit plus any monthly cash-flow
difference. Outputs are graphical and span 5 / 10 / 20 / 30-year horizons.

![horizons](docs/screenshot.png)

## What it models

| Lever | Notes |
|---|---|
| House price, deposit %, mortgage rate & term | Standard amortising repayment mortgage |
| House appreciation | Annual nominal capital growth |
| Stamp Duty (SDLT) | England & N. Ireland bands (Apr 2025), incl. first-time-buyer relief and +5% additional-property surcharge |
| Running costs | Maintenance, insurance, service charge/ground rent, council tax |
| Transaction costs | Legal + survey (one-off), selling costs (% on exit) |
| Rent alternative | Monthly rent + growth rate |
| Investment | Expected return, fund fees (OCF), capital gains tax on exit |
| Economy | Inflation (for real-terms toggle) |

The renter invests the buyer's full upfront cash (deposit + SDLT + legal) and
each month invests the difference whenever buying costs more than renting.
Net worth is compared after selling costs and capital gains tax.

## Quick start

```bash
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
```

Then open the URL Streamlit prints (default http://localhost:8501).

## Project layout

```
model.py        Pure-Python engine (no UI dependency) — testable & reusable
app.py          Streamlit interface with sliders, inputs and Plotly charts
test_model.py   Sanity tests for the engine
requirements.txt
```

## Running the tests

```bash
python test_model.py        # or: pytest
```

## Caveats

This is a deterministic single-path model for **education, not financial
advice**. It ignores sequence-of-returns risk (try low/median/high return
scenarios), assumes constant rates, and uses England & N. Ireland SDLT —
Scotland (LBTT) and Wales (LTT) differ. Tax treatment is simplified.

## License

MIT — see [LICENSE](LICENSE).
