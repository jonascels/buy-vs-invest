"""
Print the variable vs fixed history test. Run: python analysis/variable_vs_fixed.py
The same test appears in the app's Mortgage tab (Variable or fixed?). The method is in history.py.
"""

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import history  # noqa: E402

pd.set_option("display.width", 230, "display.max_columns", 30)
d = history.fetch_history()
print(f"Data {d.index[0]:%b %Y} to {d.index[-1]:%b %Y}, {len(d)} months.\n")

for label, kw in (("Continuous", dict(method="continuous")),
                  ("Segments 2-year", dict(method="segments", block=24, n_paths=10_000)),
                  ("Segments 5-year", dict(method="segments", block=60, n_paths=10_000))):
    for years in (5, 10, 15, 20, 25):
        print(f"{label}, {years} years")
        print(history.summary(history.run_test(d, years, **kw)).drop(columns="key")
              .round(1).to_string(index=False), "\n")

deals = history.single_deals(d)
print(f"Single 2-year deals ({len(deals)}): variable cheaper {100 * (deals['gap'] > 0).mean():.0f}%")
print(deals.mean().round(2).to_string())
print(deals.groupby(deals.index.year // 5 * 5).mean().round(2).to_string())
