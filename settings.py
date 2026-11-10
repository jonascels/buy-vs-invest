"""
Save and load the app's inputs as a JSON file.

Every input has a stable session-state key. A snapshot of those values (plus the
editable tables) is refreshed at the end of each run and each Mortgage tab
fragment run, and the download button serialises it when clicked, so the file
reflects edits made in either place. Loading writes the values back into session
state from the uploader's callback, before the next run creates the widgets.

Inputs whose default follows other inputs (e.g. the long-run rate follows today's
rate) use `dynamic_key`: their key includes the default, so they keep following
it, until a loaded value pins them.
"""

from __future__ import annotations

import json
from datetime import datetime

import numpy as np
import pandas as pd
import streamlit as st

APP_ID = "buy-vs-invest"
FORMAT_VERSION = 1

# Inputs with a fixed key. Anything else in a loaded file is ignored.
STATIC_KEYS = [
    # sidebar
    "sb_mode", "sb_house_price", "sb_deposit", "sb_fixed_years", "sb_use_live", "sb_rate",
    "sb_term", "sb_appreciation", "sb_ftb", "sb_addl", "sb_maintenance", "sb_insurance",
    "sb_service", "sb_council_tax", "sb_legal", "sb_selling", "sb_rent", "sb_rent_growth",
    "sb_inv_return", "sb_inv_fee", "sb_cgt", "sb_inflation", "sb_real", "sb_n_sims",
    "sb_inv_vol", "sb_house_vol", "sb_rent_vol",
    # mortgage tab
    "fin_income", "fin_savings", "fin_income_growth", "fin_cost_growth",
    "aff_step", "aff_terms", "aff_rates", "aff_rate_step", "aff_year", "aff_show",
    "aff_start_rate", "aff_detail_amount", "aff_detail_rate", "aff_split_term",
    "cmp_horizon", "cmp_premium",
    "hist_loan", "hist_term", "hist_years", "hist_method",
]
DYNAMIC_KEYS = ["sb_rate_vol", "sb_long_run", "aff_amounts"]
TABLES = ["fin_costs", "cmp_scenarios"]

_SNAPSHOT = "_settings_snapshot"
_LOADED = "_settings_loaded"      # dynamic values waiting for their widget
_PINNED = "_settings_pinned"      # alias -> [key, default it was pinned at]
_CURRENT = "_settings_current"    # alias -> key used in this run
_TABLE_DATA = "_settings_tables"  # loaded table rows
_TABLE_LIVE = "_settings_table_values"
_VERSION = "_settings_version"
_MESSAGE = "_settings_message"


def _state(name: str, default):
    if name not in st.session_state:
        st.session_state[name] = default
    return st.session_state[name]


def dynamic_key(alias: str, default) -> str:
    """Key for an input whose default follows other inputs."""
    loaded, pinned = _state(_LOADED, {}), _state(_PINNED, {})
    if alias in loaded:
        key = f"{alias}__loaded{st.session_state.get(_VERSION, 0)}"
        st.session_state[key] = loaded.pop(alias)
        pinned[alias] = [key, default]
    if alias in pinned:
        key, pinned_default = pinned[alias]
        if pinned_default != default:  # the inputs it follows changed: follow again
            del pinned[alias]
    key = pinned[alias][0] if alias in pinned else f"{alias}__{default}"
    _state(_CURRENT, {})[alias] = key
    return key


def table(name: str, default: pd.DataFrame) -> tuple[pd.DataFrame, str]:
    """Starting rows and editor key for an editable table (loaded rows if any)."""
    rows = _state(_TABLE_DATA, {}).get(name)
    data = default
    if rows is not None:
        data = pd.DataFrame(rows, columns=default.columns)
        for col, dtype in default.dtypes.items():
            try:
                data[col] = data[col].astype(dtype)
            except (TypeError, ValueError):
                pass
    return data, f"{name}__v{st.session_state.get(_VERSION, 0)}"


def remember_table(name: str, df: pd.DataFrame) -> None:
    _state(_TABLE_LIVE, {})[name] = _records(df)


def _records(df: pd.DataFrame) -> list[dict]:
    clean = df.astype(object).where(pd.notna(df), None)
    return [{k: _plain(v) for k, v in row.items()} for row in clean.to_dict("records")]


def _plain(v):
    """JSON-friendly copy of a widget value."""
    if isinstance(v, (np.integer,)):
        return int(v)
    if isinstance(v, (np.floating,)):
        return float(v)
    if isinstance(v, (tuple, list)):
        return [_plain(x) for x in v]
    return v


def refresh_snapshot() -> None:
    """Copy current input values into the snapshot the download button reads."""
    snap = _state(_SNAPSHOT, {})
    widgets = {k: _plain(st.session_state[k]) for k in STATIC_KEYS if k in st.session_state}
    for alias, key in _state(_CURRENT, {}).items():
        if key in st.session_state:
            widgets[alias] = _plain(st.session_state[key])
    snap.clear()
    snap.update(widgets=widgets, tables=dict(_state(_TABLE_LIVE, {})))


def to_json(snap: dict) -> str:
    return json.dumps({"app": APP_ID, "version": FORMAT_VERSION,
                       "saved": datetime.now().isoformat(timespec="seconds"),
                       **snap}, indent=2)


def apply(data: dict, state=None) -> int:
    """Put a loaded file's values into session state. Returns inputs restored."""
    state = st.session_state if state is None else state
    if data.get("app") != APP_ID:
        raise ValueError("This isn't a settings file from this app.")
    if data.get("version", 0) > FORMAT_VERSION:
        raise ValueError("This file comes from a newer version of the app.")
    loaded = state[_LOADED] if _LOADED in state else {}
    count = 0
    for key, value in data.get("widgets", {}).items():
        if isinstance(value, list) and key != "aff_terms":
            value = tuple(value)  # range sliders
        if key in STATIC_KEYS:
            state[key] = value
            count += 1
        elif key in DYNAMIC_KEYS:
            loaded[key] = value
            count += 1
    state[_LOADED] = loaded
    tables = state[_TABLE_DATA] if _TABLE_DATA in state else {}
    new = {k: v for k, v in data.get("tables", {}).items() if k in TABLES}
    tables.update(new)
    state[_TABLE_DATA] = tables
    state[_VERSION] = (state[_VERSION] if _VERSION in state else 0) + 1
    return count + len(new)


def _on_upload() -> None:
    file = st.session_state.get("settings_file")
    if file is None:
        return
    try:
        count = apply(json.loads(file.getvalue().decode("utf-8")))
        st.session_state[_MESSAGE] = ("success", f"Loaded {count} settings from {file.name}.")
    except Exception as exc:  # bad JSON or wrong file: say so, keep current inputs
        st.session_state[_MESSAGE] = ("error", f"Couldn't load that file. {exc}")


def save_load_panel(container) -> None:
    """Save and load controls, for the top of the sidebar."""
    snap = _state(_SNAPSHOT, {})
    with container:
        st.subheader("Your settings")
        st.download_button(
            "Save settings", lambda: to_json(snap), file_name="buy-vs-invest-settings.json",
            mime="application/json", on_click="ignore", width="stretch",
            help="Downloads a small file with everything you've entered. Nothing is "
                 "stored online.")
        st.file_uploader("Load settings", type="json", key="settings_file",
                         on_change=_on_upload, max_upload_size=1,
                         help="Choose a file you saved earlier to restore it.")
        message = st.session_state.pop(_MESSAGE, None)
        if message:
            (st.success if message[0] == "success" else st.error)(message[1])
