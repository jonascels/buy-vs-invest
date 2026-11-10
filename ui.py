"""Shared Streamlit/Plotly helpers for the app's tabs."""

from __future__ import annotations

import plotly.graph_objects as go

# Fixed order, so a series keeps its colour as long as it keeps its position
SERIES_COLOURS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100",
                  "#e87ba4", "#008300", "#6250d6", "#e34948"]
CHART_LAYOUT = dict(template="plotly_white", hovermode="x unified",
                    margin=dict(l=10, r=10, t=10, b=10),
                    legend=dict(orientation="h", y=1.12),
                    font=dict(size=12, color="#333333"),
                    paper_bgcolor="white", plot_bgcolor="#fafafa")
BOX_LAYOUT = {k: v for k, v in CHART_LAYOUT.items() if k not in ("hovermode", "legend")}


def gbp(x: float) -> str:
    return f"-£{-x:,.0f}" if x < 0 else f"£{x:,.0f}"


def rgba(hex_colour: str, alpha: float) -> str:
    r, g, b = (int(hex_colour[i:i + 2], 16) for i in (1, 3, 5))
    return f"rgba({r}, {g}, {b}, {alpha})"


def add_band(fig: go.Figure, x, median, low, high, name: str, colour: str,
             hover: str = "£%{y:,.0f}") -> None:
    """Median line with a filled 10th-90th percentile band."""
    fig.add_trace(go.Scatter(x=x, y=high, mode="lines", line=dict(width=0),
                             legendgroup=name, showlegend=False, hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=x, y=low, mode="lines", line=dict(width=0),
                             fill="tonexty", fillcolor=rgba(colour, 0.14),
                             legendgroup=name, showlegend=False, hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=x, y=median, mode="lines", name=name, legendgroup=name,
                             line=dict(color=colour, width=2),
                             hovertemplate=f"{name}: {hover}<extra></extra>"))
