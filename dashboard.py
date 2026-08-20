"""
HX-101 Digital Twin SCADA Dashboard
Enterprise-Grade Predictive Maintenance & Control Interface
────────────────────────────────────────────────────────────
Streams live telemetry from telemetry.db and renders:
  • 3-trace real-time degradation graph
  • Failure-threshold overlay at U = 280 W/m²K
  • 4 dynamic KPI cards with dropout prediction
  • Auto-refresh every 2 s via dcc.Interval
"""

import sqlite3
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from dash import Dash, dcc, html, Input, Output, no_update
import plotly.graph_objects as go

# ═══════════════════════════════════════════════════════════════════════
# CONSTANTS
# ═══════════════════════════════════════════════════════════════════════
DB_PATH = "telemetry.db"
FAILURE_THRESHOLD = 280.0          # W/m²K – critical dropout boundary
REFRESH_INTERVAL_MS = 2000         # 2-second auto-refresh

# ═══════════════════════════════════════════════════════════════════════
# COLOR PALETTE – Industrial Dark SCADA Theme
# ═══════════════════════════════════════════════════════════════════════
COLORS = {
    "bg_primary":    "#0f111a",     # App background
    "bg_card":       "#151929",     # KPI card fill
    "bg_graph":      "#131620",     # Plot area
    "border":        "#1f2640",     # Subtle card borders
    "border_glow":   "rgba(0, 210, 255, 0.12)",  # Cyan glow effect
    "text_primary":  "#e2e8f0",     # Main text
    "text_secondary":"#64748b",     # Muted labels
    "accent_cyan":   "#00d2ff",     # Accent / title
    "accent_green":  "#22c55e",     # Raw sensor trace
    "accent_yellow": "#facc15",     # PINN twin trace
    "accent_blue":   "#38bdf8",     # Forecast trace
    "accent_red":    "#ef4444",     # Failure / warning
    "grid":          "rgba(148, 163, 184, 0.06)",  # Faint grid
}

# ═══════════════════════════════════════════════════════════════════════
# STYLE TOKENS
# ═══════════════════════════════════════════════════════════════════════
FONT_STACK = "'Inter', 'Segoe UI', 'Roboto', system-ui, -apple-system, sans-serif"

PAGE_STYLE = {
    "backgroundColor": COLORS["bg_primary"],
    "color": COLORS["text_primary"],
    "fontFamily": FONT_STACK,
    "minHeight": "100vh",
    "padding": "0",
    "margin": "0",
}

HEADER_BAR_STYLE = {
    "background": "linear-gradient(135deg, #0f111a 0%, #151929 100%)",
    "borderBottom": f"1px solid {COLORS['border']}",
    "padding": "20px 40px",
    "display": "flex",
    "alignItems": "center",
    "justifyContent": "space-between",
}

TITLE_STYLE = {
    "color": COLORS["accent_cyan"],
    "fontSize": "22px",
    "fontWeight": "700",
    "letterSpacing": "1.5px",
    "textTransform": "uppercase",
    "margin": "0",
    "textShadow": f"0 0 20px rgba(0, 210, 255, 0.3)",
}

SUBTITLE_STYLE = {
    "color": COLORS["text_secondary"],
    "fontSize": "12px",
    "letterSpacing": "3px",
    "textTransform": "uppercase",
    "margin": "4px 0 0 0",
}

STATUS_DOT_STYLE = {
    "display": "inline-block",
    "width": "8px",
    "height": "8px",
    "borderRadius": "50%",
    "backgroundColor": COLORS["accent_green"],
    "marginRight": "8px",
    "boxShadow": f"0 0 8px {COLORS['accent_green']}",
    "animation": "pulse 2s infinite",
}

STATUS_TEXT_STYLE = {
    "color": COLORS["text_secondary"],
    "fontSize": "12px",
    "letterSpacing": "1px",
}

CARD_STYLE = {
    "backgroundColor": COLORS["bg_card"],
    "padding": "24px 20px",
    "borderRadius": "10px",
    "border": f"1px solid {COLORS['border']}",
    "boxShadow": f"0 0 15px {COLORS['border_glow']}, 0 4px 12px rgba(0,0,0,0.4)",
    "textAlign": "center",
    "flex": "1",
    "minWidth": "200px",
    "position": "relative",
    "overflow": "hidden",
    "transition": "box-shadow 0.3s ease, transform 0.2s ease",
}

CARD_LABEL_STYLE = {
    "color": COLORS["text_secondary"],
    "fontSize": "11px",
    "fontWeight": "600",
    "textTransform": "uppercase",
    "letterSpacing": "2px",
    "marginBottom": "12px",
}

CARD_VALUE_STYLE = {
    "fontSize": "30px",
    "fontWeight": "700",
    "color": COLORS["text_primary"],
    "lineHeight": "1.1",
}

CARD_UNIT_STYLE = {
    "fontSize": "12px",
    "color": COLORS["text_secondary"],
    "marginTop": "6px",
    "letterSpacing": "1px",
}

CONTENT_STYLE = {
    "padding": "24px 40px 40px 40px",
}

KPI_ROW_STYLE = {
    "display": "flex",
    "gap": "16px",
    "marginBottom": "20px",
    "flexWrap": "wrap",
}

GRAPH_CONTAINER_STYLE = {
    "borderRadius": "12px",
    "border": f"1px solid {COLORS['border']}",
    "boxShadow": f"0 0 20px {COLORS['border_glow']}, 0 8px 32px rgba(0,0,0,0.3)",
    "overflow": "hidden",
    "position": "relative",
}

FOOTER_STYLE = {
    "textAlign": "center",
    "color": COLORS["text_secondary"],
    "fontSize": "11px",
    "padding": "16px 40px",
    "borderTop": f"1px solid {COLORS['border']}",
    "letterSpacing": "1px",
}

# ═══════════════════════════════════════════════════════════════════════
# DATA ACCESS LAYER
# ═══════════════════════════════════════════════════════════════════════
def fetch_telemetry():
    """
    Read metrics + forecast from SQLite with bulletproof datetime parsing.
    Returns (metrics_df, forecast_df).  Both will have a proper
    DatetimeIndex-ready 'timestamp' column even if SQLite stored them as
    text.  Returns (None, None) on any failure so the callback can
    gracefully degrade.
    """
    try:
        conn = sqlite3.connect(DB_PATH)
        metrics = pd.read_sql(
            "SELECT * FROM metrics ORDER BY timestamp ASC", conn
        )
        forecast = pd.read_sql(
            "SELECT * FROM forecast ORDER BY timestamp ASC", conn
        )
        conn.close()

        if metrics.empty:
            return None, None

        # ── Robust timestamp coercion ──────────────────────────────
        # pd.to_datetime with utc=False and coerce guarantees we never
        # pass unparseable strings to Plotly's x-axis renderer.
        for df in (metrics, forecast):
            df["timestamp"] = pd.to_datetime(
                df["timestamp"],
                format="%Y-%m-%d %H:%M:%S",
                errors="coerce",
            )
            df.dropna(subset=["timestamp"], inplace=True)

        return metrics, forecast
    except Exception:
        return None, None

# ═══════════════════════════════════════════════════════════════════════
# APP INITIALIZATION
# ═══════════════════════════════════════════════════════════════════════
app = Dash(
    __name__,
    title="HX-101 SCADA | Digital Twin Control",
    update_title=None,   # Prevents "Updating…" in browser tab
)

# Inject CSS keyframes for the pulsing status dot
app.index_string = """<!DOCTYPE html>
<html>
<head>
    {%metas%}
    <title>{%title%}</title>
    {%favicon%}
    {%css%}
    <link rel="preconnect" href="https://fonts.googleapis.com">
    <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap" rel="stylesheet">
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body { overflow-x: hidden; }
        @keyframes pulse {
            0%, 100% { opacity: 1; }
            50% { opacity: 0.4; }
        }
        @keyframes shimmer {
            0% { background-position: -200% 0; }
            100% { background-position: 200% 0; }
        }
        /* Subtle top-accent glow bar on each card */
        .kpi-card::before {
            content: '';
            position: absolute;
            top: 0; left: 0; right: 0;
            height: 2px;
            background: linear-gradient(
                90deg,
                transparent 0%,
                rgba(0, 210, 255, 0.5) 50%,
                transparent 100%
            );
            background-size: 200% 100%;
            animation: shimmer 3s linear infinite;
        }
        .kpi-card:hover {
            box-shadow:
                0 0 24px rgba(0, 210, 255, 0.18),
                0 8px 24px rgba(0,0,0,0.5) !important;
            transform: translateY(-2px);
        }
        ::-webkit-scrollbar { width: 6px; }
        ::-webkit-scrollbar-track { background: #0f111a; }
        ::-webkit-scrollbar-thumb { background: #1f2640; border-radius: 4px; }
    </style>
</head>
<body>
    {%app_entry%}
    <footer>
        {%config%}
        {%scripts%}
        {%renderer%}
    </footer>
</body>
</html>"""

# ═══════════════════════════════════════════════════════════════════════
# LAYOUT
# ═══════════════════════════════════════════════════════════════════════
app.layout = html.Div(
    style=PAGE_STYLE,
    children=[
        # ── Header Bar ────────────────────────────────────────────
        html.Div(
            style=HEADER_BAR_STYLE,
            children=[
                html.Div([
                    html.H1(
                        "HX-101  ·  Predictive Maintenance & Digital Twin",
                        style=TITLE_STYLE,
                    ),
                    html.P(
                        "SCADA Supervisory Control  ·  Real-Time Monitoring",
                        style=SUBTITLE_STYLE,
                    ),
                ]),
                html.Div(
                    id="header-status",
                    style={"display": "flex", "alignItems": "center"},
                    children=[
                        html.Span(style=STATUS_DOT_STYLE),
                        html.Span("LIVE", style=STATUS_TEXT_STYLE),
                    ],
                ),
            ],
        ),

        # ── Main Content ──────────────────────────────────────────
        html.Div(
            style=CONTENT_STYLE,
            children=[
                # KPI Row
                html.Div(
                    id="live-kpis",
                    style=KPI_ROW_STYLE,
                ),

                # Graph
                html.Div(
                    style=GRAPH_CONTAINER_STYLE,
                    children=[
                        dcc.Graph(
                            id="live-graph",
                            style={"height": "62vh"},
                            config={
                                "displayModeBar": True,
                                "displaylogo": False,
                                "modeBarButtonsToRemove": [
                                    "lasso2d", "select2d",
                                ],
                            },
                        ),
                    ],
                ),
            ],
        ),

        # ── Footer ────────────────────────────────────────────────
        html.Div(
            style=FOOTER_STYLE,
            children="HX-101 DIGITAL TWIN ENGINE  ·  PINN-BACKED INFERENCE  ·  FAILURE FORECAST ACTIVE",
        ),

        # ── Timer (invisible) ────────────────────────────────────
        dcc.Interval(
            id="graph-update",
            interval=REFRESH_INTERVAL_MS,
            n_intervals=0,
        ),
    ],
)


# ═══════════════════════════════════════════════════════════════════════
# CALLBACKS
# ═══════════════════════════════════════════════════════════════════════
@app.callback(
    [Output("live-graph", "figure"), Output("live-kpis", "children")],
    [Input("graph-update", "n_intervals")],
)
def update_dashboard(n_intervals):
    """Fetch latest telemetry and rebuild figure + KPI cards."""

    metrics, forecast = fetch_telemetry()

    # ── Graceful empty-state ──────────────────────────────────────
    if metrics is None or metrics.empty:
        empty_fig = go.Figure()
        empty_fig.update_layout(
            template="plotly_dark",
            paper_bgcolor=COLORS["bg_graph"],
            plot_bgcolor=COLORS["bg_graph"],
            annotations=[
                dict(
                    text="WAITING FOR TELEMETRY STREAM …",
                    showarrow=False,
                    font=dict(size=18, color=COLORS["text_secondary"]),
                    xref="paper", yref="paper", x=0.5, y=0.5,
                )
            ],
        )
        placeholder_cards = _build_kpi_cards(
            op_hours="—", efficiency="—", rul="—", dropout="—",
            dropout_is_warning=False,
        )
        return empty_fig, placeholder_cards

    # ── Dropout / Threshold Prediction ────────────────────────────
    dropout_date_str = "STABLE"
    dropout_is_warning = False

    if forecast is not None and not forecast.empty:
        below = forecast[forecast["pinn_u_value"] <= FAILURE_THRESHOLD]
        if not below.empty:
            dropout_ts = below.iloc[0]["timestamp"]
            dropout_date_str = dropout_ts.strftime("%b %d, %Y")
            dropout_is_warning = True

    # ── Latest KPI values (absolute last row) ─────────────────────
    latest = metrics.iloc[-1]
    op_hours = f"{latest['operating_hours']:,.0f}"
    efficiency = f"{latest['thermal_efficiency']:.1f}%"
    rul_val = latest["rul_days"]
    rul_str = f"{rul_val:.0f}" if rul_val > 0 else "0"

    # ── Build Figure ──────────────────────────────────────────────
    fig = go.Figure()

    # Trace 1 – Raw Sensor U (green, thin, slight opacity)
    fig.add_trace(
        go.Scatter(
            x=metrics["timestamp"],
            y=metrics["raw_u_value"],
            mode="lines",
            name="Raw Sensor U",
            line=dict(color="rgba(34, 197, 94, 0.45)", width=1),
            hovertemplate="Raw: %{y:.1f} W/m²K<extra></extra>",
        )
    )

    # Trace 2 – PINN Digital Twin U (yellow, solid, width=2)
    fig.add_trace(
        go.Scatter(
            x=metrics["timestamp"],
            y=metrics["pinn_u_value"],
            mode="lines",
            name="PINN Digital Twin U",
            line=dict(color=COLORS["accent_yellow"], width=2),
            hovertemplate="PINN: %{y:.1f} W/m²K<extra></extra>",
        )
    )

    # Trace 3 – Degradation Forecast (blue, dashed, width=3)
    if forecast is not None and not forecast.empty:
        fig.add_trace(
            go.Scatter(
                x=forecast["timestamp"],
                y=forecast["pinn_u_value"],
                mode="lines",
                name="Degradation Forecast",
                line=dict(
                    color=COLORS["accent_blue"],
                    width=3,
                    dash="dash",
                ),
                hovertemplate="Forecast: %{y:.1f} W/m²K<extra></extra>",
            )
        )

    # ── Failure Threshold Line (red, dotted, bold) ────────────────
    fig.add_hline(
        y=FAILURE_THRESHOLD,
        line_dash="dot",
        line_color=COLORS["accent_red"],
        line_width=2.5,
        annotation_text=f"CRITICAL FAILURE THRESHOLD  ·  U = {FAILURE_THRESHOLD:.0f} W/m²K",
        annotation_position="bottom left",
        annotation_font=dict(
            color=COLORS["accent_red"],
            size=11,
            family=FONT_STACK,
        ),
    )

    # ── Layout ────────────────────────────────────────────────────
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor=COLORS["bg_graph"],
        plot_bgcolor=COLORS["bg_graph"],
        margin=dict(l=64, r=32, t=56, b=48),
        font=dict(family=FONT_STACK, color=COLORS["text_secondary"]),
        title=dict(
            text="HEAT TRANSFER COEFFICIENT  ·  REAL-TIME DEGRADATION MONITOR",
            font=dict(size=13, color=COLORS["text_secondary"]),
            x=0.5,
            xanchor="center",
        ),
        yaxis=dict(
            title=dict(
                text="U  (W/m²K)",
                font=dict(size=12, color=COLORS["text_secondary"]),
            ),
            gridcolor=COLORS["grid"],
            gridwidth=1,
            zeroline=False,
            showline=True,
            linecolor=COLORS["border"],
            linewidth=1,
        ),
        xaxis=dict(
            title=dict(
                text="Operational Timeline",
                font=dict(size=12, color=COLORS["text_secondary"]),
            ),
            gridcolor=COLORS["grid"],
            gridwidth=1,
            zeroline=False,
            showline=True,
            linecolor=COLORS["border"],
            linewidth=1,
            type="date",        # Force Plotly to treat axis as datetime
        ),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
            font=dict(size=11, color=COLORS["text_secondary"]),
            bgcolor="rgba(0,0,0,0)",
        ),
        hovermode="x unified",
        hoverlabel=dict(
            bgcolor=COLORS["bg_card"],
            bordercolor=COLORS["border"],
            font=dict(color=COLORS["text_primary"], family=FONT_STACK),
        ),
        # Prevent flickering on auto-refresh
        uirevision="constant",
    )

    # ── Build KPI Cards ───────────────────────────────────────────
    kpi_cards = _build_kpi_cards(
        op_hours=op_hours,
        efficiency=efficiency,
        rul=rul_str,
        dropout=dropout_date_str,
        dropout_is_warning=dropout_is_warning,
    )

    return fig, kpi_cards


# ═══════════════════════════════════════════════════════════════════════
# KPI CARD BUILDER
# ═══════════════════════════════════════════════════════════════════════
def _build_kpi_cards(
    op_hours: str,
    efficiency: str,
    rul: str,
    dropout: str,
    dropout_is_warning: bool,
):
    """Return a list of 4 KPI card Div components."""

    def _card(label, value, unit, value_color=COLORS["text_primary"], icon=""):
        return html.Div(
            className="kpi-card",
            style=CARD_STYLE,
            children=[
                html.Div(label, style=CARD_LABEL_STYLE),
                html.Div(
                    f"{icon}  {value}" if icon else value,
                    style={**CARD_VALUE_STYLE, "color": value_color},
                ),
                html.Div(unit, style=CARD_UNIT_STYLE),
            ],
        )

    dropout_color = COLORS["accent_red"] if dropout_is_warning else COLORS["accent_green"]
    dropout_icon = "⚠" if dropout_is_warning else "✓"

    return [
        _card(
            "Operating Hours",
            op_hours,
            "cumulative runtime",
            COLORS["accent_cyan"],
            "⏱",
        ),
        _card(
            "Twin Efficiency",
            efficiency,
            "thermal performance",
            COLORS["accent_yellow"],
            "⚡",
        ),
        _card(
            "RUL Forecast",
            f"{rul} days",
            "remaining useful life",
            COLORS["accent_blue"],
            "📉",
        ),
        _card(
            "Predicted Dropout",
            dropout,
            "failure projection",
            dropout_color,
            dropout_icon,
        ),
    ]


# ═══════════════════════════════════════════════════════════════════════
# ENTRY POINT
# ═══════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    app.run(debug=False, host="0.0.0.0", port=8050)