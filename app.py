import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent / "src"))

import json
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from forecast import generate_forecast, get_recent_accuracy

# ─────────────────────────────────────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Demand Planning · Supply Chain",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────────────────────
# Design tokens
# ─────────────────────────────────────────────────────────────────────────────
C = dict(
    navy="#0F1E35", blue="#2563EB", green="#16A34A",
    amber="#D97706", red="#DC2626", slate="#64748B",
    border="#E2E8F0", bg="#F1F5F9", white="#FFFFFF",
    navy_lt="#EFF6FF", green_lt="#F0FDF4",
)


def chart(**overrides) -> dict:
    """Return a Plotly layout dict, safely merging any overrides."""
    base = dict(
        paper_bgcolor=C["white"],
        plot_bgcolor=C["white"],
        font=dict(family="Inter, sans-serif", size=11, color="#334155"),
        hovermode="x unified",
        hoverlabel=dict(
            bgcolor=C["navy"], font_color="white",
            font_size=12, bordercolor=C["navy"], namelength=-1,
        ),
        legend=dict(
            orientation="h", yanchor="bottom", y=1.02,
            xanchor="left", x=0, bgcolor="rgba(0,0,0,0)",
            font=dict(size=11),
        ),
        xaxis=dict(
            showgrid=False, zeroline=False,
            showline=True, linecolor=C["border"],
            tickfont=dict(size=10),
        ),
        yaxis=dict(
            showgrid=True, gridcolor="#F1F5F9",
            zeroline=False, tickfont=dict(size=10),
        ),
        margin=dict(t=24, b=40, l=12, r=12),
    )
    return {**base, **overrides}


# ─────────────────────────────────────────────────────────────────────────────
# CSS
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
@import url('https://fonts.googleapis.com/icon?family=Material+Icons+Round');

*, html, body, [class*="css"] { font-family: 'Inter', sans-serif !important; }
#MainMenu, footer, header { visibility: hidden; }

/* Fix Streamlit sidebar collapse button showing raw icon text */
[data-testid="stSidebarCollapsedControl"] { display: none !important; }
[data-testid="stSidebarCollapseButton"] .material-icons-round,
button[kind="header"] span { font-family: 'Material Icons Round' !important; }
.stApp { background: #F1F5F9; }
.main .block-container { padding: 0 1.75rem 2rem !important; max-width: 100% !important; }

/* Sidebar */
[data-testid="stSidebar"] { background: #0F1E35 !important; border-right: 1px solid #1E3A5F; }
[data-testid="stSidebar"] * { color: #94A3B8 !important; }
[data-testid="stSidebar"] strong { color: #CBD5E1 !important; }
[data-testid="stSidebar"] hr { border-color: #1E3A5F !important; margin: 0.6rem 0 !important; }
[data-testid="stSidebar"] label {
    color: #475569 !important; font-size: 0.7rem !important;
    font-weight: 700 !important; letter-spacing: 0.08em !important;
    text-transform: uppercase !important;
}

/* Tabs */
.stTabs [data-baseweb="tab-list"] {
    gap: 0; border-bottom: 1px solid #E2E8F0; background: transparent;
}
.stTabs [data-baseweb="tab"] {
    background: transparent; border: none;
    border-bottom: 2px solid transparent;
    color: #64748B; font-size: 0.82rem; font-weight: 500;
    padding: 0.65rem 1.3rem;
}
.stTabs [aria-selected="true"] {
    background: transparent !important;
    border-bottom: 2px solid #2563EB !important;
    color: #0F1E35 !important; font-weight: 600 !important;
}

/* Cards */
.card {
    background: #FFFFFF; border: 1px solid #E2E8F0; border-radius: 10px;
    box-shadow: 0 1px 3px rgba(0,0,0,0.06), 0 1px 2px rgba(0,0,0,0.04);
    padding: 1.3rem 1.4rem; margin-bottom: 0.1rem;
}

/* KPI */
.kpi {
    background: #FFFFFF; border: 1px solid #E2E8F0; border-radius: 10px;
    border-left-width: 4px; border-left-style: solid;
    box-shadow: 0 1px 3px rgba(0,0,0,0.06);
    padding: 1.1rem 1.25rem; height: 100%;
}
.kpi-lbl {
    font-size: 0.65rem; font-weight: 700; letter-spacing: 0.1em;
    text-transform: uppercase; color: #94A3B8; margin-bottom: 0.35rem;
}
.kpi-val { font-size: 1.8rem; font-weight: 700; color: #0F1E35; line-height: 1.1; }
.kpi-sub { font-size: 0.71rem; color: #94A3B8; margin-top: 0.3rem; }

/* Section label */
.slabel {
    font-size: 0.65rem; font-weight: 700; letter-spacing: 0.1em;
    text-transform: uppercase; color: #94A3B8;
    padding-bottom: 0.5rem; border-bottom: 1px solid #F1F5F9;
    margin-bottom: 0.9rem;
}

/* Badges */
.badge {
    display: inline-flex; align-items: center; gap: 5px;
    padding: 0.15rem 0.6rem; border-radius: 20px;
    font-size: 0.68rem; font-weight: 600; line-height: 1.5;
}
.b-green { background: #DCFCE7; color: #166534; }
.b-amber { background: #FEF9C3; color: #854D0E; }
.b-red   { background: #FEE2E2; color: #991B1B; }
.b-blue  { background: #DBEAFE; color: #1D4ED8; }
.dot::before {
    content: ''; display: inline-block; width: 6px; height: 6px;
    border-radius: 50%; background: currentColor;
}

/* Data table */
.dt { width: 100%; border-collapse: collapse; font-size: 0.79rem; }
.dt thead tr { border-bottom: 2px solid #E2E8F0; }
.dt th {
    padding: 0.45rem 0.7rem; text-align: left;
    font-size: 0.63rem; font-weight: 700;
    letter-spacing: 0.09em; text-transform: uppercase; color: #94A3B8;
}
.dt th.r, .dt td.r { text-align: right; }
.dt th.c, .dt td.c { text-align: center; }
.dt tbody tr { border-bottom: 1px solid #F8FAFC; }
.dt tbody tr:hover { background: #F8FAFC; }
.dt td { padding: 0.5rem 0.7rem; color: #334155; }
.dt .mono { font-variant-numeric: tabular-nums; font-weight: 500; }
.dt .hi   { color: #166534; font-weight: 600; background: #F0FDF4; }

/* Topbar */
.topbar {
    background: white; border-bottom: 1px solid #E2E8F0;
    padding: 0.9rem 0; margin-bottom: 1.4rem;
    display: flex; align-items: center; justify-content: space-between;
}

/* Info box */
.infobox {
    background: #F8FAFC; border: 1px solid #E2E8F0; border-radius: 8px;
    padding: 0.85rem 1rem; font-size: 0.77rem; color: #475569;
    line-height: 1.7; margin-top: 0.75rem;
}

/* Container card (border=True override) */
[data-testid="stVerticalBlockBorderWrapper"] {
    background: white !important;
    border: 1px solid #E2E8F0 !important;
    border-radius: 10px !important;
    box-shadow: 0 1px 3px rgba(0,0,0,0.06), 0 1px 2px rgba(0,0,0,0.04) !important;
    padding: 1.1rem 1.3rem !important;
}

/* Download button */
.stDownloadButton > button {
    background: white !important; color: #0F1E35 !important;
    border: 1px solid #CBD5E1 !important; border-radius: 6px !important;
    font-size: 0.79rem !important; font-weight: 500 !important;
    padding: 0.5rem 1.1rem !important; width: 100% !important;
    transition: border-color 0.15s, box-shadow 0.15s !important;
}
.stDownloadButton > button:hover {
    border-color: #94A3B8 !important;
    box-shadow: 0 1px 4px rgba(0,0,0,0.08) !important;
}
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# Guard
# ─────────────────────────────────────────────────────────────────────────────
META_PATH = Path("models/metadata.json")
if not META_PATH.exists():
    st.error("Model not found. Run `python src/train.py` first.")
    st.stop()

meta = json.loads(META_PATH.read_text())

# ─────────────────────────────────────────────────────────────────────────────
# Sidebar
# ─────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style="padding:1.1rem 1.4rem 1rem;border-bottom:1px solid #1E3A5F;">
      <div style="font-size:0.58rem;font-weight:700;letter-spacing:0.12em;
                  text-transform:uppercase;color:#475569;margin-bottom:0.5rem;">
        Forecast Settings
      </div>
    </div>
    """, unsafe_allow_html=True)

    service_level = st.selectbox(
        "Service level",
        options=[0.99, 0.95, 0.90],
        index=1,
        format_func=lambda x: f"{int(x*100)}%",
    )
    n_weeks = st.slider("Horizon (weeks)", min_value=4, max_value=16, value=12, step=2)

    z_map = {0.90: 1.282, 0.95: 1.645, 0.99: 2.326}
    z     = z_map[service_level]
    rmse  = meta["holdout_metrics"]["rmse"]
    smape = meta["holdout_metrics"]["smape"]
    buf   = round(z * rmse)

    health_cls = "b-green dot" if smape < 5 else "b-amber dot"
    health_txt = "Healthy" if smape < 5 else "Degraded"

    st.markdown("---")
    st.markdown(f"""
    <div style="padding:0 1.4rem 1.4rem;">
      <div style="font-size:0.58rem;font-weight:700;letter-spacing:0.12em;
                  text-transform:uppercase;color:#334155;margin-bottom:0.6rem;">
        Model Status
      </div>
      <div style="margin-bottom:0.75rem;">
        <span class="badge {health_cls}">{health_txt}</span>
      </div>
      <table style="width:100%;border-collapse:collapse;font-size:0.73rem;">
        <tr><td style="color:#475569;padding:0.22rem 0">Algorithm</td>
            <td style="color:#CBD5E1;text-align:right;font-weight:500">XGBoost</td></tr>
        <tr><td style="color:#475569;padding:0.22rem 0">Trained</td>
            <td style="color:#CBD5E1;text-align:right;font-weight:500">{meta['trained_at'][:10]}</td></tr>
        <tr><td style="color:#475569;padding:0.22rem 0">Data through</td>
            <td style="color:#CBD5E1;text-align:right;font-weight:500">{meta['last_data_date']}</td></tr>
        <tr><td style="color:#475569;padding:0.22rem 0">Training weeks</td>
            <td style="color:#CBD5E1;text-align:right;font-weight:500">{meta['training_weeks']}</td></tr>
      </table>
      <div style="margin-top:0.85rem;padding-top:0.85rem;border-top:1px solid #1E3A5F;">
        <div style="font-size:0.58rem;font-weight:700;letter-spacing:0.12em;
                    text-transform:uppercase;color:#334155;margin-bottom:0.5rem;">
          Holdout Performance
        </div>
        <table style="width:100%;border-collapse:collapse;font-size:0.73rem;">
          <tr><td style="color:#475569;padding:0.22rem 0">SMAPE</td>
              <td style="color:#CBD5E1;text-align:right;font-weight:600">{smape:.2f}%</td></tr>
          <tr><td style="color:#475569;padding:0.22rem 0">MAE</td>
              <td style="color:#CBD5E1;text-align:right;font-weight:600">{meta['holdout_metrics']['mae']:.0f} units</td></tr>
          <tr><td style="color:#475569;padding:0.22rem 0">RMSE</td>
              <td style="color:#CBD5E1;text-align:right;font-weight:600">{rmse:.0f} units</td></tr>
          <tr><td style="color:#475569;padding:0.22rem 0">Safety stock</td>
              <td style="color:#CBD5E1;text-align:right;font-weight:600">{buf:,} units</td></tr>
        </table>
      </div>
    </div>
    """, unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# Data
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_data
def load_forecast(n, sl):
    return generate_forecast(n_weeks=n, service_level=sl)

@st.cache_data
def load_accuracy():
    return get_recent_accuracy(n_weeks=12)

@st.cache_data
def load_history():
    return pd.read_csv("models/training_series.csv", parse_dates=["ds"])

fc = load_forecast(n_weeks, service_level)
acc  = load_accuracy()
hist = load_history()

# Derived
nw            = fc.iloc[0]
four_wk       = int(fc.iloc[:4]["forecast"].sum())
acc_rate      = float((np.abs(acc["error_pct"]) < 5).mean() * 100)
avg_err       = float(np.abs(acc["error_pct"]).mean())
acc_cls       = "b-green" if acc_rate >= 75 else "b-amber" if acc_rate >= 50 else "b-red"
acc_label     = "On Track" if acc_rate >= 75 else "Watch" if acc_rate >= 50 else "At Risk"
kpi_acc_color = C["green"] if acc_rate >= 75 else C["amber"] if acc_rate >= 50 else C["red"]

# ─────────────────────────────────────────────────────────────────────────────
# Page header
# ─────────────────────────────────────────────────────────────────────────────
st.markdown(f"""
<div style="background:linear-gradient(135deg,#0F1E35 0%,#1E3A5F 100%);
            border-radius:12px;padding:1.75rem 2rem 1.6rem;
            margin-bottom:1.5rem;display:flex;align-items:center;
            justify-content:space-between;
            box-shadow:0 4px 16px rgba(15,30,53,0.18);">
  <div>
    <div style="font-size:0.6rem;font-weight:700;letter-spacing:0.2em;
                text-transform:uppercase;color:#60A5FA;margin-bottom:0.5rem;">
      Supply Chain Analytics
    </div>
    <div style="font-size:1.9rem;font-weight:700;color:#F8FAFC;line-height:1.1;
                letter-spacing:-0.025em;">
      Demand Planning
    </div>
    <div style="font-size:0.78rem;color:#94A3B8;margin-top:0.55rem;letter-spacing:0.01em;">
      {int(service_level*100)}% service level &nbsp;·&nbsp;
      {n_weeks}-week horizon &nbsp;·&nbsp;
      Data through {meta['last_data_date']}
    </div>
  </div>
  <div style="text-align:right;">
    <div style="display:inline-flex;align-items:center;gap:0.5rem;
                background:rgba(22,163,74,0.15);border:1px solid rgba(22,163,74,0.35);
                border-radius:20px;padding:0.35rem 0.9rem;">
      <span style="width:7px;height:7px;border-radius:50%;
                   background:#4ADE80;display:inline-block;"></span>
      <span style="font-size:0.73rem;font-weight:600;color:#4ADE80;">Model Healthy</span>
    </div>
    <div style="font-size:0.71rem;color:#64748B;margin-top:0.55rem;">
      XGBoost &nbsp;·&nbsp; SMAPE {smape:.2f}% &nbsp;·&nbsp; Trained {meta['trained_at'][:10]}
    </div>
  </div>
</div>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# KPI strip
# ─────────────────────────────────────────────────────────────────────────────
k1, k2, k3, k4 = st.columns(4, gap="small")

k1.markdown(f"""
<div class="kpi" style="border-left-color:{C['blue']}">
  <div class="kpi-lbl">Next Week Forecast</div>
  <div class="kpi-val">{nw['forecast']:,}</div>
  <div class="kpi-sub">units &nbsp;·&nbsp; w/e {pd.Timestamp(nw['week']).strftime('%d %b %Y')}</div>
</div>""", unsafe_allow_html=True)

k2.markdown(f"""
<div class="kpi" style="border-left-color:{C['green']}">
  <div class="kpi-lbl">Recommended Order Qty</div>
  <div class="kpi-val">{nw['recommended_order']:,}</div>
  <div class="kpi-sub">incl. {buf:,}-unit safety stock &nbsp;·&nbsp; {int(service_level*100)}% SL</div>
</div>""", unsafe_allow_html=True)

k3.markdown(f"""
<div class="kpi" style="border-left-color:{C['navy']}">
  <div class="kpi-lbl">4-Week Demand Outlook</div>
  <div class="kpi-val">{four_wk:,}</div>
  <div class="kpi-sub">
    {pd.Timestamp(fc['week'].iloc[0]).strftime('%d %b')} –
    {pd.Timestamp(fc['week'].iloc[3]).strftime('%d %b %Y')}
  </div>
</div>""", unsafe_allow_html=True)

k4.markdown(f"""
<div class="kpi" style="border-left-color:{kpi_acc_color}">
  <div class="kpi-lbl">Forecast Accuracy (12w)</div>
  <div class="kpi-val">{acc_rate:.0f}%</div>
  <div class="kpi-sub">
    within ±5% &nbsp;
    <span class="badge {acc_cls}">{acc_label}</span>
  </div>
</div>""", unsafe_allow_html=True)

st.markdown("<div style='height:1.2rem'></div>", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# Tabs
# ─────────────────────────────────────────────────────────────────────────────
tab1, tab2, tab3 = st.tabs(["  Demand Forecast  ", "  Recent Performance  ", "  Export  "])

# ═════════════════════════════════════════════════════════════════════════════
# TAB 1 — Demand Forecast
# ═════════════════════════════════════════════════════════════════════════════
with tab1:
    col_l, col_r = st.columns([5, 3], gap="medium")

    # ── Forecast chart ────────────────────────────────────────────────────────
    with col_l:
        with st.container(border=True):
            st.markdown(
                f'<div class="slabel">Forward Demand Forecast · Next {n_weeks} Weeks</div>',
                unsafe_allow_html=True,
            )

            tail  = hist.iloc[-20:]
            div_x = hist["ds"].iloc[-1]
            y_lo  = min(tail["y"].min(), fc["lower_bound"].min()) * 0.96
            y_hi  = fc["upper_bound"].max() * 1.04

            fig1 = go.Figure()
            fig1.add_trace(go.Scatter(
                x=tail["ds"], y=tail["y"],
                mode="lines", name="Historical demand",
                line=dict(color=C["navy"], width=2.2),
                fill="tozeroy", fillcolor="rgba(15,30,53,0.04)",
                hovertemplate="<b>%{x|%d %b %Y}</b><br>Actual: %{y:,}<extra></extra>",
            ))
            x_band = list(fc["week"]) + list(reversed(list(fc["week"])))
            y_band = list(fc["upper_bound"]) + list(reversed(list(fc["lower_bound"])))
            fig1.add_trace(go.Scatter(
                x=x_band, y=y_band,
                fill="toself", fillcolor="rgba(37,99,235,0.08)",
                line=dict(color="rgba(0,0,0,0)"),
                name=f"{int(service_level*100)}% CI",
                hoverinfo="skip", showlegend=True,
            ))
            fig1.add_trace(go.Scatter(
                x=fc["week"], y=fc["forecast"],
                mode="lines+markers", name="Forecast",
                line=dict(color=C["blue"], width=2, dash="dash"),
                marker=dict(size=5, color="white", symbol="circle",
                            line=dict(color=C["blue"], width=2)),
                hovertemplate="<b>%{x|%d %b %Y}</b><br>Forecast: %{y:,}<extra></extra>",
            ))
            fig1.add_trace(go.Scatter(
                x=fc["week"], y=fc["recommended_order"],
                mode="markers", name="Order qty",
                marker=dict(size=8, color=C["green"], symbol="triangle-up",
                            line=dict(color="white", width=1.5)),
                hovertemplate="<b>%{x|%d %b %Y}</b><br>Order qty: %{y:,}<extra></extra>",
            ))
            fig1.add_vline(x=div_x, line_color="#CBD5E1", line_width=1, line_dash="dot")
            fig1.add_annotation(
                x=div_x, y=0.96, yref="paper", xshift=48,
                text="Forecast →", showarrow=False,
                font=dict(size=10, color=C["slate"]),
            )
            fig1.update_layout(**chart(
                height=360,
                yaxis_title="Weekly order units",
                yaxis=dict(showgrid=True, gridcolor="#F1F5F9", zeroline=False,
                           tickfont=dict(size=10), tickformat=",d",
                           range=[y_lo, y_hi]),
            ))
            st.plotly_chart(fig1, use_container_width=True, config={"displayModeBar": False})

    # ── Order plan table ──────────────────────────────────────────────────────
    with col_r:
        with st.container(border=True):
            st.markdown('<div class="slabel">Week-by-Week Order Plan</div>', unsafe_allow_html=True)
            rows = "".join(f"""
            <tr>
              <td class="mono">{pd.Timestamp(r['week']).strftime('%d %b')}</td>
              <td class="mono r">{r['forecast']:,}</td>
              <td class="mono r" style="color:{C['slate']};font-size:0.74rem;">
                {r['lower_bound']:,}–{r['upper_bound']:,}</td>
              <td class="mono r hi">{r['recommended_order']:,}</td>
            </tr>""" for _, r in fc.iterrows())
            st.markdown(f"""
            <table class="dt">
              <thead><tr>
                <th>Week of</th><th class="r">Forecast</th>
                <th class="r">Range</th><th class="r">Order Qty</th>
              </tr></thead>
              <tbody>{rows}</tbody>
            </table>
            """, unsafe_allow_html=True)

        st.markdown("<div style='height:0.6rem'></div>", unsafe_allow_html=True)
        st.markdown(f"""
        <div class="infobox">
          <strong>Safety stock formula</strong><br>
          Order Qty = Forecast + z × RMSE<br>
          = Forecast + {z} × {rmse:.0f} = Forecast + <strong>{buf:,} units</strong><br>
          <span style="color:#94A3B8;">
            At {int(service_level*100)}% SL, this covers demand
            variability {int(service_level*100)}% of weeks.
          </span>
        </div>
        """, unsafe_allow_html=True)

# ═════════════════════════════════════════════════════════════════════════════
# TAB 2 — Recent Performance
# ═════════════════════════════════════════════════════════════════════════════
with tab2:
    col_l, col_r = st.columns([5, 3], gap="medium")

    with col_l:
        with st.container(border=True):
            st.markdown('<div class="slabel">Actual vs Forecast — Last 12 Weeks</div>',
                        unsafe_allow_html=True)

            fig2 = go.Figure()
            fig2.add_trace(go.Scatter(
                x=list(acc["week"]) + list(reversed(list(acc["week"]))),
                y=list(acc["actual"]) + list(reversed(list(acc["forecast"]))),
                fill="toself", fillcolor="rgba(37,99,235,0.06)",
                line=dict(color="rgba(0,0,0,0)"),
                hoverinfo="skip", showlegend=False,
            ))
            fig2.add_trace(go.Scatter(
                x=acc["week"], y=acc["actual"],
                mode="lines+markers", name="Actual",
                line=dict(color=C["navy"], width=2.5),
                marker=dict(size=7, color=C["navy"], line=dict(color="white", width=1.5)),
                hovertemplate="<b>%{x|%d %b}</b><br>Actual: %{y:,}<extra></extra>",
            ))
            fig2.add_trace(go.Scatter(
                x=acc["week"], y=acc["forecast"],
                mode="lines+markers", name="Forecast",
                line=dict(color=C["blue"], width=1.8, dash="dash"),
                marker=dict(size=6, color="white", symbol="circle",
                            line=dict(color=C["blue"], width=2)),
                hovertemplate="<b>%{x|%d %b}</b><br>Forecast: %{y:,}<extra></extra>",
            ))
            fig2.update_layout(**chart(
                height=300,
                yaxis_title="Weekly order units",
                yaxis=dict(showgrid=True, gridcolor="#F1F5F9", zeroline=False,
                           tickfont=dict(size=10), tickformat=",d"),
            ))
            st.plotly_chart(fig2, use_container_width=True, config={"displayModeBar": False})

            st.markdown('<div class="slabel" style="margin-top:0.5rem">Weekly Forecast Error (%)</div>',
                        unsafe_allow_html=True)
            bar_c = [C["green"] if abs(e) < 3 else C["amber"] if abs(e) < 7 else C["red"]
                     for e in acc["error_pct"]]
            fig3 = go.Figure()
            fig3.add_trace(go.Bar(
                x=acc["week"], y=acc["error_pct"],
                marker_color=bar_c,
                text=[f"{v:+.1f}%" for v in acc["error_pct"]],
                textposition="outside",
                textfont=dict(size=9, color="#475569"),
                hovertemplate="<b>%{x|%d %b}</b><br>Error: %{y:+.1f}%<extra></extra>",
            ))
            fig3.add_hline(y=0, line_color="#CBD5E1", line_width=1.5)
            fig3.add_hrect(
                y0=-5, y1=5, fillcolor="rgba(22,163,74,0.05)", line_width=0,
                annotation_text="±5% target", annotation_font_size=9,
                annotation_font_color=C["green"], annotation_position="top right",
            )
            fig3.update_layout(**chart(
                height=190,
                showlegend=False,
                margin=dict(t=10, b=40, l=12, r=40),
                yaxis=dict(showgrid=True, gridcolor="#F1F5F9", zeroline=False,
                           tickformat="+.0f", ticksuffix="%", tickfont=dict(size=10)),
            ))
            st.plotly_chart(fig3, use_container_width=True, config={"displayModeBar": False})
            st.markdown(
                '<div style="font-size:0.71rem;color:#94A3B8;padding-top:0.3rem;">'
                f'<span style="color:{C["green"]};">&#9679;</span> &lt;3% error &nbsp;'
                f'<span style="color:{C["amber"]};">&#9679;</span> 3–7% &nbsp;'
                f'<span style="color:{C["red"]};">&#9679;</span> &gt;7%'
                '</div>',
                unsafe_allow_html=True,
            )

    # ── Accuracy table ────────────────────────────────────────────────────────
    with col_r:
        with st.container(border=True):
            st.markdown('<div class="slabel">Week-by-Week Breakdown</div>', unsafe_allow_html=True)

            def badge(pct):
                a = abs(pct)
                if a < 3:   return '<span class="badge b-green">On track</span>'
                elif a < 7: return '<span class="badge b-amber">Watch</span>'
                else:       return '<span class="badge b-red">Miss</span>'

            rows2 = "".join(f"""
            <tr>
              <td>{pd.Timestamp(r['week']).strftime('%d %b')}</td>
              <td class="mono r">{r['actual']:,}</td>
              <td class="mono r">{r['forecast']:,}</td>
              <td class="mono r" style="color:{
                  C['green'] if abs(r['error_pct'])<3 else
                  C['amber'] if abs(r['error_pct'])<7 else C['red']
              };font-weight:600;">{r['error_pct']:+.1f}%</td>
              <td class="c">{badge(r['error_pct'])}</td>
            </tr>""" for _, r in acc.iterrows())

            on_track = int((np.abs(acc["error_pct"]) < 5).sum())
            st.markdown(f"""
            <table class="dt">
              <thead><tr>
                <th>Week</th>
                <th class="r">Actual</th>
                <th class="r">Forecast</th>
                <th class="r">Error</th>
                <th class="c">Status</th>
              </tr></thead>
              <tbody>{rows2}</tbody>
            </table>
            <div style="margin-top:0.85rem;padding-top:0.7rem;border-top:1px solid #F1F5F9;
                        display:flex;justify-content:space-between;">
              <span style="font-size:0.71rem;color:#94A3B8;">{on_track}/12 within ±5%</span>
              <span style="font-size:0.71rem;color:#94A3B8;">Avg error {avg_err:.1f}%</span>
            </div>
            """, unsafe_allow_html=True)

# ═════════════════════════════════════════════════════════════════════════════
# TAB 3 — Export
# ═════════════════════════════════════════════════════════════════════════════
with tab3:
    col1, col2 = st.columns(2, gap="medium")

    export_fc = fc.copy()
    export_fc["week"] = export_fc["week"].dt.strftime("%Y-%m-%d")
    export_fc.columns = ["week_start", "forecast_units", "lower_bound",
                         "upper_bound", "safety_stock_units", "recommended_order_qty"]

    export_acc = acc.copy()
    export_acc["week"] = pd.to_datetime(export_acc["week"]).dt.strftime("%Y-%m-%d")

    with col1:
        with st.container(border=True):
            st.markdown('<div class="slabel">Forward Demand Forecast</div>', unsafe_allow_html=True)
            st.markdown(f"""
            <div style="font-size:0.77rem;color:#64748B;margin-bottom:0.9rem;">
              Next {n_weeks} weeks &nbsp;·&nbsp;
              {int(service_level*100)}% service level &nbsp;·&nbsp;
              Generated {meta['trained_at'][:10]}
            </div>
            """, unsafe_allow_html=True)
            st.download_button(
                "↓  Download forecast CSV",
                data=export_fc.to_csv(index=False).encode(),
                file_name=f"demand_forecast_{pd.Timestamp(fc['week'].iloc[0]).strftime('%Y%m%d')}.csv",
                mime="text/csv",
            )
            st.dataframe(export_fc, use_container_width=True, hide_index=True, height=300)

    with col2:
        with st.container(border=True):
            st.markdown('<div class="slabel">Forecast Accuracy Log</div>', unsafe_allow_html=True)
            st.markdown("""
            <div style="font-size:0.77rem;color:#64748B;margin-bottom:0.9rem;">
              Last 12 weeks &nbsp;·&nbsp; Actual vs model forecast
            </div>
            """, unsafe_allow_html=True)
            st.download_button(
                "↓  Download accuracy log CSV",
                data=export_acc.to_csv(index=False).encode(),
                file_name=f"accuracy_log_{meta['last_data_date']}.csv",
                mime="text/csv",
            )
            st.dataframe(export_acc, use_container_width=True, hide_index=True, height=300)

    st.markdown("<div style='height:1rem'></div>", unsafe_allow_html=True)
    st.markdown(f"""
    <div class="card infobox" style="background:#F8FAFC;">
      <div class="slabel" style="margin-bottom:0.6rem;">Column Reference</div>
      <table style="width:100%;font-size:0.76rem;border-collapse:collapse;color:#475569;">
        <tr style="border-bottom:1px solid #F1F5F9;">
          <td style="padding:0.3rem 0.5rem;width:220px;">
            <code style="background:#EFF6FF;color:#1D4ED8;padding:0.1rem 0.35rem;border-radius:3px;">
              forecast_units</code></td>
          <td style="padding:0.3rem 0.5rem;">Point estimate for weekly demand</td></tr>
        <tr style="border-bottom:1px solid #F1F5F9;">
          <td style="padding:0.3rem 0.5rem;">
            <code style="background:#EFF6FF;color:#1D4ED8;padding:0.1rem 0.35rem;border-radius:3px;">
              lower / upper_bound</code></td>
          <td style="padding:0.3rem 0.5rem;">
            {int(service_level*100)}% CI — RMSE = {rmse:.0f} units</td></tr>
        <tr style="border-bottom:1px solid #F1F5F9;">
          <td style="padding:0.3rem 0.5rem;">
            <code style="background:#EFF6FF;color:#1D4ED8;padding:0.1rem 0.35rem;border-radius:3px;">
              safety_stock_units</code></td>
          <td style="padding:0.3rem 0.5rem;">
            z={z} × RMSE = {buf:,} units at {int(service_level*100)}% SL</td></tr>
        <tr>
          <td style="padding:0.3rem 0.5rem;">
            <code style="background:#EFF6FF;color:#1D4ED8;padding:0.1rem 0.35rem;border-radius:3px;">
              recommended_order_qty</code></td>
          <td style="padding:0.3rem 0.5rem;">
            Forecast + safety stock — map to planned order in ERP</td></tr>
      </table>
    </div>
    """, unsafe_allow_html=True)
