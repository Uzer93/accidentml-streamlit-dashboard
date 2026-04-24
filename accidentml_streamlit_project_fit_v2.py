# =============================================================================
# MLOps Dashboard — Road Accident Severity Prediction (France)
# Team: Mohammad Reza Nilchiyan | Megha Panchal | Ahmad Melhem
# Run: streamlit run app.py
# =============================================================================

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import time
import json
from urllib import request

# ─── PAGE CONFIG ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="AccidentML · France",
    page_icon="🚦",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── GLOBAL CSS ──────────────────────────────────────────────────────────────
st.markdown("""
<style>
  @import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=DM+Mono:wght@400;500&family=DM+Sans:wght@300;400;500&display=swap');

  /* ── Base ── */
  html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }
  .main { background: #0b0f1a; color: #e2e8f0; }
  .block-container { padding: 2rem 3rem; max-width: 1300px; }

  /* ── Sidebar ── */
  [data-testid="stSidebar"] {
    background: #080c16;
    border-right: 1px solid #1e2a40;
  }
  [data-testid="stSidebar"] * { color: #cbd5e1 !important; }

  /* ── Typography ── */
  h1, h2, h3 { font-family: 'Syne', sans-serif !important; }
  .page-title {
    font-family: 'Syne', sans-serif;
    font-size: 2.6rem;
    font-weight: 800;
    background: linear-gradient(135deg, #f97316, #fb923c, #fbbf24);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin-bottom: 0.2rem;
  }
  .page-subtitle {
    font-family: 'DM Sans', sans-serif;
    color: #64748b;
    font-size: 1rem;
    margin-bottom: 2rem;
  }

  /* ── Cards ── */
  .metric-card {
    background: #111827;
    border: 1px solid #1e2a40;
    border-radius: 12px;
    padding: 1.2rem 1.5rem;
    margin-bottom: 1rem;
  }
  .metric-value {
    font-family: 'Syne', sans-serif;
    font-size: 2rem;
    font-weight: 700;
    color: #f97316;
  }
  .metric-label {
    font-size: 0.82rem;
    color: #64748b;
    text-transform: uppercase;
    letter-spacing: 0.08em;
  }

  /* ── Section headers ── */
  .section-label {
    font-family: 'DM Mono', monospace;
    font-size: 0.72rem;
    color: #f97316;
    text-transform: uppercase;
    letter-spacing: 0.15em;
    margin-bottom: 0.4rem;
  }
  .section-title {
    font-family: 'Syne', sans-serif;
    font-size: 1.4rem;
    font-weight: 700;
    color: #f1f5f9;
    margin-bottom: 1rem;
  }

  /* ── Code blocks ── */
  .stCodeBlock { border-radius: 8px !important; }

  /* ── Pills / badges ── */
  .badge {
    display: inline-block;
    background: #1e2a40;
    border: 1px solid #f97316;
    color: #f97316;
    font-family: 'DM Mono', monospace;
    font-size: 0.72rem;
    padding: 2px 10px;
    border-radius: 20px;
    margin: 2px 3px;
  }
  .badge-green { border-color: #22c55e; color: #22c55e; }
  .badge-red   { border-color: #ef4444; color: #ef4444; }
  .badge-blue  { border-color: #3b82f6; color: #3b82f6; }

  /* ── Divider ── */
  .fancy-divider {
    border: none;
    height: 1px;
    background: linear-gradient(to right, #f97316, #1e2a40, transparent);
    margin: 2rem 0;
  }

  /* ── Streamlit overrides ── */
  [data-testid="metric-container"] {
    background: #111827;
    border: 1px solid #1e2a40;
    border-radius: 10px;
    padding: 0.8rem;
  }
  div[data-testid="stMetricValue"] { color: #f97316 !important; font-family: 'Syne', sans-serif !important; }
</style>
""", unsafe_allow_html=True)

# ─── PLOTLY THEME ─────────────────────────────────────────────────────────────
PLOT_THEME = dict(
    paper_bgcolor="#111827",
    plot_bgcolor="#0b0f1a",
    font_color="#e2e8f0",
    font_family="DM Sans",
    colorway=["#f97316","#3b82f6","#22c55e","#a855f7","#ec4899","#fbbf24"],
)

def apply_theme(fig):
    fig.update_layout(**PLOT_THEME,
                      margin=dict(l=20, r=20, t=40, b=20),
                      legend=dict(bgcolor="rgba(0,0,0,0)"))
    fig.update_xaxes(gridcolor="#1e2a40", linecolor="#1e2a40")
    fig.update_yaxes(gridcolor="#1e2a40", linecolor="#1e2a40")
    return fig

# ─── PROJECT STATUS / DEMO HELPERS ───────────────────────────────────────────
PROJECT_REPO = "https://github.com/Megha-2023/mar26bmlops_int_accidents"

STATUS_ROWS = [
    ("Core ML pipeline", "Implemented / tested", "Preprocessing, XGBoost training, evaluation and saved artifacts worked during project testing."),
    ("FastAPI inference service", "Implemented / tested", "Prediction endpoint and health endpoint were validated."),
    ("Docker image build", "Implemented / tested", "The API Docker image build completed successfully."),
    ("Docker Compose stack", "Implemented / tested", "The local multi-service stack was tested."),
    ("MLflow tracking", "Implemented / tested", "Experiment tracking and model registration were validated."),
    ("Evidently monitoring", "Implemented / tested", "Data/prediction drift reporting was included and tested."),
    ("GitHub Actions CI", "Implemented", "Unit-test CI is represented; confirm latest run status in the repository."),
    ("DVC reproducibility", "Needs cleanup", "Fresh-clone reproducibility still needs work, especially data/model artifact restoration."),
    ("nginx HTTPS", "Needs cleanup", "HTTPS setup is blocked by the missing private key file."),
    ("Kubernetes / Prometheus / Grafana", "Design / extension", "Shown as a future deployment blueprint, not a fully tested production stack."),
]

def status_badge(status: str) -> str:
    low = status.lower()
    if "tested" in low or low == "implemented":
        klass = "badge badge-green"
    elif "needs" in low:
        klass = "badge badge-red"
    else:
        klass = "badge badge-blue"
    return f"<span class='{klass}'>{status}</span>"

def demo_notice(kind="overview"):
    if kind == "mock":
        st.info("Demo note: this panel uses representative/mock data so the dashboard can run without the full BAAC dataset, DVC remote, MLflow server, or API stack.")
    elif kind == "real_or_mock":
        st.info("This panel is designed to show real project artifacts when connected; otherwise the displayed values are representative demo values.")
    else:
        st.info("This dashboard is a presentation layer for the AccidentML project. It separates tested components from demo visualizations and known cleanup items.")

def try_api_prediction(api_url: str, payload: dict):
    """Call a running FastAPI endpoint if available; return an error dictionary on failure."""
    try:
        data = json.dumps(payload).encode("utf-8")
        req = request.Request(
            api_url.rstrip("/") + "/predict",
            data=data,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with request.urlopen(req, timeout=4) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except Exception as exc:
        return {"error": str(exc)}

# ─── MOCK DATA GENERATORS ────────────────────────────────────────────────────
@st.cache_data
def get_sample_data(n=500):
    np.random.seed(42)
    gravites = np.random.choice([1, 2, 3, 4], size=n, p=[0.10, 0.25, 0.40, 0.25])
    df = pd.DataFrame({
        "Num_Acc":      [f"2012{str(i).zfill(6)}" for i in range(n)],
        "an":           np.random.choice(range(2010, 2016), n),
        "mois":         np.random.randint(1, 13, n),
        "jour":         np.random.randint(1, 8, n),
        "hrmn":         np.random.randint(0, 2400, n),
        "lum":          np.random.choice([1,2,3,4,5], n),
        "agg":          np.random.choice([1,2], n),
        "int":          np.random.choice([1,2,3,4,5,6,7,9], n),
        "atm":          np.random.choice([1,2,3,4,5,6,7,8,9], n),
        "col":          np.random.choice([1,2,3,4,5,6,7], n),
        "dep":          np.random.choice(["75","13","69","31","33","59","67","06"], n),
        "lat":          np.random.uniform(43.0, 51.0, n).round(4),
        "long":         np.random.uniform(-4.0, 7.5, n).round(4),
        "catr":         np.random.choice([1,2,3,4,5,6,9], n),
        "vosp":         np.random.choice([0,1,2,3], n),
        "circ":         np.random.choice([1,2,3,4], n),
        "surf":         np.random.choice([1,2,3,4,5,6,7,8,9], n),
        "plan":         np.random.choice([1,2,3,4], n),
        "age":          np.random.randint(16, 85, n),
        "sexe":         np.random.choice([1,2], n),
        "catv":         np.random.choice([1,2,7,10,33,37], n),
        "grav":         gravites,
    })
    # Introduce a few missing values
    for col in ["atm","surf","plan","vosp"]:
        mask = np.random.choice([True,False], n, p=[0.04, 0.96])
        df.loc[mask, col] = np.nan
    return df

@st.cache_data
def get_preprocessed_data():
    np.random.seed(42)
    n = 400
    X = pd.DataFrame({
        "lum":   np.random.choice([1,2,3,4,5], n),
        "agg":   np.random.choice([1,2], n),
        "atm":   np.random.choice([1,2,3,4,5], n),
        "col":   np.random.choice([1,2,3,4,5,6,7], n),
        "catr":  np.random.choice([1,2,3,4,5], n),
        "surf":  np.random.choice([1,2,3,4,5], n),
        "age":   np.random.randint(16, 85, n),
        "sexe":  np.random.choice([0,1], n),
        "month": np.random.randint(1,13,n),
        "hour":  np.random.randint(0,24,n),
    })
    y = np.random.choice([0,1,2,3], n, p=[0.10,0.25,0.40,0.25])
    return X, y

# ─── SIDEBAR ─────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style='text-align:center; padding: 1rem 0;'>
      <div style='font-size:2.5rem;'>🚦</div>
      <div style='font-family:Syne,sans-serif; font-size:1.1rem; font-weight:800;
                  background:linear-gradient(135deg,#f97316,#fbbf24);
                  -webkit-background-clip:text; -webkit-text-fill-color:transparent;'>
        AccidentML
      </div>
      <div style='font-size:0.72rem; color:#475569; font-family:DM Mono,monospace;
                  letter-spacing:0.1em; margin-top:2px;'>
        FRANCE · SEVERITY PRED.
      </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<hr style='border:1px solid #1e2a40; margin:0.5rem 0 1rem;'>", unsafe_allow_html=True)

    page = st.radio("Navigation", [
        "✅  Project Status",
        "📊  Dataset & EDA",
        "⚙️  Data Processing",
        "🤖  Baseline Model",
        "🔗  Microservices & MLflow",
        "🚀  Orchestration & Deploy",
        "📡  Monitoring & Maintenance",
    ], label_visibility="collapsed")

    st.markdown("<hr style='border:1px solid #1e2a40; margin:1rem 0;'>", unsafe_allow_html=True)
    st.markdown("""
    <div style='font-size:0.72rem; color:#334155; font-family:DM Mono,monospace;'>
      <div style='margin-bottom:4px;'>👤 Mohammad Reza Nilchiyan</div>
      <div style='margin-bottom:4px;'>👤 Megha Panchal</div>
      <div style='margin-bottom:4px;'>👤 Ahmad Melhem</div>
      <div style='margin-top:10px; color:#475569;'>MLOps · March 2026</div>
    </div>
    """, unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# PAGE 0 — PROJECT STATUS
# ══════════════════════════════════════════════════════════════════════════════
if page == "✅  Project Status":
    st.markdown('<div class="page-title">Project Status & Demo Scope</div>', unsafe_allow_html=True)
    st.markdown('<div class="page-subtitle">What is implemented, what is demonstrated, and what still needs cleanup</div>', unsafe_allow_html=True)
    demo_notice()

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Problem", "Severity prediction", "BAAC France")
    c2.metric("Model", "XGBoost", "multiclass")
    c3.metric("Serving", "FastAPI", "tested")
    c4.metric("Demo scope", "Hybrid", "real + mock panels")

    st.markdown("<hr class='fancy-divider'>", unsafe_allow_html=True)
    st.markdown('<div class="section-title">Component Readiness</div>', unsafe_allow_html=True)

    st.markdown("""
    <style>
    .status-table td, .status-table th { padding: 0.65rem 0.8rem; border-bottom: 1px solid #1e2a40; }
    .status-table th { color: #94a3b8; text-align: left; font-family: 'DM Mono', monospace; font-size: 0.75rem; text-transform: uppercase; }
    .status-table td { color: #cbd5e1; font-size: 0.88rem; vertical-align: top; }
    </style>
    """, unsafe_allow_html=True)
    rows_html = "".join(
        f"<tr><td><b>{comp}</b></td><td>{status_badge(status)}</td><td>{note}</td></tr>"
        for comp, status, note in STATUS_ROWS
    )
    st.markdown(f"""
    <table class="status-table" style="width:100%; border-collapse:collapse; background:#111827; border:1px solid #1e2a40; border-radius:12px; overflow:hidden;">
      <thead><tr><th>Component</th><th>Status</th><th>Evidence / note</th></tr></thead>
      <tbody>{rows_html}</tbody>
    </table>
    """, unsafe_allow_html=True)

    st.markdown("<hr class='fancy-divider'>", unsafe_allow_html=True)
    st.markdown('<div class="section-title">How to interpret this Streamlit app</div>', unsafe_allow_html=True)
    st.markdown("""
    - **Implemented/tested** panels describe components validated during project testing: ML pipeline, FastAPI, Docker image, Docker Compose, MLflow and Evidently.
    - **Demo/mock** panels use generated sample data so the dashboard can run during review without restoring the full BAAC data or DVC remote.
    - **Cleanup** items are shown transparently: DVC fresh-clone reproducibility and nginx HTTPS/private-key setup.
    - **Architecture extensions** such as Kubernetes, Prometheus and Grafana are shown as deployment blueprints, not claimed production infrastructure.
    """)

    st.markdown("<hr class='fancy-divider'>", unsafe_allow_html=True)
    st.markdown('<div class="section-title">Recommended reviewer flow</div>', unsafe_allow_html=True)
    st.graphviz_chart("""
    digraph review {
      rankdir=LR;
      node [shape=box style=filled fontname="Helvetica" fontsize=10]
      A [label="Project status\n(scope + honesty)" fillcolor="#1e293b" fontcolor="white"]
      B [label="Dataset & EDA\nBAAC context" fillcolor="#7c3aed" fontcolor="white"]
      C [label="Processing\ntime split + features" fillcolor="#c2410c" fontcolor="white"]
      D [label="Model + API\nXGBoost/FastAPI" fillcolor="#166534" fontcolor="white"]
      E [label="MLOps layer\nMLflow/DVC/Docker" fillcolor="#0369a1" fontcolor="white"]
      F [label="Known gaps\nDVC + HTTPS" fillcolor="#7f1d1d" fontcolor="white"]
      A -> B -> C -> D -> E -> F
    }
    """)

# ══════════════════════════════════════════════════════════════════════════════
# PAGE 1 — DATASET & EDA
# ══════════════════════════════════════════════════════════════════════════════
elif page == "📊  Dataset & EDA":
    st.markdown('<div class="page-title">Dataset & Exploration</div>', unsafe_allow_html=True)
    st.markdown('<div class="page-subtitle">French Road Accident Data (2005–2020) · Source: data.gouv.fr</div>', unsafe_allow_html=True)
    demo_notice("mock")

    df = get_sample_data()

    # ── KPI row
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Records", "1.2M+", "2005–2020")
    c2.metric("Features", "34", "across 4 tables")
    c3.metric("Train Years", "2010–2015", "time-split")
    c4.metric("Test Year", "2016", "future simulation")

    st.markdown("<hr class='fancy-divider'>", unsafe_allow_html=True)

    # ── Data Story
    with st.expander("📖  Data Story — Why this dataset?", expanded=True):
        st.markdown("""
        <div style='font-family:DM Sans,sans-serif; line-height:1.8; color:#cbd5e1;'>
        France's national road accident register (<b>BAAC</b>) has tracked every police-recorded accident
        since 1999. This project leverages the <b>2005–2020</b> slice, spanning four relational tables:
        <br><br>
        <span class='badge'>caracteristiques</span> accident context (time, weather, location)
        <span class='badge'>lieux</span> road geometry & surface
        <span class='badge'>usagers</span> people involved — <b>grav</b> (severity) lives here
        <span class='badge'>vehicules</span> vehicle types
        <br><br>
        <b>Target variable — <code>grav</code>:</b> 1 = Uninjured · 2 = Killed · 3 = Hospitalised · 4 = Light injury
        <br><br>
        The challenge: <b>class imbalance</b>, schema changes post-2019, and temporal autocorrelation all
        make a naive train/test split inappropriate. We use a <b>time-based split</b> (train 2010–2015, test 2016)
        to simulate real-world future prediction.
        </div>
        """, unsafe_allow_html=True)

    # ── Sample data
    st.markdown('<div class="section-label">Sample Rows</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-title">Merged Dataset Preview</div>', unsafe_allow_html=True)
    st.dataframe(df.head(12), use_container_width=True)

    st.markdown("<hr class='fancy-divider'>", unsafe_allow_html=True)

    # ── Visualisations
    col_l, col_r = st.columns(2)

    with col_l:
        # Severity distribution
        grav_map = {1:"Uninjured",2:"Killed",3:"Hospitalised",4:"Light Injury"}
        grav_counts = df["grav"].map(grav_map).value_counts().reset_index()
        grav_counts.columns = ["Severity","Count"]
        fig = px.bar(grav_counts, x="Severity", y="Count", color="Severity",
                     title="Target — Accident Severity Distribution",
                     color_discrete_sequence=["#22c55e","#ef4444","#f97316","#3b82f6"])
        apply_theme(fig)
        st.plotly_chart(fig, use_container_width=True)

    with col_r:
        # Accidents by year
        yr = df["an"].value_counts().sort_index().reset_index()
        yr.columns = ["Year","Count"]
        fig2 = px.line(yr, x="Year", y="Count", title="Accidents by Year (sample)",
                       markers=True, line_shape="spline")
        fig2.update_traces(line_color="#f97316", marker_color="#fbbf24", line_width=3)
        apply_theme(fig2)
        st.plotly_chart(fig2, use_container_width=True)

    col_l2, col_r2 = st.columns(2)

    with col_l2:
        # Accidents by hour
        df["hour"] = (df["hrmn"] // 100).clip(0, 23)
        hr = df["hour"].value_counts().sort_index().reset_index()
        hr.columns = ["Hour","Count"]
        fig3 = px.bar(hr, x="Hour", y="Count", title="Accidents by Hour of Day",
                      color="Count", color_continuous_scale="Oranges")
        apply_theme(fig3)
        st.plotly_chart(fig3, use_container_width=True)

    with col_r2:
        # Missing value heatmap
        missing = df.isnull().sum().reset_index()
        missing.columns = ["Feature","Missing"]
        missing = missing[missing["Missing"] > 0]
        fig4 = px.bar(missing, x="Feature", y="Missing", title="Missing Values per Feature",
                      color="Missing", color_continuous_scale="Reds")
        apply_theme(fig4)
        st.plotly_chart(fig4, use_container_width=True)

    # Correlation heatmap
    st.markdown('<div class="section-title">Feature Correlation Heatmap</div>', unsafe_allow_html=True)
    num_cols = ["lum","agg","atm","col","catr","surf","age","sexe","grav"]
    corr = df[num_cols].dropna().corr().round(2)
    fig5 = go.Figure(data=go.Heatmap(
        z=corr.values, x=corr.columns, y=corr.index,
        colorscale="RdBu", zmid=0,
        text=corr.values, texttemplate="%{text}",
    ))
    fig5.update_layout(title="Pearson Correlation Matrix", **PLOT_THEME,
                       margin=dict(l=20,r=20,t=40,b=20))
    st.plotly_chart(fig5, use_container_width=True)

    # Department breakdown
    with st.expander("🗺️  Accidents by Department (Top 8)"):
        dep = df["dep"].value_counts().reset_index()
        dep.columns = ["Department","Count"]
        fig6 = px.bar(dep, x="Department", y="Count",
                      color="Count", color_continuous_scale="Oranges",
                      title="Accident Count by Department")
        apply_theme(fig6)
        st.plotly_chart(fig6, use_container_width=True)

# ══════════════════════════════════════════════════════════════════════════════
# PAGE 2 — DATA PROCESSING
# ══════════════════════════════════════════════════════════════════════════════
elif page == "⚙️  Data Processing":
    st.markdown('<div class="page-title">Data Processing & Preparation</div>', unsafe_allow_html=True)
    st.markdown('<div class="page-subtitle">Pipeline: Merge → Clean → Feature Engineer → Time Split</div>', unsafe_allow_html=True)
    demo_notice("real_or_mock")

    # Pipeline steps
    steps = [
        ("01", "Load & Merge", "Join caracteristiques, lieux, usagers, vehicules on Num_Acc"),
        ("02", "Time Filter",  "Keep 2010–2016 only — avoids schema drift from 2019 changes"),
        ("03", "Drop Columns", "Remove near-constant, ID, and redundant geo columns"),
        ("04", "Handle NaN",   "Median-fill numeric; mode-fill categorical; drop rows >30% missing"),
        ("05", "Encode",       "Ordinal encode categoricals (no OHE — XGBoost handles ordinals)"),
        ("06", "Feature Eng.", "Extract hour, day-of-week, season from hrmn+date; is_night, is_weekend"),
        ("07", "Time Split",   "Train: 2010–2015 | Test: 2016 — no shuffle, respects temporal order"),
        ("08", "Scale",        "StandardScaler on age only; tree model doesn't need full scaling"),
    ]

    cols = st.columns(4)
    for i, (num, title, desc) in enumerate(steps):
        with cols[i % 4]:
            st.markdown(f"""
            <div class='metric-card' style='min-height:110px;'>
              <div style='font-family:DM Mono,monospace;font-size:0.65rem;color:#f97316;
                          letter-spacing:0.15em;'>STEP {num}</div>
              <div style='font-family:Syne,sans-serif;font-weight:700;
                          font-size:0.95rem;color:#f1f5f9;margin:4px 0;'>{title}</div>
              <div style='font-size:0.78rem;color:#64748b;line-height:1.5;'>{desc}</div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("<hr class='fancy-divider'>", unsafe_allow_html=True)

    # Before / After
    st.markdown('<div class="section-title">Before & After Transformations</div>', unsafe_allow_html=True)
    tab1, tab2, tab3 = st.tabs(["🧹  Cleaning", "🔧  Feature Engineering", "✂️  Train/Test Split"])

    with tab1:
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("**Before — raw grav column**")
            st.dataframe(pd.DataFrame({"grav":[1,2,3,4,np.nan,1,3,2,np.nan,4]}))
        with c2:
            st.markdown("**After — cleaned & cast**")
            st.dataframe(pd.DataFrame({"grav":[1,2,3,4,3,1,3,2,2,4]}))
        st.code("""
# src/make_dataset.py — cleaning snippet
df.dropna(subset=['grav'], inplace=True)          # target must be present
df['grav'] = df['grav'].astype(int)

# Fill numeric NaN with median
for col in numeric_cols:
    df[col].fillna(df[col].median(), inplace=True)

# Fill categorical NaN with mode
for col in categorical_cols:
    df[col].fillna(df[col].mode()[0], inplace=True)
        """, language="python")

    with tab2:
        st.code("""
# Feature engineering — extract temporal features
df['hour']       = (df['hrmn'] // 100).clip(0, 23)
df['minute']     = df['hrmn'] % 100
df['is_night']   = df['hour'].apply(lambda h: 1 if h < 6 or h >= 22 else 0)
df['is_weekend'] = df['jour'].apply(lambda d: 1 if d in [6,7] else 0)
df['season']     = df['mois'].apply(
    lambda m: 'Winter' if m in [12,1,2]
    else 'Spring' if m in [3,4,5]
    else 'Summer' if m in [6,7,8]
    else 'Autumn'
)
df = pd.get_dummies(df, columns=['season'], drop_first=True)
        """, language="python")
        st.markdown('<span class="badge">hour</span><span class="badge">is_night</span><span class="badge">is_weekend</span><span class="badge">season_*</span> added to feature set', unsafe_allow_html=True)

    with tab3:
        st.code("""
# Time-based split — NO random shuffling
train_df = df[df['an'].between(2010, 2015)]
test_df  = df[df['an'] == 2016]

X_train = train_df.drop(columns=['grav','Num_Acc','an'])
y_train = train_df['grav']
X_test  = test_df.drop(columns=['grav','Num_Acc','an'])
y_test  = test_df['grav']

# Saved to data/preprocessed/
X_train.to_csv('data/preprocessed/X_train.csv', index=False)
X_test.to_csv('data/preprocessed/X_test.csv',  index=False)
y_train.to_csv('data/preprocessed/y_train.csv', index=False)
y_test.to_csv('data/preprocessed/y_test.csv',   index=False)
        """, language="python")
        c1, c2, c3 = st.columns(3)
        c1.metric("Train Samples", "~900K", "2010–2015")
        c2.metric("Test Samples",  "~150K", "2016")
        c3.metric("Features",      "18",    "post-engineering")

    st.markdown("<hr class='fancy-divider'>", unsafe_allow_html=True)

    # Class distribution post-split
    st.markdown('<div class="section-title">Class Distribution — Train vs Test</div>', unsafe_allow_html=True)
    grav_labels = ["Uninjured","Killed","Hospitalised","Light Injury"]
    train_dist = [0.10, 0.08, 0.42, 0.40]
    test_dist  = [0.11, 0.07, 0.41, 0.41]
    fig = go.Figure()
    fig.add_trace(go.Bar(name="Train", x=grav_labels, y=train_dist, marker_color="#f97316"))
    fig.add_trace(go.Bar(name="Test",  x=grav_labels, y=test_dist,  marker_color="#3b82f6"))
    fig.update_layout(barmode="group", title="Normalised Class Proportions", **PLOT_THEME)
    apply_theme(fig)
    st.plotly_chart(fig, use_container_width=True)
    st.info("ℹ️ Class imbalance is mild but present. **class_weight** and **scale_pos_weight** in XGBoost mitigate bias toward the majority class.")

# ══════════════════════════════════════════════════════════════════════════════
# PAGE 3 — BASELINE MODEL
# ══════════════════════════════════════════════════════════════════════════════
elif page == "🤖  Baseline Model":
    st.markdown('<div class="page-title">Baseline ML Model</div>', unsafe_allow_html=True)
    st.markdown('<div class="page-subtitle">Algorithm: XGBoost Classifier · Evaluation · Unit Tests · Inference API</div>', unsafe_allow_html=True)
    demo_notice("real_or_mock")

    # Model card
    st.markdown('<div class="section-title">Model Configuration</div>', unsafe_allow_html=True)
    c1, c2 = st.columns([1,2])
    with c1:
        st.markdown("""
        <div class='metric-card'>
          <div class='metric-label'>Algorithm</div>
          <div class='metric-value' style='font-size:1.3rem;'>XGBoost</div>
          <div class='metric-label' style='margin-top:1rem;'>Type</div>
          <div style='color:#cbd5e1;'>Multiclass Classification</div>
          <div class='metric-label' style='margin-top:0.8rem;'>Classes</div>
          <div style='color:#cbd5e1;'>4 (grav 1–4)</div>
        </div>
        """, unsafe_allow_html=True)
    with c2:
        st.code("""
# src/train_model.py
import xgboost as xgb
import pickle, mlflow

params = {
    "n_estimators":     300,
    "max_depth":        6,
    "learning_rate":    0.1,
    "subsample":        0.8,
    "colsample_bytree": 0.8,
    "objective":        "multi:softprob",
    "num_class":        4,
    "eval_metric":      "mlogloss",
    "use_label_encoder": False,
    "random_state":     42,
}

model = xgb.XGBClassifier(**params)
model.fit(X_train, y_train,
          eval_set=[(X_test, y_test)],
          early_stopping_rounds=20,
          verbose=False)

with open("models/xgb_model.pkl", "wb") as f:
    pickle.dump(model, f)
        """, language="python")

    st.markdown("<hr class='fancy-divider'>", unsafe_allow_html=True)

    # Metrics
    st.markdown('<div class="section-title">Evaluation Dashboard</div>', unsafe_allow_html=True)
    c1,c2,c3,c4 = st.columns(4)
    c1.metric("Accuracy",  "0.612", "+0.04 vs dummy")
    c2.metric("F1-Score",  "0.589", "weighted avg")
    c3.metric("Log-Loss",  "0.987", "↓ lower is better")
    c4.metric("AUC (OvR)", "0.741", "macro avg")

    col_l, col_r = st.columns(2)

    with col_l:
        # Mock confusion matrix
        cm = np.array([[82,  3, 10,  5],
                       [ 4, 71,  8,  17],
                       [ 9,  6,248, 37],
                       [ 7, 12, 41,178]])
        fig = px.imshow(cm,
                        labels=dict(x="Predicted", y="Actual", color="Count"),
                        x=["Uninj.","Killed","Hosp.","Light"],
                        y=["Uninj.","Killed","Hosp.","Light"],
                        color_continuous_scale="Oranges",
                        title="Confusion Matrix")
        apply_theme(fig)
        st.plotly_chart(fig, use_container_width=True)

    with col_r:
        # Mock ROC curves
        fig = go.Figure()
        colors = ["#f97316","#3b82f6","#22c55e","#a855f7"]
        labels_roc = ["Uninjured","Killed","Hospitalised","Light Injury"]
        aucs = [0.78, 0.85, 0.74, 0.71]
        for i, (lbl, auc, col) in enumerate(zip(labels_roc, aucs, colors)):
            t = np.linspace(0, 1, 50)
            fpr = t
            tpr = np.clip(t + np.random.beta(2,1,50)*0.3, 0, 1)
            tpr = np.sort(tpr)
            fig.add_trace(go.Scatter(x=fpr, y=tpr, name=f"{lbl} (AUC={auc})",
                                     line=dict(color=col, width=2)))
        fig.add_trace(go.Scatter(x=[0,1], y=[0,1], name="Random",
                                 line=dict(dash="dash", color="#475569")))
        fig.update_layout(title="ROC Curves (One-vs-Rest)", xaxis_title="FPR",
                          yaxis_title="TPR", **PLOT_THEME)
        apply_theme(fig)
        st.plotly_chart(fig, use_container_width=True)

    # Training curve
    with st.expander("📉  Training Loss Curve"):
        epochs = np.arange(1, 301)
        train_loss = 1.4 * np.exp(-0.008 * epochs) + 0.05 + np.random.normal(0, 0.005, 300)
        val_loss   = 1.5 * np.exp(-0.007 * epochs) + 0.12 + np.random.normal(0, 0.008, 300)
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=epochs, y=train_loss, name="Train Loss", line=dict(color="#f97316")))
        fig.add_trace(go.Scatter(x=epochs, y=val_loss,   name="Val Loss",   line=dict(color="#3b82f6")))
        fig.update_layout(title="mlogloss over Boosting Rounds", **PLOT_THEME)
        apply_theme(fig)
        st.plotly_chart(fig, use_container_width=True)

    # Feature importance
    with st.expander("🔎  Feature Importance (XGBoost gain)"):
        feats = ["hour","age","catr","surf","atm","col","lum","is_night","agg","sexe","is_weekend","season_Summer"]
        gains = sorted(np.random.uniform(0.02, 0.18, len(feats)), reverse=True)
        fig = px.bar(x=gains, y=feats, orientation="h",
                     color=gains, color_continuous_scale="Oranges",
                     title="Feature Importance by Gain")
        apply_theme(fig)
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("<hr class='fancy-divider'>", unsafe_allow_html=True)

    # Unit tests
    st.markdown('<div class="section-title">Unit Test Results</div>', unsafe_allow_html=True)
    tests = [
        ("test_data_shape",         "tests/test_pipeline.py", "PASS", "Train shape (N,18)"),
        ("test_no_nan_after_clean",  "tests/test_pipeline.py", "PASS", "0 NaN post-clean"),
        ("test_label_range",         "tests/test_pipeline.py", "PASS", "grav ∈ {1,2,3,4}"),
        ("test_model_output_shape",  "tests/test_model.py",    "PASS", "predict returns (N,)"),
        ("test_model_classes",       "tests/test_model.py",    "PASS", "4 unique classes"),
        ("test_api_status_200",      "tests/test_api.py",      "PASS", "POST /predict → 200"),
        ("test_api_invalid_input",   "tests/test_api.py",      "PASS", "POST /predict → 422"),
        ("test_time_split_leak",     "tests/test_pipeline.py", "PASS", "No 2016 rows in train"),
    ]
    test_df = pd.DataFrame(tests, columns=["Test Name","File","Status","Description"])

    def colour_status(val):
        if val == "PASS":
            return "background-color:#14532d;color:#86efac;"
        return "background-color:#7f1d1d;color:#fca5a5;"

    st.dataframe(
        test_df.style.map(colour_status, subset=["Status"]),
        use_container_width=True, hide_index=True
    )

    st.markdown("<hr class='fancy-divider'>", unsafe_allow_html=True)

    # Inference API
    st.markdown('<div class="section-title">Inference API (FastAPI)</div>', unsafe_allow_html=True)
    with st.expander("📄  FastAPI Endpoint Code"):
        st.code("""
# src/api/main.py
from fastapi import FastAPI
from pydantic import BaseModel
import pickle, numpy as np

app = FastAPI(title="AccidentML API", version="1.0")

with open("models/xgb_model.pkl", "rb") as f:
    model = pickle.load(f)

class AccidentFeatures(BaseModel):
    lum: int; agg: int; atm: int; col: int
    catr: int; surf: int; age: int; sexe: int
    hour: int; is_night: int; is_weekend: int

@app.post("/predict")
def predict(features: AccidentFeatures):
    X = np.array([[features.lum, features.agg, features.atm, features.col,
                   features.catr, features.surf, features.age, features.sexe,
                   features.hour, features.is_night, features.is_weekend]])
    proba = model.predict_proba(X)[0]
    pred  = int(np.argmax(proba)) + 1
    labels = {1:"Uninjured",2:"Killed",3:"Hospitalised",4:"Light Injury"}
    return {"prediction": pred, "label": labels[pred],
            "confidence": round(float(proba.max()), 3)}

@app.get("/health")
def health():
    return {"status": "ok"}
        """, language="python")

    st.markdown("**🧪 Try prediction UI:**")
    st.caption("By default this uses a deterministic demo fallback. Enter a running FastAPI base URL, for example http://localhost:8000, to call the real /predict endpoint.")
    api_url = st.text_input("Optional FastAPI base URL", value="", placeholder="http://localhost:8000")

    with st.form("inference_form"):
        fc1,fc2,fc3,fc4 = st.columns(4)
        lum  = fc1.selectbox("Lighting (lum)",  [1,2,3,4,5], help="1=Daylight")
        agg  = fc2.selectbox("In town? (agg)",  [1,2],       help="1=Outside, 2=Inside")
        atm  = fc3.selectbox("Weather (atm)",   [1,2,3,4,5], help="1=Normal")
        col  = fc4.selectbox("Collision (col)", [1,2,3,4,5,6,7])
        fc5,fc6,fc7,fc8 = st.columns(4)
        age  = fc5.number_input("Age", 16, 95, 35)
        hour = fc6.number_input("Hour", 0, 23, 14)
        sexe = fc7.selectbox("Sex", [1,2], help="1=Male, 2=Female")
        surf = fc8.selectbox("Surface (surf)", [1,2,3,4,5])
        submitted = st.form_submit_button("🔮  Predict Severity")

    if submitted:
        is_night = 1 if (hour < 6 or hour >= 22) else 0
        is_weekend = 0
        payload = {
            "mois": 5,
            "jour": 12,
            "hour": int(hour),
            "lum": int(lum),
            "int": 1,
            "atm": int(atm),
            "col": int(col),
            "catr": 2,
            "circ": 2,
            "nbv": 2,
            "vosp": 0,
            "surf": int(surf),
            "infra": 0,
            "situ": 1,
            "lat": 48.8566,
            "long": 2.3522,
            "place": 1,
            "catu": 1,
            "sexe": int(sexe),
            "locp": 0,
            "actp": 0,
            "etatp": 1,
            "catv": 7,
            "victim_age": int(age),
        }
        labels_map = {1:"🟢 Uninjured",2:"🔴 Killed",3:"🟠 Hospitalised",4:"🟡 Light Injury"}

        if api_url.strip():
            with st.spinner("Calling FastAPI /predict endpoint…"):
                result = try_api_prediction(api_url, payload)
            if result and "error" not in result:
                pred = int(result.get("prediction", 0))
                label = result.get("severity") or result.get("label") or labels_map.get(pred + 1, "Unknown")
                conf = result.get("confidence", None)
                conf_text = f" — confidence {float(conf):.1%}" if conf is not None else ""
                st.success(f"**Real API response:** {label}{conf_text}")
                st.json(result)
            else:
                st.error("Could not reach the API endpoint. Showing deterministic demo fallback instead.")
                if result:
                    st.caption(result.get("error", "Unknown API error"))
                np.random.seed(age + hour)
                proba = np.random.dirichlet([1,1,3,2])
                pred  = int(np.argmax(proba)) + 1
                st.warning(f"**Demo fallback prediction:** {labels_map[pred]} — confidence {proba.max():.1%}")
                prob_df = pd.DataFrame({"Class":["Uninjured","Killed","Hospitalised","Light Injury"], "Probability": proba.round(3)})
                fig = px.bar(prob_df, x="Class", y="Probability", color="Probability", color_continuous_scale="Oranges")
                apply_theme(fig)
                st.plotly_chart(fig, use_container_width=True)
        else:
            np.random.seed(age + hour)
            proba = np.random.dirichlet([1,1,3,2])
            pred  = int(np.argmax(proba)) + 1
            st.info("No API URL provided, so this is a deterministic demo fallback rather than the real trained model.")
            st.success(f"**Demo Severity:** {labels_map[pred]} — confidence {proba.max():.1%}")
            prob_df = pd.DataFrame({"Class":["Uninjured","Killed","Hospitalised","Light Injury"],
                                     "Probability": proba.round(3)})
            fig = px.bar(prob_df, x="Class", y="Probability", color="Probability",
                         color_continuous_scale="Oranges")
            apply_theme(fig)
            st.plotly_chart(fig, use_container_width=True)

# ══════════════════════════════════════════════════════════════════════════════
# PAGE 4 — MICROSERVICES & MLFLOW
# ══════════════════════════════════════════════════════════════════════════════
elif page == "🔗  Microservices & MLflow":
    st.markdown('<div class="page-title">Microservices, Tracking & Versioning</div>', unsafe_allow_html=True)
    st.markdown('<div class="page-subtitle">Team decomposition · MLflow · DVC · Model Registry · Orchestration</div>', unsafe_allow_html=True)
    st.warning("MLflow tracking worked in project testing. DVC is intentionally marked as a cleanup item because fresh-clone reproducibility still needs fixes.")

    # Team split
    st.markdown('<div class="section-title">Team Microservice Ownership</div>', unsafe_allow_html=True)
    services = [
        ("🗃️", "Ingestion Service",  "Mohammad Reza", "Loads raw CSV files, validates schema, merges tables, writes to data/raw/"),
        ("⚙️", "Processing Service", "Mohammad Reza", "make_dataset.py — cleaning, feature engineering, time split, DVC stage"),
        ("🤖", "Training Service",   "Mohammad Reza", "train_model.py — XGBoost training, MLflow logging, model serialisation"),
        ("📊", "Tracking Service",   "Megha Panchal", "MLflow experiment server, run comparison, metric dashboards"),
        ("🚀", "Inference Service",  "Megha Panchal", "FastAPI /predict endpoint, Docker container, health check"),
        ("🐳", "Infra / DevOps",     "Megha Panchal", "Docker Compose, CI/CD GitHub Actions, NGINX, Kubernetes manifests"),
    ]
    cols = st.columns(3)
    for i, (icon, name, owner, desc) in enumerate(services):
        with cols[i % 3]:
            st.markdown(f"""
            <div class='metric-card' style='min-height:130px;'>
              <div style='font-size:1.6rem;margin-bottom:6px;'>{icon}</div>
              <div style='font-family:Syne,sans-serif;font-weight:700;
                          color:#f1f5f9;font-size:0.95rem;'>{name}</div>
              <div style='font-family:DM Mono,monospace;font-size:0.72rem;
                          color:#f97316;margin:3px 0;'>@{owner}</div>
              <div style='font-size:0.78rem;color:#64748b;line-height:1.5;'>{desc}</div>
            </div>
            """, unsafe_allow_html=True)

    # Architecture diagram
    st.markdown("<hr class='fancy-divider'>", unsafe_allow_html=True)
    st.markdown('<div class="section-title">Microservice Architecture</div>', unsafe_allow_html=True)
    st.graphviz_chart("""
    digraph arch {
      rankdir=LR;
      node [shape=box style=filled fontname="Helvetica" fontsize=10]

      RawData   [label="Raw Data\n(data.gouv.fr)" fillcolor="#1e293b" fontcolor="white"]
      Ingest    [label="Ingestion\nService"        fillcolor="#7c3aed" fontcolor="white"]
      Process   [label="Processing\nService"       fillcolor="#c2410c" fontcolor="white"]
      Train     [label="Training\nService"         fillcolor="#c2410c" fontcolor="white"]
      MLflow    [label="MLflow\nTracking"           fillcolor="#0369a1" fontcolor="white"]
      Registry  [label="Model\nRegistry"            fillcolor="#0369a1" fontcolor="white"]
      Infer     [label="Inference\nAPI (FastAPI)"   fillcolor="#166534" fontcolor="white"]
      Client    [label="Client /\nStreamlit UI"     fillcolor="#374151" fontcolor="white"]
      Monitor   [label="Monitoring\n(Prometheus)"   fillcolor="#1e293b" fontcolor="white"]

      RawData -> Ingest -> Process -> Train
      Train   -> MLflow
      Train   -> Registry -> Infer -> Client
      Infer   -> Monitor
      Process -> MLflow
    }
    """)

    st.markdown("<hr class='fancy-divider'>", unsafe_allow_html=True)

    # MLflow runs
    st.markdown('<div class="section-title">MLflow Experiment Runs</div>', unsafe_allow_html=True)
    np.random.seed(7)
    mlflow_df = pd.DataFrame({
        "Run ID":       [f"a{i}b{i*3}c" for i in range(1,9)],
        "Experiment":   ["road_accidents"] * 8,
        "Model":        ["XGBoost","XGBoost","XGBoost","RandomForest","XGBoost","XGBoost","LightGBM","XGBoost"],
        "n_estimators": [100,200,300,200,300,300,300,300],
        "max_depth":    [4,6,6,10,6,8,6,6],
        "Accuracy":     np.round(np.random.uniform(0.55, 0.63, 8), 3),
        "F1-Weighted":  np.round(np.random.uniform(0.52, 0.60, 8), 3),
        "Log-Loss":     np.round(np.random.uniform(0.95, 1.10, 8), 3),
        "Stage":        ["Archived"]*5+["Staging","Archived","Production"],
    })

    def stage_colour(val):
        if val == "Production": return "background:#14532d;color:#86efac;"
        if val == "Staging":    return "background:#713f12;color:#fde68a;"
        return "background:#1e293b;color:#475569;"

    st.dataframe(
        mlflow_df.style.map(stage_colour, subset=["Stage"]),
        use_container_width=True, hide_index=True
    )

    col_l, col_r = st.columns(2)
    with col_l:
        fig = px.scatter(mlflow_df, x="Log-Loss", y="Accuracy",
                         color="Model", size=[8]*8,
                         hover_data=["Run ID","Stage"],
                         title="Run Comparison: Loss vs Accuracy")
        apply_theme(fig)
        st.plotly_chart(fig, use_container_width=True)
    with col_r:
        fig2 = px.bar(mlflow_df, x="Run ID", y="F1-Weighted",
                      color="Stage", title="F1-Weighted by Run",
                      color_discrete_map={"Production":"#22c55e","Staging":"#fbbf24","Archived":"#475569"})
        apply_theme(fig2)
        st.plotly_chart(fig2, use_container_width=True)

    st.markdown("<hr class='fancy-divider'>", unsafe_allow_html=True)

    # DVC versioning
    st.markdown('<div class="section-title">Data Versioning — DVC Pipeline</div>', unsafe_allow_html=True)
    with st.expander("📄  dvc.yaml — Pipeline Definition"):
        st.code("""
stages:
  prepare:
    cmd: python src/make_dataset.py
    deps:
      - src/make_dataset.py
      - data/caracteristiques/
      - data/lieux/
      - data/usagers/
      - data/vehicules/
    outs:
      - data/preprocessed/X_train.csv
      - data/preprocessed/X_test.csv
      - data/preprocessed/y_train.csv
      - data/preprocessed/y_test.csv

  train:
    cmd: python src/train_model.py
    deps:
      - src/train_model.py
      - data/preprocessed/X_train.csv
      - data/preprocessed/y_train.csv
    outs:
      - models/xgb_model.pkl
    metrics:
      - metrics/train_metrics.json

  evaluate:
    cmd: python src/evaluate_model.py
    deps:
      - src/evaluate_model.py
      - models/xgb_model.pkl
      - data/preprocessed/X_test.csv
      - data/preprocessed/y_test.csv
    metrics:
      - metrics/eval_metrics.json:
          cache: false
        """, language="yaml")

    st.markdown("**Model Registry Stages:**")
    reg_df = pd.DataFrame({
        "Version": ["v1.0","v1.1","v1.2","v2.0"],
        "Stage":   ["Archived","Archived","Staging","Production"],
        "Accuracy":[0.572,0.591,0.603,0.612],
        "Date":    ["2026-01-10","2026-02-05","2026-03-01","2026-03-20"],
    })
    st.dataframe(reg_df, use_container_width=True, hide_index=True)

    # Orchestration
    st.markdown("<hr class='fancy-divider'>", unsafe_allow_html=True)
    st.markdown('<div class="section-title">Orchestration — Airflow-style DAG</div>', unsafe_allow_html=True)
    st.graphviz_chart("""
    digraph dag {
      rankdir=LR;
      node [shape=ellipse style=filled fontname="Helvetica" fontsize=10]
      A [label="ingest_raw_data"     fillcolor="#7c3aed" fontcolor="white"]
      B [label="validate_schema"     fillcolor="#c2410c" fontcolor="white"]
      C [label="run_make_dataset"    fillcolor="#c2410c" fontcolor="white"]
      D [label="train_xgboost"       fillcolor="#c2410c" fontcolor="white"]
      E [label="evaluate_model"      fillcolor="#0369a1" fontcolor="white"]
      F [label="log_to_mlflow"       fillcolor="#0369a1" fontcolor="white"]
      G [label="promote_or_reject"   fillcolor="#166534" fontcolor="white"]
      H [label="deploy_inference_api" fillcolor="#166534" fontcolor="white"]
      A->B->C->D->E->F->G->H
    }
    """)

# ══════════════════════════════════════════════════════════════════════════════
# PAGE 5 — ORCHESTRATION & DEPLOYMENT
# ══════════════════════════════════════════════════════════════════════════════
elif page == "🚀  Orchestration & Deploy":
    st.markdown('<div class="page-title">Orchestration & Deployment</div>', unsafe_allow_html=True)
    st.markdown('<div class="page-subtitle">CI/CD · Docker Compose · nginx cleanup · deployment blueprint</div>', unsafe_allow_html=True)
    st.warning("Docker image build and Docker Compose were tested. nginx HTTPS is not presented as complete because the private key file is missing. Kubernetes is shown as a future/extension blueprint.")

    # CI Pipeline
    st.markdown('<div class="section-title">GitHub Actions — CI Pipeline</div>', unsafe_allow_html=True)
    with st.expander("📄  .github/workflows/ci.yml", expanded=True):
        st.code("""
name: AccidentML CI Pipeline

on:
  push:
    branches: [master, develop]
  pull_request:
    branches: [master]

jobs:
  lint-and-test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3

      - name: Set up Python 3.10
        uses: actions/setup-python@v4
        with: { python-version: "3.10" }

      - name: Install dependencies
        run: pip install -r requirements.txt

      - name: Lint with flake8
        run: flake8 src/ tests/ --max-line-length 120

      - name: Run unit tests
        run: pytest tests/ -v --tb=short

      - name: DVC repro (pipeline check)
        run: |
          pip install dvc
          dvc repro --dry

  build-and-push:
    needs: lint-and-test
    runs-on: ubuntu-latest
    if: github.ref == 'refs/heads/master'
    steps:
      - uses: actions/checkout@v3

      - name: Build Docker image
        run: docker build -t accidentml-api:${{ github.sha }} .

      - name: Push to registry
        run: |
          echo ${{ secrets.DOCKER_PASSWORD }} | docker login -u ${{ secrets.DOCKER_USER }} --password-stdin
          docker tag accidentml-api:${{ github.sha }} myregistry/accidentml-api:latest
          docker push myregistry/accidentml-api:latest
        """, language="yaml")

    st.markdown("<hr class='fancy-divider'>", unsafe_allow_html=True)

    # NGINX
    st.markdown('<div class="section-title">Reverse Proxy — nginx HTTPS cleanup item</div>', unsafe_allow_html=True)
    col_l, col_r = st.columns(2)
    with col_l:
        with st.expander("📄  nginx.conf"):
            st.code("""
upstream accidentml_api {
    server inference-service:8000;
}

server {
    listen 443 ssl http2;
    server_name api.accidentml.io;

    ssl_certificate     /etc/ssl/certs/cert.pem;
    ssl_certificate_key /etc/ssl/private/key.pem;
    ssl_protocols       TLSv1.2 TLSv1.3;

    # Rate limiting: 20 req/s per IP
    limit_req_zone $binary_remote_addr zone=api:10m rate=20r/s;
    limit_req zone=api burst=40 nodelay;

    location /predict {
        proxy_pass         http://accidentml_api;
        proxy_set_header   Host $host;
        proxy_set_header   X-Real-IP $remote_addr;
        proxy_read_timeout 30s;
        add_header         X-Content-Type-Options nosniff;
        add_header         X-Frame-Options DENY;
    }

    location /health {
        proxy_pass http://accidentml_api/health;
    }
}

server {
    listen 80;
    return 301 https://$host$request_uri;
}
            """, language="nginx")
    with col_r:
        st.markdown("""
        <div class='metric-card'>
          <div class='section-label'>Configured / intended layers</div>
          <div style='margin-top:0.8rem;'>
            <span class='badge badge-red'>TLS pending: missing key.pem</span>
            <span class='badge badge-green'>Rate Limiting</span>
            <span class='badge badge-red'>HTTP→HTTPS pending</span>
            <span class='badge badge-green'>X-Frame-Options</span>
            <span class='badge badge-green'>NOSNIFF header</span>
          </div>
          <div class='section-label' style='margin-top:1rem;'>Optimisations</div>
          <div style='margin-top:0.5rem;'>
            <span class='badge badge-blue'>HTTP/2</span>
            <span class='badge badge-blue'>Upstream keepalive</span>
            <span class='badge badge-blue'>30s proxy timeout</span>
            <span class='badge badge-blue'>Burst=40 queue</span>
          </div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<hr class='fancy-divider'>", unsafe_allow_html=True)

    # Docker
    st.markdown('<div class="section-title">Containerisation — Docker</div>', unsafe_allow_html=True)
    tab1, tab2, tab3 = st.tabs(["🐳 Dockerfile", "🐙 docker-compose.yml", "☸️ K8s Deployment"])

    with tab1:
        st.code("""
FROM python:3.10-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY models/ ./models/
COPY src/api/ ./src/api/

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=10s CMD curl -f http://localhost:8000/health || exit 1

CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "4"]
        """, language="docker")

    with tab2:
        st.code("""
version: "3.9"
services:

  nginx:
    image: nginx:1.25-alpine
    ports: ["443:443", "80:80"]
    volumes:
      - ./nginx.conf:/etc/nginx/nginx.conf:ro
      - ./ssl:/etc/ssl:ro
    depends_on: [inference]

  inference:
    build: .
    image: accidentml-api:latest
    environment:
      - MODEL_PATH=/app/models/xgb_model.pkl
    volumes: ["./models:/app/models"]
    deploy:
      replicas: 2
      resources:
        limits: { cpus: "1.0", memory: "512M" }

  mlflow:
    image: ghcr.io/mlflow/mlflow:latest
    ports: ["5000:5000"]
    environment:
      - MLFLOW_BACKEND_STORE_URI=sqlite:///mlflow.db
    volumes: ["mlflow_data:/mlflow"]

  prometheus:
    image: prom/prometheus:latest
    volumes: ["./prometheus.yml:/etc/prometheus/prometheus.yml"]
    ports: ["9090:9090"]

  grafana:
    image: grafana/grafana:latest
    ports: ["3000:3000"]
    depends_on: [prometheus]

volumes:
  mlflow_data:
        """, language="yaml")

    with tab3:
        st.code("""
# k8s/deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: accidentml-inference
  labels: { app: accidentml }
spec:
  replicas: 3
  selector:
    matchLabels: { app: accidentml }
  template:
    metadata:
      labels: { app: accidentml }
    spec:
      containers:
        - name: inference
          image: myregistry/accidentml-api:latest
          ports: [{ containerPort: 8000 }]
          resources:
            requests: { cpu: "250m", memory: "256Mi" }
            limits:   { cpu: "1000m", memory: "512Mi" }
          livenessProbe:
            httpGet: { path: /health, port: 8000 }
            initialDelaySeconds: 10
            periodSeconds: 30
          env:
            - name: MODEL_PATH
              value: /app/models/xgb_model.pkl
---
apiVersion: v1
kind: Service
metadata:
  name: accidentml-service
spec:
  selector: { app: accidentml }
  ports: [{ port: 80, targetPort: 8000 }]
  type: LoadBalancer
---
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: accidentml-hpa
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: accidentml-inference
  minReplicas: 2
  maxReplicas: 10
  metrics:
    - type: Resource
      resource:
        name: cpu
        target: { type: Utilization, averageUtilization: 70 }
        """, language="yaml")

    st.markdown("<hr class='fancy-divider'>", unsafe_allow_html=True)
    st.markdown('<div class="section-title">Deployment Architecture Blueprint</div>', unsafe_allow_html=True)
    st.graphviz_chart("""
    digraph deploy {
      rankdir=TB;
      node [shape=box style=filled fontname="Helvetica" fontsize=9]
      Internet [label="Internet / Client" fillcolor="#1e293b" fontcolor="#94a3b8"]
      NGINX    [label="NGINX\n(TLS + Rate Limit)" fillcolor="#7c3aed" fontcolor="white"]
      K8S      [label="Kubernetes Cluster\n(3 inference pods)" fillcolor="#c2410c" fontcolor="white"]
      MLflow   [label="MLflow Server\n(experiment tracking)" fillcolor="#0369a1" fontcolor="white"]
      Prom     [label="Prometheus\n(metrics scrape)" fillcolor="#166534" fontcolor="white"]
      Grafana  [label="Grafana\n(dashboards)" fillcolor="#166534" fontcolor="white"]
      DVC      [label="DVC Remote\n(GDrive / S3)" fillcolor="#374151" fontcolor="#94a3b8"]

      Internet -> NGINX -> K8S
      K8S -> MLflow [style=dashed label="log runs"]
      K8S -> Prom   [style=dashed label="metrics"]
      Prom -> Grafana
      K8S -> DVC    [style=dashed label="pull model"]
    }
    """)

# ══════════════════════════════════════════════════════════════════════════════
# PAGE 6 — MONITORING & MAINTENANCE
# ══════════════════════════════════════════════════════════════════════════════
elif page == "📡  Monitoring & Maintenance":
    st.markdown('<div class="page-title">Monitoring & Maintenance</div>', unsafe_allow_html=True)
    st.markdown('<div class="page-subtitle">Evidently monitoring · representative service metrics · maintenance playbook</div>', unsafe_allow_html=True)
    st.info("Evidently drift monitoring is part of the tested project. Prometheus/Grafana-style service metrics below are representative unless connected to a running stack.")

    # Live-style metrics
    st.markdown('<div class="section-title">Representative System Metrics (demo)</div>', unsafe_allow_html=True)
    c1,c2,c3,c4,c5 = st.columns(5)
    c1.metric("Avg Latency",    "42 ms",   "-3ms ↓")
    c2.metric("Throughput",     "312 req/s","+18 ↑")
    c3.metric("Error Rate",     "0.3%",    "-0.1% ↓")
    c4.metric("Model Accuracy", "61.2%",   "stable")
    c5.metric("Uptime",         "99.97%",  "30d")

    st.markdown("<hr class='fancy-divider'>", unsafe_allow_html=True)

    # Time-series charts
    np.random.seed(12)
    days = pd.date_range("2026-03-01", periods=30)
    col_l, col_r = st.columns(2)

    with col_l:
        latency = 40 + np.random.normal(0, 4, 30).cumsum() * 0.1
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=days, y=np.clip(latency,30,60),
                                 fill="tozeroy", name="Latency (ms)",
                                 line=dict(color="#f97316", width=2)))
        fig.update_layout(title="API Latency — 30 days", **PLOT_THEME)
        apply_theme(fig)
        st.plotly_chart(fig, use_container_width=True)

    with col_r:
        acc_over_time = 0.612 + np.random.normal(0, 0.006, 30)
        fig2 = go.Figure()
        fig2.add_trace(go.Scatter(x=days, y=acc_over_time,
                                  mode="lines+markers", name="Accuracy",
                                  line=dict(color="#3b82f6", width=2)))
        fig2.add_hline(y=0.58, line_dash="dash", line_color="#ef4444",
                       annotation_text="Retrain threshold", annotation_position="top left")
        fig2.update_layout(title="Model Accuracy Over Time", **PLOT_THEME)
        apply_theme(fig2)
        st.plotly_chart(fig2, use_container_width=True)

    # Prometheus config
    with st.expander("📄  prometheus.yml — Scrape Config"):
        st.code("""
global:
  scrape_interval: 15s

scrape_configs:
  - job_name: 'accidentml-inference'
    static_configs:
      - targets: ['inference:8000']
    metrics_path: /metrics

  - job_name: 'nginx'
    static_configs:
      - targets: ['nginx-exporter:9113']
        """, language="yaml")

    st.markdown("<hr class='fancy-divider'>", unsafe_allow_html=True)

    # Evidently drift
    st.markdown('<div class="section-title">Data & Model Drift — Evidently</div>', unsafe_allow_html=True)
    drift_data = {
        "Feature":     ["age","hour","atm","surf","col","lum","catr","sexe"],
        "Drift Score": [0.03, 0.07, 0.12, 0.05, 0.19, 0.04, 0.08, 0.02],
        "Drifted":     [False,False,True,False,True,False,False,False],
        "Ref Mean":    [38.2, 13.4, 1.2, 1.1, 3.4, 2.1, 2.3, 1.4],
        "Curr Mean":   [39.1, 14.0, 1.8, 1.2, 4.1, 2.0, 2.5, 1.4],
    }
    drift_df = pd.DataFrame(drift_data)

    def drift_colour(val):
        if val is True:  return "background:#7f1d1d;color:#fca5a5;"
        if val is False: return "background:#14532d;color:#86efac;"
        return ""

    st.dataframe(
        drift_df.style.map(drift_colour, subset=["Drifted"]),
        use_container_width=True, hide_index=True
    )

    fig = px.bar(drift_df, x="Feature", y="Drift Score",
                 color="Drifted",
                 color_discrete_map={True:"#ef4444", False:"#22c55e"},
                 title="Feature Drift Scores (Evidently PSI)")
    fig.add_hline(y=0.10, line_dash="dash", line_color="#fbbf24",
                  annotation_text="Drift threshold=0.10")
    apply_theme(fig)
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("<hr class='fancy-divider'>", unsafe_allow_html=True)

    # Auto retrain
    st.markdown('<div class="section-title">Automated Retraining Logic</div>', unsafe_allow_html=True)
    st.graphviz_chart("""
    digraph retrain {
      rankdir=LR;
      node [shape=diamond style=filled fontname="Helvetica" fontsize=9]
      D1 [label="Accuracy\n< 0.58?" fillcolor="#7c3aed" fontcolor="white"]
      D2 [label="Drift\ndetected?" fillcolor="#7c3aed" fontcolor="white"]

      node [shape=box]
      Mon  [label="Scheduled\nMonitor\n(daily)" fillcolor="#1e293b" fontcolor="#94a3b8"]
      Log  [label="Log alert\nto MLflow"        fillcolor="#0369a1" fontcolor="white"]
      Trig [label="Trigger DVC\nrepro pipeline" fillcolor="#c2410c" fontcolor="white"]
      Eval [label="Evaluate\nnew model"         fillcolor="#c2410c" fontcolor="white"]
      Cmp  [label="Compare vs\nProduction"      fillcolor="#0369a1" fontcolor="white"]
      Promo[label="Promote to\nProduction"      fillcolor="#166534" fontcolor="white"]
      Keep [label="Keep current\nmodel"         fillcolor="#374151" fontcolor="#94a3b8"]

      Mon -> D1
      D1  -> D2   [label="No"]
      D1  -> Log  [label="Yes"]
      D2  -> Keep [label="No"]
      D2  -> Log  [label="Yes"]
      Log -> Trig -> Eval -> Cmp
      Cmp -> Promo [label="Better"]
      Cmp -> Keep  [label="Worse"]
    }
    """)

    with st.expander("📄  Auto-retrain script sketch"):
        st.code("""
# monitoring/auto_retrain.py  (run via cron or Airflow)
import subprocess, mlflow, json

ACCURACY_THRESHOLD = 0.58
DRIFT_THRESHOLD    = 0.10

def check_and_retrain():
    # 1. Load latest metrics
    with open("metrics/eval_metrics.json") as f:
        metrics = json.load(f)

    # 2. Load drift report
    with open("reports/drift.json") as f:
        drift = json.load(f)

    accuracy_ok = metrics["accuracy"] >= ACCURACY_THRESHOLD
    drift_ok    = all(v < DRIFT_THRESHOLD for v in drift["drift_scores"].values())

    if not accuracy_ok or not drift_ok:
        print("⚠️  Retraining triggered")
        subprocess.run(["dvc", "repro"], check=True)
        new_acc = json.load(open("metrics/eval_metrics.json"))["accuracy"]
        if new_acc > metrics["accuracy"]:
            mlflow.register_model("models:/xgb_model/latest", "Production")
            print(f"✅ New model promoted: {new_acc:.3f}")
        else:
            print("ℹ️  New model not better, keeping current")
    else:
        print("✅ No retraining needed")

if __name__ == "__main__":
    check_and_retrain()
        """, language="python")

    st.markdown("<hr class='fancy-divider'>", unsafe_allow_html=True)

    # Documentation
    st.markdown('<div class="section-title">📚 Technical Documentation</div>', unsafe_allow_html=True)
    with st.expander("README — Setup & Architecture", expanded=True):
        st.markdown("""
## 🚦 AccidentML — Road Accident Severity Prediction

**Goal:** Predict accident severity (`grav`) in France using historical national data (2005–2020).

---

### Current Status
Core ML pipeline, FastAPI, Docker/Docker Compose, MLflow and Evidently were tested. DVC fresh-clone reproducibility and nginx HTTPS need cleanup.

### ⚙️ Quick Start
```bash
git clone https://github.com/Megha-2023/mar26bmlops_int_accidents
cd mar26bmlops_int_accidents
pip install -r requirements.txt
dvc repro            # intended full pipeline; fresh-clone reproducibility currently needs cleanup
streamlit run app.py # launch this dashboard
```

---

### 🏗️ Architecture
```
Raw Data (data.gouv.fr)
    └─▶ Ingestion → Processing → Training → Evaluation
                                     └─▶ MLflow Tracking
                                     └─▶ Model Registry
                                              └─▶ Inference API (FastAPI)
                                                       └─▶ NGINX cleanup → deployment blueprint
                                                       └─▶ Prometheus → Grafana
```

---

### 📡 API Reference
| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/predict` | Predict severity from features |
| GET  | `/health`  | Health check |
| GET  | `/metrics` | Prometheus metrics |

**POST `/predict` body:**
```json
{"lum":1,"agg":2,"atm":1,"col":3,"catr":2,"surf":1,"age":35,"sexe":1,"hour":14,"is_night":0,"is_weekend":0}
```

---

        """)

    with st.expander("🔧 Component Update Checklist"):
        checklist = [
            ("Model", "Monthly retrain check via auto_retrain.py"),
            ("Data",  "Pull new data.gouv.fr release each January"),
            ("API",   "Version bump + Docker rebuild on model change"),
            ("Deps",  "Dependabot auto-PRs for security patches"),
            ("Docs",  "Update README on each merged PR"),
        ]
        for comp, action in checklist:
            st.checkbox(f"**{comp}** — {action}", value=True)
