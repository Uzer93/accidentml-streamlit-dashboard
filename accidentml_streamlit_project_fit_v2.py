# =============================================================================
# MLOps Dashboard — Road Accident Severity Prediction (France)
# Team: Mohammad Reza Nilchiyan | Megha Panchal | Ahmad Melhem
# Run: streamlit run accidentml_streamlit_project_fit_v2.py
# =============================================================================

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import time
import json
from pathlib import Path
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

# ─── PROJECT STATUS / APP HELPERS ────────────────────────────────────────────
PROJECT_REPO = "https://github.com/Megha-2023/mar26bmlops_int_accidents"

STATUS_ROWS = [
    ("Core ML pipeline", "Verified", "Data preprocessing, XGBoost training, evaluation, model artifacts and evaluation outputs were run successfully."),
    ("FastAPI inference service", "Verified", "Local API serving worked. Endpoints tested: /, /health, /model-info and /predict."),
    ("Evidently monitoring", "Verified", "Data drift and prediction drift monitoring were integrated and tested on the monitoring branch."),
    ("Unit tests / pytest", "Verified", "Base branch tests passed, and monitoring branch tests passed with meaningful API and pipeline coverage."),
    ("MLflow tracking + registry", "Verified with limits", "Docker MLflow server worked; runs, parameters, metrics, artifacts and model registration were visible. File-based backend limits advanced behavior."),
    ("Docker image build", "Verified", "Docker image build succeeded; build was slow because the context was too large, which is a recorded engineering finding."),
    ("Docker Compose stack", "Partially verified", "Compose services present: mlflow, dvc, api and nginx. mlflow/dvc/api worked; nginx still needs cleanup for a stable reverse-proxy claim."),
    ("DVC pipeline", "Partially verified", "dvc repro worked inside the DVC container, but fresh-clone reproducibility and remote pull setup need cleanup."),
    ("nginx deployment layer", "Needs cleanup", "Reverse-proxy layer exists, but HTTPS/private-key setup and end-to-end stability were not fully validated."),
    ("Airflow / Prometheus / Grafana", "Not validated", "Mentioned as intended/team infrastructure only; not claimed as fully working in this review."),
]

def status_badge(status: str) -> str:
    low = status.lower()
    if low == "verified" or "verified with limits" in low:
        klass = "badge badge-green"
    elif "needs" in low or "not validated" in low:
        klass = "badge badge-red"
    else:
        klass = "badge badge-blue"
    return f"<span class='{klass}'>{status}</span>"

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

# ─── DATA LOADERS / GENERATORS ───────────────────────────────────────────────
PROJECT_DIR = Path(__file__).resolve().parent
ACCIDENTS_FULL_PATH = PROJECT_DIR / "data" / "accidents_full.csv"
PREPROCESSED_DIR = PROJECT_DIR / "data" / "preprocessed"
DASHBOARD_ARTIFACTS_DIR = PROJECT_DIR / "dashboard_artifacts"
METRICS_PATH = PROJECT_DIR / "metrics" / "metrics.json"
CLASSIFICATION_REPORT_PATH = PROJECT_DIR / "metrics" / "classification_report.txt"
CONFUSION_MATRIX_PATH = PROJECT_DIR / "metrics" / "plots" / "confusion_matrix.png"
ROC_CURVE_PATH = PROJECT_DIR / "metrics" / "plots" / "roc_curve.png"
EVIDENTLY_REPORT_PATH = PROJECT_DIR / "reports" / "xtrain_vs_xtest_drift_report.html"

@st.cache_data(show_spinner="Loading BAAC merged dataset...")
def load_eda_data():
    """Load the merged accident dataset columns used by the EDA page."""
    if not ACCIDENTS_FULL_PATH.exists():
        return pd.DataFrame(), f"Missing dataset file: {ACCIDENTS_FULL_PATH}"

    usecols = [
        "num_acc", "an", "mois", "jour", "hrmn", "lum", "agg", "int",
        "atm", "col", "lat", "long", "catr", "circ", "nbv", "vosp",
        "prof", "plan", "surf", "infra", "situ", "sexe", "catv",
        "grav", "an_nais"
    ]
    try:
        df = pd.read_csv(ACCIDENTS_FULL_PATH, usecols=usecols, low_memory=False)
    except Exception as exc:
        return pd.DataFrame(), str(exc)

    df["an"] = pd.to_numeric(df["an"], errors="coerce")
    df.loc[df["an"].between(0, 99), "an"] = df.loc[df["an"].between(0, 99), "an"] + 2000
    df["hour"] = (pd.to_numeric(df["hrmn"], errors="coerce") // 100).clip(0, 23)
    df["age"] = df["an"] - pd.to_numeric(df["an_nais"], errors="coerce")
    df.loc[~df["age"].between(0, 120), "age"] = np.nan
    return df, None

@st.cache_data
def load_preprocessed_split(split: str):
    x_path = PREPROCESSED_DIR / f"X_{split}.csv"
    y_path = PREPROCESSED_DIR / f"y_{split}.csv"
    if not x_path.exists() or not y_path.exists():
        return pd.DataFrame(), pd.Series(dtype="float64"), f"Missing preprocessed files for split: {split}"
    X = pd.read_csv(x_path)
    y_df = pd.read_csv(y_path)
    y = y_df.iloc[:, 0] if not y_df.empty else pd.Series(dtype="float64")
    return X, y, None

@st.cache_data
def load_metrics():
    if not METRICS_PATH.exists():
        return {}
    with open(METRICS_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


@st.cache_data
def load_json_artifact(path: Path):
    if not path.exists():
        return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


@st.cache_data
def load_csv_artifact(path: Path):
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


def count_test_functions(path: Path) -> int:
    try:
        return sum(1 for line in path.read_text(encoding="utf-8").splitlines() if line.strip().startswith("def test_"))
    except Exception:
        return 0

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
        "🔗  Services & MLflow",
        "🚀  Docker & Deployment",
        "📡  Monitoring & Status",
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
    st.markdown('<div class="page-title">Validation Summary</div>', unsafe_allow_html=True)
    st.markdown('<div class="page-subtitle">Current validation state of the AccidentML graduation MLOps project</div>', unsafe_allow_html=True)
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("ML Task", "Severity prediction", "multiclass grav")
    c2.metric("Model", "XGBoost", "verified baseline")
    c3.metric("Serving", "FastAPI", "endpoints tested")
    c4.metric("Monitoring", "Evidently", "drift verified")

    st.markdown("<hr class='fancy-divider'>", unsafe_allow_html=True)
    st.markdown('<div class="section-title">Component Validation Matrix</div>', unsafe_allow_html=True)

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
    st.markdown('<div class="section-title">How to read this dashboard</div>', unsafe_allow_html=True)
    st.markdown("""
    - **Verified** means the component was actually run and tested during the project review work.
    - **Partially verified** means the core implementation worked, but reproducibility, deployment stability, or newcomer setup still needs cleanup.
    - **Data charts** are computed from local files in this repository. When a file or field is unavailable, the dashboard states that directly.
    - **Not validated** infrastructure is shown only as intended/team context, not as a working production claim.
    """)

    st.markdown("<hr class='fancy-divider'>", unsafe_allow_html=True)
    st.markdown('<div class="section-title">Recommended Reviewer Flow</div>', unsafe_allow_html=True)
    reviewer_flow = pd.DataFrame([
        ("1", "Project Status", "Start with the validation matrix and cleanup items."),
        ("2", "Dataset & EDA", "Inspect BAAC-derived counts and missingness."),
        ("3", "Data Processing", "Review preprocessed train/test artifacts."),
        ("4", "Baseline Model", "Review metrics and saved plot artifacts."),
        ("5", "Services & MLflow", "Review verified service capabilities and DVC stages."),
        ("6", "Docker & Deployment", "Review actual Docker/Compose/nginx files."),
        ("7", "Monitoring & Status", "Review Evidently artifact and final status."),
    ], columns=["Step", "Page", "Evidence to inspect"])
    st.dataframe(reviewer_flow, use_container_width=True, hide_index=True)

# ══════════════════════════════════════════════════════════════════════════════
# PAGE 1 — DATASET & EDA
# ══════════════════════════════════════════════════════════════════════════════
elif page == "📊  Dataset & EDA":
    st.markdown('<div class="page-title">Dataset & Exploration</div>', unsafe_allow_html=True)
    st.markdown('<div class="page-subtitle">French Road Accident Data (2005–2020) · Source: data.gouv.fr</div>', unsafe_allow_html=True)

    eda_meta = load_json_artifact(DASHBOARD_ARTIFACTS_DIR / "eda_meta.json")
    eda_preview = load_csv_artifact(DASHBOARD_ARTIFACTS_DIR / "eda_preview.csv")
    eda_severity = load_csv_artifact(DASHBOARD_ARTIFACTS_DIR / "eda_severity_counts.csv")
    eda_year = load_csv_artifact(DASHBOARD_ARTIFACTS_DIR / "eda_year_counts.csv")
    eda_hour = load_csv_artifact(DASHBOARD_ARTIFACTS_DIR / "eda_hour_counts.csv")
    eda_missing = load_csv_artifact(DASHBOARD_ARTIFACTS_DIR / "eda_missing_counts.csv")
    eda_corr = load_csv_artifact(DASHBOARD_ARTIFACTS_DIR / "eda_correlation_matrix.csv")
    eda_age_severity = load_csv_artifact(DASHBOARD_ARTIFACTS_DIR / "eda_age_severity.csv")
    use_artifacts = eda_meta is not None and not eda_preview.empty and not eda_severity.empty and not eda_year.empty and not eda_corr.empty

    if not use_artifacts:
        df, load_error = load_eda_data()
        if load_error:
            st.error(f"Cannot load EDA data: {load_error}")
            st.stop()
    else:
        df = None

    # ── KPI row
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Loaded Rows", f"{eda_meta['loaded_rows']:,}" if use_artifacts else f"{len(df):,}", "merged CSV")
    c2.metric("Loaded Features", f"{eda_meta['loaded_features']}" if use_artifacts else f"{df.shape[1]}", "EDA subset")
    c3.metric("Year Range", f"{eda_meta['year_min']}–{eda_meta['year_max']}" if use_artifacts else f"{int(df['an'].min())}–{int(df['an'].max())}", "from CSV")
    c4.metric("Target", "grav", "severity class")

    st.markdown("<hr class='fancy-divider'>", unsafe_allow_html=True)

    # ── Data Story
    with st.expander("📖  Data Story — Why this dataset?", expanded=True):
        st.markdown("""
        <div style='font-family:DM Sans,sans-serif; line-height:1.8; color:#cbd5e1;'>
        The French national road traffic injury accident file (<b>BAAC</b>) is built from reports completed
        by police and gendarmerie units for injury accidents on roads open to public traffic.
        This project uses a merged <b>2005–2020</b> dataset derived from the four BAAC file families:
        <br><br>
        <span class='badge'>caracteristiques</span> accident context (time, weather, location)
        <span class='badge'>lieux</span> road geometry & surface
        <span class='badge'>usagers</span> people involved — <b>grav</b> (severity) lives here
        <span class='badge'>vehicules</span> vehicle types
        <br><br>
        <b>Target variable — <code>grav</code>:</b> 1 = Uninjured · 2 = Killed · 3 = Hospitalised · 4 = Light injury
        <br><br>
        The challenge: <b>class imbalance</b>, year-to-year data definition/schema differences, and temporal
        effects make a naive random split inappropriate. The validated preprocessing pipeline filters the
        modeling data to <b>2010–2016</b> and uses a <b>time-based split</b>:
        train <b>2010–2015</b>, test <b>2016</b>.
        </div>
        """, unsafe_allow_html=True)

    # ── Data preview
    st.markdown('<div class="section-label">Rows</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-title">Merged Dataset Preview</div>', unsafe_allow_html=True)
    st.dataframe(eda_preview if use_artifacts else df.head(12), use_container_width=True)

    st.markdown("<hr class='fancy-divider'>", unsafe_allow_html=True)

    # ── Visualisations
    col_l, col_r = st.columns(2)

    with col_l:
        # Severity distribution
        grav_map = {1:"Uninjured",2:"Killed",3:"Hospitalised",4:"Light Injury"}
        if use_artifacts:
            grav_counts = eda_severity.copy()
        else:
            grav_counts = df["grav"].map(grav_map).fillna(df["grav"].astype(str)).value_counts().reset_index()
            grav_counts.columns = ["Severity","Count"]
        fig = px.bar(grav_counts, x="Severity", y="Count", color="Severity",
                     title="Target — Severity Distribution",
                     color_discrete_sequence=["#22c55e","#ef4444","#f97316","#3b82f6"])
        apply_theme(fig)
        st.plotly_chart(fig, use_container_width=True)

    with col_r:
        # Accidents by year
        if use_artifacts:
            yr = eda_year.copy()
        else:
            yr = df["an"].value_counts().sort_index().reset_index()
            yr.columns = ["Year","Count"]
            yr["Year"] = yr["Year"].astype(int)
        fig2 = px.line(yr, x="Year", y="Count", title="Accident/User Rows by Year",
                       markers=True)
        fig2.update_traces(line_color="#f97316", marker_color="#fbbf24", line_width=3)
        apply_theme(fig2)
        st.plotly_chart(fig2, use_container_width=True)

    col_l2, col_r2 = st.columns(2)

    with col_l2:
        # Accidents by hour
        if use_artifacts and eda_hour.empty:
            st.info("Hour-of-day summary is omitted from the lightweight deployment bundle.")
        else:
            hr = eda_hour.copy() if use_artifacts else df["hour"].value_counts().sort_index().reset_index()
            if not use_artifacts:
                hr.columns = ["Hour","Count"]
            fig3 = px.bar(hr, x="Hour", y="Count", title="Accident/User Rows by Hour of Day",
                          color="Count", color_continuous_scale="Oranges")
            apply_theme(fig3)
            st.plotly_chart(fig3, use_container_width=True)

    with col_r2:
        # Missing value heatmap
        if use_artifacts and eda_missing.empty:
            st.info("Per-feature missing-value summary is omitted from the lightweight deployment bundle.")
        else:
            missing = eda_missing.copy() if use_artifacts else df.isnull().sum().reset_index()
            if not use_artifacts:
                missing.columns = ["Feature","Missing"]
                missing = missing[missing["Missing"] > 0]
            fig4 = px.bar(missing, x="Feature", y="Missing", title="Missing Values per Loaded Feature",
                          color="Missing", color_continuous_scale="Reds")
            apply_theme(fig4)
            st.plotly_chart(fig4, use_container_width=True)

    # Correlation heatmap
    st.markdown('<div class="section-title">Feature Correlation Heatmap</div>', unsafe_allow_html=True)
    if use_artifacts:
        corr = eda_corr.set_index("Feature")
        corr = corr.apply(pd.to_numeric, errors="coerce")
        fig5 = go.Figure(data=go.Heatmap(
            z=corr.values, x=corr.columns, y=corr.index,
            colorscale="RdBu", zmid=0,
            text=corr.values, texttemplate="%{text}",
        ))
        fig5.update_layout(title="Pearson Correlation Matrix", **PLOT_THEME,
                           margin=dict(l=20,r=20,t=40,b=20))
        st.plotly_chart(fig5, use_container_width=True)
    else:
        num_cols = ["lum","agg","atm","col","catr","surf","age","sexe","grav"]
        available_num_cols = [col for col in num_cols if col in df.columns]
        corr_data = df[available_num_cols].apply(pd.to_numeric, errors="coerce")
        if len(corr_data.dropna(how="all")) >= 2:
            corr = corr_data.corr(min_periods=100).round(2)
            fig5 = go.Figure(data=go.Heatmap(
                z=corr.values, x=corr.columns, y=corr.index,
                colorscale="RdBu", zmid=0,
                text=corr.values, texttemplate="%{text}",
            ))
            fig5.update_layout(title="Pearson Correlation Matrix from data/accidents_full.csv", **PLOT_THEME,
                               margin=dict(l=20,r=20,t=40,b=20))
            st.plotly_chart(fig5, use_container_width=True)
        else:
            st.warning("Not enough complete numeric rows are available to compute a reliable correlation matrix.")

    with st.expander("👥  Severity by Age Group"):
        if use_artifacts and eda_age_severity.empty:
            st.info("Age-group severity summary is omitted from the lightweight deployment bundle.")
        else:
            if use_artifacts:
                age_severity = eda_age_severity.copy()
            else:
                age_view = df[["age", "grav"]].copy()
                age_view["age"] = pd.to_numeric(age_view["age"], errors="coerce")
                age_view["grav"] = pd.to_numeric(age_view["grav"], errors="coerce")
                age_view = age_view.dropna(subset=["age", "grav"])

                age_bins = [0, 10, 18, 25, 35, 45, 55, 65, 75, 121]
                age_labels = ["0-9", "10-17", "18-24", "25-34", "35-44", "45-54", "55-64", "65-74", "75+"]
                age_view["Age Group"] = pd.cut(age_view["age"], bins=age_bins, labels=age_labels, right=False, include_lowest=True)
                age_view["Severity"] = age_view["grav"].map(grav_map).fillna(age_view["grav"].astype(int).astype(str))

                age_severity = (
                    age_view.groupby(["Age Group", "Severity"], observed=False)
                    .size()
                    .reset_index(name="Count")
                )

            age_labels = ["0-9", "10-17", "18-24", "25-34", "35-44", "45-54", "55-64", "65-74", "75+"]
            fig6 = px.bar(
                age_severity,
                x="Age Group",
                y="Count",
                color="Severity",
                barmode="group",
                title="Severity Distribution by Age Group",
                category_orders={"Age Group": age_labels, "Severity": ["Uninjured", "Light Injury", "Hospitalised", "Killed"]},
                color_discrete_map={
                    "Uninjured": "#22c55e",
                    "Light Injury": "#ef4444",
                    "Hospitalised": "#f97316",
                    "Killed": "#3b82f6",
                },
            )
            apply_theme(fig6)
            st.plotly_chart(fig6, use_container_width=True)

            age_summary = age_severity.pivot(index="Age Group", columns="Severity", values="Count").fillna(0).reset_index()
            st.dataframe(age_summary, use_container_width=True, hide_index=True)

# ══════════════════════════════════════════════════════════════════════════════
# PAGE 2 — DATA PROCESSING
# ══════════════════════════════════════════════════════════════════════════════
elif page == "⚙️  Data Processing":
    st.markdown('<div class="page-title">Data Processing & Preparation</div>', unsafe_allow_html=True)
    st.markdown('<div class="page-subtitle">Pipeline: Merge → Clean → Feature Engineer → Time Split</div>', unsafe_allow_html=True)
    st.warning("Engineering note: the DVC pipeline works, including containerized dvc repro, but fresh-clone reproducibility and remote artifact restoration were not perfectly smooth and remain cleanup items.")

    # Pipeline steps
    steps = [
        ("01", "Load & Merge", "Join caracteristiques, lieux, usagers, vehicules on Num_Acc"),
        ("02", "Time Filter",  "Keep 2010–2016 only — matches the validated modeling period"),
        ("03", "Drop Columns", "Remove near-constant, ID, and redundant geo columns"),
        ("04", "Handle NaN",   "Median-fill numeric; mode-fill categorical; drop rows >30% missing"),
        ("05", "Encode",       "Ordinal encode categoricals (no OHE — XGBoost handles ordinals)"),
        ("06", "Feature Eng.", "Derive hour from hrmn and victim_age from an minus an_nais."),
        ("07", "Time Split",   "Train: 2010–2015 | Test: 2016 — no shuffle, respects temporal order"),
        ("08", "Monitor Prep", "Prepare reference/current splits used for Evidently data and prediction drift monitoring"),
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

    split_overview = load_json_artifact(DASHBOARD_ARTIFACTS_DIR / "split_overview.json")
    split_train_preview = load_csv_artifact(DASHBOARD_ARTIFACTS_DIR / "split_train_preview.csv")
    split_feature_summary = load_csv_artifact(DASHBOARD_ARTIFACTS_DIR / "split_feature_summary.csv")
    split_class_distribution = load_csv_artifact(DASHBOARD_ARTIFACTS_DIR / "split_class_distribution.csv")
    use_split_artifacts = split_overview is not None and not split_train_preview.empty and not split_feature_summary.empty and not split_class_distribution.empty

    if not use_split_artifacts:
        X_train_df, y_train_series, train_error = load_preprocessed_split("train")
        X_test_df, y_test_series, test_error = load_preprocessed_split("test")
    else:
        X_train_df, y_train_series, train_error = split_train_preview, None, None
        X_test_df, y_test_series, test_error = pd.DataFrame(), None, None

    st.markdown('<div class="section-title">Preprocessed Outputs</div>', unsafe_allow_html=True)
    tab1, tab2, tab3 = st.tabs(["🧹  Training Data", "🔧  Feature Set", "✂️  Train/Test Split"])

    with tab1:
        if train_error:
            st.error(train_error)
        else:
            c1, c2, c3 = st.columns(3)
            c1.metric("X_train rows", f"{split_overview['train_rows']:,}" if use_split_artifacts else f"{len(X_train_df):,}")
            c2.metric("X_train features", f"{split_overview['feature_count']}" if use_split_artifacts else f"{X_train_df.shape[1]}")
            c3.metric("y_train rows", f"{split_overview['y_train_rows']:,}" if use_split_artifacts else f"{len(y_train_series):,}")
            st.markdown("**Preview from training features**")
            st.dataframe(split_train_preview if use_split_artifacts else X_train_df.head(10), use_container_width=True)

    with tab2:
        if train_error:
            st.error(train_error)
        else:
            feature_df = split_feature_summary if use_split_artifacts else pd.DataFrame({
                "Feature": X_train_df.columns,
                "Missing in X_train": X_train_df.isna().sum().values,
                "dtype": [str(dtype) for dtype in X_train_df.dtypes],
            })
            st.dataframe(feature_df, use_container_width=True, hide_index=True)

    with tab3:
        if train_error or test_error:
            st.error(train_error or test_error)
        else:
            split_df = pd.DataFrame([
                ("Train", split_overview["train_rows"], split_overview["feature_count"], split_overview["y_train_rows"]),
                ("Test", split_overview["test_rows"], split_overview["feature_count"], split_overview["y_test_rows"]),
            ], columns=["Split", "Feature rows", "Feature count", "Target rows"]) if use_split_artifacts else pd.DataFrame([
                ("Train", len(X_train_df), X_train_df.shape[1], len(y_train_series)),
                ("Test", len(X_test_df), X_test_df.shape[1], len(y_test_series)),
            ], columns=["Split", "Feature rows", "Feature count", "Target rows"])
            st.dataframe(split_df, use_container_width=True, hide_index=True)
        c1, c2, c3 = st.columns(3)
        c1.metric("Train Samples", f"{split_overview['train_rows']:,}" if use_split_artifacts else (f"{len(X_train_df):,}" if not train_error else "missing"))
        c2.metric("Test Samples",  f"{split_overview['test_rows']:,}" if use_split_artifacts else (f"{len(X_test_df):,}" if not test_error else "missing"))
        c3.metric("Features",      f"{split_overview['feature_count']}" if use_split_artifacts else (f"{X_train_df.shape[1]}" if not train_error else "missing"))

    st.markdown("<hr class='fancy-divider'>", unsafe_allow_html=True)

    # Class distribution post-split
    st.markdown('<div class="section-title">Class Distribution — Train vs Test</div>', unsafe_allow_html=True)
    if train_error or test_error:
        st.error(train_error or test_error)
    else:
        if use_split_artifacts:
            class_df = split_class_distribution[["Class", "Train Share", "Test Share"]].copy()
            class_df.columns = ["Class", "Train", "Test"]
        else:
            train_share = y_train_series.value_counts(normalize=True).sort_index()
            test_share = y_test_series.value_counts(normalize=True).sort_index()
            class_df = pd.DataFrame({
                "Class": sorted(set(train_share.index).union(set(test_share.index))),
            })
            class_df["Train"] = class_df["Class"].map(train_share).fillna(0)
            class_df["Test"] = class_df["Class"].map(test_share).fillna(0)
            class_df["Class"] = class_df["Class"].astype(str)
        fig = px.bar(class_df, x="Class", y=["Train", "Test"], barmode="group",
                     title="Normalised Class Proportions from Training and Test Splits")
        apply_theme(fig)
        st.plotly_chart(fig, use_container_width=True)
        st.dataframe(class_df, use_container_width=True, hide_index=True)

# ══════════════════════════════════════════════════════════════════════════════
# PAGE 3 — BASELINE MODEL
# ══════════════════════════════════════════════════════════════════════════════
elif page == "🤖  Baseline Model":
    st.markdown('<div class="page-title">Baseline ML Model</div>', unsafe_allow_html=True)
    st.markdown('<div class="page-subtitle">Algorithm: XGBoost Classifier · Evaluation · Unit Tests · Inference API</div>', unsafe_allow_html=True)
    st.success("Validated model work: XGBoost training, evaluation, artifact saving, API serving and pytest coverage were run successfully.")

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
        train_path = PROJECT_DIR / "src" / "models" / "train_model.py"
        st.code(train_path.read_text(encoding="utf-8") if train_path.exists() else "src/models/train_model.py not found", language="python")

    st.markdown("<hr class='fancy-divider'>", unsafe_allow_html=True)

    st.markdown('<div class="section-title">Training/Test Class Distribution</div>', unsafe_allow_html=True)
    split_class_distribution = load_csv_artifact(DASHBOARD_ARTIFACTS_DIR / "split_class_distribution.csv")
    split_overview = load_json_artifact(DASHBOARD_ARTIFACTS_DIR / "split_overview.json")
    use_split_artifacts = split_overview is not None and not split_class_distribution.empty
    if not use_split_artifacts:
        X_train_df, y_train_series, train_error = load_preprocessed_split("train")
        X_test_df, y_test_series, test_error = load_preprocessed_split("test")
    else:
        train_error = None
        test_error = None
    if train_error or test_error:
        st.error(train_error or test_error)
    else:
        if use_split_artifacts:
            class_counts = split_class_distribution.copy()
        else:
            class_labels = {
                0: "0",
                1: "1",
                2: "2",
                3: "3",
            }
            train_counts = y_train_series.value_counts().sort_index()
            test_counts = y_test_series.value_counts().sort_index()
            class_counts = pd.DataFrame({
                "Class": sorted(set(train_counts.index).union(set(test_counts.index))),
            })
            class_counts["Train Count"] = class_counts["Class"].map(train_counts).fillna(0).astype(int)
            class_counts["Test Count"] = class_counts["Class"].map(test_counts).fillna(0).astype(int)
            class_counts["Train Share"] = class_counts["Train Count"] / class_counts["Train Count"].sum()
            class_counts["Test Share"] = class_counts["Test Count"] / class_counts["Test Count"].sum()
            class_counts["Class"] = class_counts["Class"].map(lambda v: class_labels.get(v, str(v)))

        col_counts, col_share = st.columns(2)
        with col_counts:
            fig_counts = px.bar(
                class_counts,
                x="Class",
                y=["Train Count", "Test Count"],
                barmode="group",
                title="Class Counts from Training and Test Splits",
            )
            apply_theme(fig_counts)
            st.plotly_chart(fig_counts, use_container_width=True)
        with col_share:
            fig_share = px.bar(
                class_counts,
                x="Class",
                y=["Train Share", "Test Share"],
                barmode="group",
                title="Class Proportions from Training and Test Splits",
            )
            fig_share.update_yaxes(tickformat=".0%")
            apply_theme(fig_share)
            st.plotly_chart(fig_share, use_container_width=True)
        st.dataframe(class_counts, use_container_width=True, hide_index=True)

    st.markdown("<hr class='fancy-divider'>", unsafe_allow_html=True)

    # Metrics
    st.markdown('<div class="section-title">Evaluation Dashboard</div>', unsafe_allow_html=True)
    metrics = load_metrics()
    c1,c2,c3,c4 = st.columns(4)
    c1.metric("Accuracy",  f"{metrics.get('Accuracy', 0):.4f}" if metrics else "missing", "metrics/metrics.json")
    c2.metric("F1-Score",  f"{metrics.get('F1 Score', 0):.4f}" if metrics else "missing", "metrics/metrics.json")
    c3.metric("ROC AUC",   f"{metrics.get('roc_auc', 0):.4f}" if metrics else "missing", "metrics/metrics.json")
    c4.metric("Known Risk", "minority class", "low recall")

    col_l, col_r = st.columns(2)

    with col_l:
        if CONFUSION_MATRIX_PATH.exists():
            st.image(str(CONFUSION_MATRIX_PATH), caption="Artifact: metrics/plots/confusion_matrix.png")
        else:
            st.warning("Confusion matrix artifact is missing.")

    with col_r:
        if ROC_CURVE_PATH.exists():
            st.image(str(ROC_CURVE_PATH), caption="Artifact: metrics/plots/roc_curve.png")
        else:
            st.warning("ROC curve artifact is missing.")

    with st.expander("📄 Classification report artifact"):
        if CLASSIFICATION_REPORT_PATH.exists():
            st.code(CLASSIFICATION_REPORT_PATH.read_text(encoding="utf-8"), language="text")
        else:
            st.warning("classification_report.txt is missing.")

    st.markdown("<hr class='fancy-divider'>", unsafe_allow_html=True)

    # Unit tests
    st.markdown('<div class="section-title">Validated Test Coverage</div>', unsafe_allow_html=True)
    test_files = sorted((PROJECT_DIR / "tests").glob("test_*.py"))
    test_inventory = pd.DataFrame({
        "Test file": [path.name for path in test_files],
        "Path": [str(path.relative_to(PROJECT_DIR)) for path in test_files],
        "Test cases": [count_test_functions(path) for path in test_files],
    })

    c1, c2, c3 = st.columns(3)
    c1.metric("Test files", f"{len(test_files)}")
    c2.metric("Detected test cases", f"{int(test_inventory['Test cases'].sum())}" if not test_inventory.empty else "0")
    c3.metric("Coverage areas", "6", "API, schema, data, loader, train, eval")

    coverage_df = pd.DataFrame([
        ("FastAPI endpoints", "test_api.py", "6", "Checks /, /health, /model-info, /predict success, invalid payload 422, and model failure 500."),
        ("API schemas", "test_schemas.py", "10", "Validates request bounds, alias handling for int/intersection_type, response schema, and confidence validation."),
        ("Data preprocessing", "test_make_dataset.py", "4", "Covers year fixing, 2010-2016 filtering, hour derivation, victim_age derivation, expected columns, and split output files."),
        ("Model loading", "test_model_loader.py", "2", "Checks missing-model failure and successful joblib load behavior."),
        ("Training workflow", "test_train_model.py", "1", "Verifies training reads the expected files, fits XGBoost, evaluates outputs, and saves the model artifact."),
        ("Evaluation workflow", "test_evaluate_model.py", "3", "Checks metric calls, report generation flow, expected paths, and handling of single-column target inputs."),
    ], columns=["Area", "Evidence file", "Tests", "What is covered"])
    st.dataframe(coverage_df, use_container_width=True, hide_index=True)
    st.caption("This section summarizes test scope from the repository test suite. It describes what the tests cover without inventing a live pytest run output inside the app.")

    with st.expander("📄 Test file inventory"):
        st.dataframe(test_inventory, use_container_width=True, hide_index=True)

    st.markdown("<hr class='fancy-divider'>", unsafe_allow_html=True)

    # Inference API
    st.markdown('<div class="section-title">Inference API (FastAPI)</div>', unsafe_allow_html=True)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Service status", "Validated", "local FastAPI")
    c2.metric("Endpoints tested", "4", "/, /health, /model-info, /predict")
    c3.metric("Request features", "24", "schema-aligned payload")
    c4.metric("Error paths", "Validated", "422 and 500 covered")

    api_validation_df = pd.DataFrame([
        ("GET", "/", "Verified", "Root metadata endpoint returns docs and endpoint references."),
        ("GET", "/health", "Verified", "Health check returns service status."),
        ("GET", "/model-info", "Verified", "Returns expected feature count and MODEL_COLUMNS."),
        ("POST", "/predict", "Verified", "Returns prediction, severity label, description, confidence, and class probabilities."),
        ("POST", "/predict invalid payload", "Verified", "Schema validation rejects bad input with HTTP 422."),
        ("POST", "/predict model failure", "Verified", "Internal model failure path returns HTTP 500."),
    ], columns=["Method", "Path", "Status", "Evidence"])
    st.dataframe(api_validation_df, use_container_width=True, hide_index=True)

    endpoint_df = pd.DataFrame([
        ("GET", "/", "Root endpoint", "Quick API overview and endpoint references."),
        ("GET", "/health", "Health check", "Returns service health status."),
        ("GET", "/model-info", "Model metadata", "Exposes expected feature columns and feature count."),
        ("POST", "/predict", "Prediction endpoint", "Accepts a 24-feature payload and returns multiclass severity output."),
    ], columns=["Method", "Path", "Name", "Purpose"])
    st.markdown("**Endpoint contract**")
    st.dataframe(endpoint_df, use_container_width=True, hide_index=True)

    sample_payload = {
        "mois": 5, "jour": 12, "hour": 14, "lum": 1, "int": 1, "atm": 1,
        "col": 3, "catr": 4, "circ": 2, "nbv": 2, "vosp": 0, "surf": 1,
        "infra": 0, "situ": 1, "lat": 48.8566, "long": 2.3522, "place": 1,
        "catu": 1, "sexe": 1, "locp": 0, "actp": 0, "etatp": 1, "catv": 7,
        "victim_age": 35,
    }
    sample_response = {
        "prediction": 2,
        "severity": "Serious injury",
        "description": "Predicted as an accident with serious injuries.",
        "confidence": 0.82,
        "probabilities": {
            "no_injury_minor": 0.05,
            "slight_injury": 0.10,
            "serious_injury": 0.82,
            "fatal": 0.03,
        },
    }

    req_col, res_col = st.columns(2)
    with req_col:
        st.markdown("**Sample request body**")
        st.code(json.dumps(sample_payload, indent=2), language="json")
    with res_col:
        st.markdown("**Sample response body**")
        st.code(json.dumps(sample_response, indent=2), language="json")

    with st.expander("📄 FastAPI source code"):
        api_path = PROJECT_DIR / "src" / "api" / "main.py"
        st.code(api_path.read_text(encoding="utf-8") if api_path.exists() else "src/api/main.py not found", language="python")

    with st.expander("🧪 Live prediction demo"):
        st.caption("Use this only if you want to demonstrate a live API call during the presentation. Enter the running FastAPI service URL, for example http://localhost:8000, and the app will send the form payload to /predict.")
        api_url = st.text_input("Live API URL for demo calls", value="", placeholder="http://localhost:8000")

        with st.form("inference_form"):
            st.caption("Fields match src/api/schemas.py and src/api/main.py MODEL_COLUMNS.")
            fc1, fc2, fc3, fc4 = st.columns(4)
            mois = fc1.number_input("Month (mois)", 1, 12, 5)
            jour = fc2.number_input("Day (jour)", 1, 31, 12)
            hour = fc3.number_input("Hour", 0, 23, 14)
            lum = fc4.selectbox("Lighting (lum)", [0, 1, 2, 3, 4, 5], index=1)

            fc5, fc6, fc7, fc8 = st.columns(4)
            intersection_type = fc5.selectbox("Intersection (int)", [0, 1, 2, 3, 4, 5, 6, 7, 8, 9], index=1)
            atm = fc6.selectbox("Weather (atm)", [0, 1, 2, 3, 4, 5, 6, 7, 8, 9], index=1)
            col = fc7.selectbox("Collision (col)", [0, 1, 2, 3, 4, 5, 6, 7], index=3)
            catr = fc8.selectbox("Road category (catr)", [0, 1, 2, 3, 4, 5, 6, 9], index=4)

            fc9, fc10, fc11, fc12 = st.columns(4)
            circ = fc9.selectbox("Circulation (circ)", [0, 1, 2, 3, 4], index=2)
            nbv = fc10.number_input("Lanes (nbv)", 0, 20, 2)
            vosp = fc11.selectbox("Special lane (vosp)", [0, 1, 2, 3], index=0)
            surf = fc12.selectbox("Surface (surf)", [0, 1, 2, 3, 4, 5, 6, 7, 8, 9], index=1)

            fc13, fc14, fc15, fc16 = st.columns(4)
            infra = fc13.selectbox("Infrastructure (infra)", [0, 1, 2, 3, 4, 5, 6, 7, 8, 9], index=0)
            situ = fc14.selectbox("Situation (situ)", [0, 1, 2, 3, 4, 5, 6, 8], index=1)
            lat = fc15.number_input("Latitude (lat)", -90.0, 90.0, 48.8566, format="%.6f")
            long = fc16.number_input("Longitude (long)", -180.0, 180.0, 2.3522, format="%.6f")

            fc17, fc18, fc19, fc20 = st.columns(4)
            place = fc17.number_input("Seat/place", 0, 10, 1)
            catu = fc18.selectbox("User category (catu)", [0, 1, 2, 3, 4], index=1)
            sexe = fc19.selectbox("Sexe", [0, 1, 2], index=1)
            catv = fc20.number_input("Vehicle category (catv)", 0, 99, 7)

            fc21, fc22, fc23, fc24 = st.columns(4)
            locp = fc21.selectbox("Pedestrian location (locp)", list(range(0, 10)), index=0)
            actp = fc22.selectbox("Pedestrian action (actp)", list(range(0, 10)), index=0)
            etatp = fc23.selectbox("Pedestrian state (etatp)", [0, 1, 2, 3], index=1)
            victim_age = fc24.number_input("Victim age", 0, 120, 35)

            submitted = st.form_submit_button("🔮  Call /predict")

    if submitted:
        payload = {
            "mois": int(mois),
            "jour": int(jour),
            "hour": int(hour),
            "lum": int(lum),
            "int": int(intersection_type),
            "atm": int(atm),
            "col": int(col),
            "catr": int(catr),
            "circ": int(circ),
            "nbv": int(nbv),
            "vosp": int(vosp),
            "surf": int(surf),
            "infra": int(infra),
            "situ": int(situ),
            "lat": float(lat),
            "long": float(long),
            "place": int(place),
            "catu": int(catu),
            "sexe": int(sexe),
            "locp": int(locp),
            "actp": int(actp),
            "etatp": int(etatp),
            "catv": int(catv),
            "victim_age": int(victim_age),
        }
        labels_map = {1:"🟢 Uninjured",2:"🔴 Killed",3:"🟠 Hospitalised",4:"🟡 Light Injury"}

        if api_url.strip():
            resolved_endpoint = api_url.rstrip("/") + "/predict"
            st.caption(f"Live request target: POST {resolved_endpoint}")
            with st.spinner("Calling FastAPI /predict endpoint…"):
                result = try_api_prediction(api_url, payload)
            if result and "error" not in result:
                pred = int(result.get("prediction", 0))
                label = result.get("severity") or result.get("label") or labels_map.get(pred + 1, "Unknown")
                conf = result.get("confidence", None)
                conf_text = f" — confidence {float(conf):.1%}" if conf is not None else ""
                st.success(f"**API response:** {label}{conf_text}")
                st.json(result)
            else:
                st.error("Could not reach the API endpoint. No prediction is shown because synthetic fallback predictions are disabled for accuracy.")
                if result:
                    st.caption(result.get("error", "Unknown API error"))
        else:
            st.info("No API URL provided. The form payload is shown below, but no prediction is generated without the FastAPI service.")
            st.json(payload)

# ══════════════════════════════════════════════════════════════════════════════
# PAGE 4 — SERVICES & MLFLOW
# ══════════════════════════════════════════════════════════════════════════════
elif page == "🔗  Services & MLflow":
    st.markdown('<div class="page-title">Services & MLflow</div>', unsafe_allow_html=True)
    st.markdown('<div class="page-subtitle">Verified FastAPI, Evidently, MLflow, DVC pipeline and Docker service work</div>', unsafe_allow_html=True)
    st.warning("This page reports the services that were actually validated. Airflow, Prometheus and Grafana are not presented as working services in this review.")

    # Team split
    st.markdown('<div class="section-title">Validated Service Surface</div>', unsafe_allow_html=True)
    services = [
        ("🗃️", "Data preparation", "Verified", "make_dataset.py ran and produced cleaned/preprocessed training and evaluation outputs."),
        ("🤖", "Model training", "Verified", "XGBoost baseline training completed and saved the model artifact."),
        ("📈", "Evaluation", "Verified", "Evaluation outputs and metrics were generated for the trained classifier."),
        ("🚀", "FastAPI service", "Verified", "Local endpoints worked: /, /health, /model-info and /predict."),
        ("📊", "MLflow", "Verified with limits", "Docker MLflow server logged runs, parameters, metrics, artifacts and model registration using a file-based backend."),
        ("📡", "Evidently monitoring", "Verified", "Data drift and prediction drift monitoring worked on the monitoring branch."),
        ("🧪", "pytest", "Verified", "Base branch and monitoring branch tests passed with meaningful API and pipeline coverage."),
        ("🧱", "DVC pipeline", "Partially verified", "dvc repro worked in the DVC container; remote pull/fresh-clone reproducibility still needs cleanup."),
    ]
    cols = st.columns(4)
    for i, (icon, name, status, desc) in enumerate(services):
        with cols[i % 4]:
            st.markdown(f"""
            <div class='metric-card' style='min-height:130px;'>
              <div style='font-size:1.6rem;margin-bottom:6px;'>{icon}</div>
              <div style='font-family:Syne,sans-serif;font-weight:700;
                          color:#f1f5f9;font-size:0.95rem;'>{name}</div>
              <div style='font-family:DM Mono,monospace;font-size:0.72rem;
                          color:#f97316;margin:3px 0;'>{status}</div>
              <div style='font-size:0.78rem;color:#64748b;line-height:1.5;'>{desc}</div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("<hr class='fancy-divider'>", unsafe_allow_html=True)
    st.markdown('<div class="section-title">Validated Project Flow</div>', unsafe_allow_html=True)
    flow_df = pd.DataFrame([
        ("Raw data", "data/accidents_full.csv", "Loaded for EDA and DVC input."),
        ("Preprocessing", "src/data/make_dataset.py", "Creates preprocessed train/test CSVs."),
        ("Training", "src/models/train_model.py", "Trains XGBoost and saves model artifact."),
        ("Evaluation", "src/models/evaluate_model.py", "Writes metrics and plot artifacts."),
        ("Serving", "src/api/main.py", "FastAPI endpoints for health/model-info/predict."),
        ("Monitoring", "reports/xtrain_vs_xtest_drift_report.html", "Evidently drift report artifact."),
    ], columns=["Layer", "Project artifact", "Evidence"])
    st.dataframe(flow_df, use_container_width=True, hide_index=True)

    st.markdown("<hr class='fancy-divider'>", unsafe_allow_html=True)

    # MLflow runs
    st.markdown('<div class="section-title">MLflow Experiment Tracking</div>', unsafe_allow_html=True)
    st.info("MLflow was validated in Docker: training runs, parameters, metrics, artifacts and model registration were visible. The current setup uses a file-based backend, so it is appropriate for project tracking but not presented as a hardened production registry.")
    mlflow_evidence = pd.DataFrame([
        ("Tracking server", "Verified", "MLflow server ran in Docker."),
        ("Parameters", "Verified", "Training parameters were visible in MLflow."),
        ("Metrics", "Verified", "Evaluation metrics were logged and visible."),
        ("Artifacts", "Verified", "Model/evaluation artifacts were logged."),
        ("Model registration", "Verified with limits", "Registration worked with the file-based backend."),
    ], columns=["Capability", "Status", "Evidence"])
    st.dataframe(mlflow_evidence, use_container_width=True, hide_index=True)

    st.markdown("<hr class='fancy-divider'>", unsafe_allow_html=True)

    # DVC versioning
    st.markdown('<div class="section-title">Data Versioning — DVC Pipeline</div>', unsafe_allow_html=True)
    st.warning("DVC pipeline execution is partially verified: dvc repro worked inside the DVC container. Fresh-clone reproducibility and remote pull setup still need cleanup.")
    with st.expander("📄  dvc.yaml — Pipeline Definition"):
        dvc_path = PROJECT_DIR / "dvc.yaml"
        st.code(dvc_path.read_text(encoding="utf-8") if dvc_path.exists() else "dvc.yaml not found", language="yaml")

    st.markdown("**DVC stages from the actual dvc.yaml:**")
    dvc_stage_df = pd.DataFrame([
        ("make_dataset", "python src/data/make_dataset.py"),
        ("validate_data", "python src/data/validate_data.py"),
        ("train_model", "python src/models/train_model.py"),
        ("evaluate_model", "python src/models/evaluate_model.py"),
        ("track_experiment", "python src/track_experiment.py"),
    ], columns=["Stage", "Command"])
    st.dataframe(dvc_stage_df, use_container_width=True, hide_index=True)

    st.markdown("<hr class='fancy-divider'>", unsafe_allow_html=True)
    st.markdown('<div class="section-title">Orchestration Context — Not Fully Validated</div>', unsafe_allow_html=True)
    st.warning("Airflow was not fully validated in this local review, so no orchestration DAG graph is displayed.")

# ══════════════════════════════════════════════════════════════════════════════
# PAGE 5 — DOCKER & DEPLOYMENT
# ══════════════════════════════════════════════════════════════════════════════
elif page == "🚀  Docker & Deployment":
    st.markdown('<div class="page-title">Docker & Deployment</div>', unsafe_allow_html=True)
    st.markdown('<div class="page-subtitle">What was built, what ran in Compose, and what still needs deployment cleanup</div>', unsafe_allow_html=True)
    st.warning("Docker image build succeeded and the Compose services mlflow, dvc and api worked. nginx is present as a deployment layer but not claimed as fully stable or production-ready.")

    # CI Pipeline
    st.markdown('<div class="section-title">CI Context — Tests Verified Locally</div>', unsafe_allow_html=True)
    st.caption("The project includes pytest coverage and tests passed on the base and monitoring branches. Treat the workflow below as CI representation unless the latest remote run is checked separately.")
    workflow_files = sorted((PROJECT_DIR / ".github" / "workflows").glob("*.y*ml"))
    with st.expander("📄  Actual workflow files", expanded=True):
        if workflow_files:
            for workflow in workflow_files:
                st.markdown(f"**{workflow.relative_to(PROJECT_DIR)}**")
                st.code(workflow.read_text(encoding="utf-8"), language="yaml")
        else:
            st.warning("No workflow files found under .github/workflows.")

    st.markdown("<hr class='fancy-divider'>", unsafe_allow_html=True)

    # NGINX
    st.markdown('<div class="section-title">Reverse Proxy — nginx Needs Cleanup</div>', unsafe_allow_html=True)
    col_l, col_r = st.columns(2)
    with col_l:
        with st.expander("📄  nginx.conf"):
            nginx_path = PROJECT_DIR / "deployments" / "nginx" / "nginx.conf"
            st.code(nginx_path.read_text(encoding="utf-8") if nginx_path.exists() else "nginx.conf not found", language="nginx")
    with col_r:
        st.markdown("""
        <div class='metric-card'>
          <div class='section-label'>Configured / intended layers</div>
          <div style='margin-top:0.8rem;'>
            <span class='badge badge-blue'>HTTP reverse proxy</span>
            <span class='badge badge-green'>Rate Limiting</span>
            <span class='badge badge-red'>End-to-end validation pending</span>
            <span class='badge badge-blue'>Docker DNS resolver</span>
            <span class='badge badge-blue'>Forwarded headers</span>
          </div>
          <div class='section-label' style='margin-top:1rem;'>Cleanup</div>
          <div style='margin-top:0.5rem;'>
            <span class='badge badge-red'>HTTPS not finalized</span>
            <span class='badge badge-red'>Production hardening pending</span>
          </div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<hr class='fancy-divider'>", unsafe_allow_html=True)

    # Docker
    st.markdown('<div class="section-title">Containerisation — Docker</div>', unsafe_allow_html=True)
    st.info("Docker image build succeeded. A key technical finding is that the build was slow because the build context was too large; this is a cleanup opportunity, not a hidden failure.")
    tab1, tab2 = st.tabs(["🐳 Dockerfile", "🐙 docker-compose.yml"])

    with tab1:
        dockerfile_path = PROJECT_DIR / "Dockerfile"
        st.code(dockerfile_path.read_text(encoding="utf-8") if dockerfile_path.exists() else "Dockerfile not found", language="docker")

    with tab2:
        compose_path = PROJECT_DIR / "docker-compose.yaml"
        st.code(compose_path.read_text(encoding="utf-8") if compose_path.exists() else "docker-compose.yaml not found", language="yaml")

    st.markdown("<hr class='fancy-divider'>", unsafe_allow_html=True)
    st.markdown('<div class="section-title">Actual Compose Service Summary</div>', unsafe_allow_html=True)
    compose_services = pd.DataFrame([
        ("mlflow", "working in Docker", "Experiment tracking server"),
        ("dvc", "working in Docker", "Runs dvc repro"),
        ("api", "working in Docker", "FastAPI service on port 8000"),
        ("nginx", "needs cleanup", "Reverse proxy layer present, not fully validated"),
    ], columns=["Service", "Validated status", "Role"])
    st.dataframe(compose_services, use_container_width=True, hide_index=True)

# ══════════════════════════════════════════════════════════════════════════════
# PAGE 6 — MONITORING & STATUS
# ══════════════════════════════════════════════════════════════════════════════
elif page == "📡  Monitoring & Status":
    st.markdown('<div class="page-title">Monitoring & Project Status</div>', unsafe_allow_html=True)
    st.markdown('<div class="page-subtitle">Evidently drift monitoring was implemented and tested; infrastructure extensions are clearly separated</div>', unsafe_allow_html=True)
    st.info("Evidently is one of the strongest verified MLOps components in this project: data drift and prediction drift monitoring worked on the monitoring branch.")

    st.markdown('<div class="section-title">Overall Validation Status</div>', unsafe_allow_html=True)
    status_df = pd.DataFrame([
        ("Preprocessing / make_dataset.py", "Verified", "Pipeline ran and produced processed data outputs."),
        ("Training / XGBoost", "Verified", "Baseline model trained and artifact was saved."),
        ("Evaluation", "Verified", "Evaluation outputs and metrics were generated."),
        ("FastAPI endpoints", "Verified", "/, /health, /model-info and /predict were tested locally."),
        ("Evidently monitoring", "Verified", "Data drift and prediction drift worked."),
        ("pytest", "Verified", "Base and monitoring branch tests passed."),
        ("MLflow", "Verified with limits", "Runs, params, metrics, artifacts and registration worked with file-based backend."),
        ("Docker image build", "Verified", "Build succeeded, but build context was large and slow."),
        ("Docker Compose deployment", "Partially verified", "mlflow, dvc and api worked; nginx layer still needs cleanup."),
        ("DVC newcomer reproducibility", "Needs cleanup", "dvc pull/fresh clone remote setup was not smooth."),
        ("nginx reverse proxy", "Needs cleanup", "Deployment layer exists but was not fully stable end-to-end."),
        ("Airflow / Prometheus / Grafana", "Not validated", "Mentioned as intended/team infrastructure, not validated in this review."),
    ], columns=["Component", "Status", "Evidence / limitation"])

    st.dataframe(status_df, use_container_width=True, hide_index=True)

    st.markdown('<div class="section-title">Available Monitoring Artifact</div>', unsafe_allow_html=True)
    if EVIDENTLY_REPORT_PATH.exists():
        c1, c2 = st.columns(2)
        c1.metric("Evidently report", "present", EVIDENTLY_REPORT_PATH.name)
        c2.metric("Report size", f"{EVIDENTLY_REPORT_PATH.stat().st_size / 1024 / 1024:.1f} MB", "HTML artifact")
        st.markdown(f"Artifact path: `{EVIDENTLY_REPORT_PATH.relative_to(PROJECT_DIR)}`")
    else:
        st.warning("Evidently report artifact is missing.")

    with st.expander("📄  Prometheus/Grafana context — not validated in this review"):
        st.warning("Prometheus and Grafana were not fully validated locally, and no repository config artifact is displayed here.")

    st.markdown("<hr class='fancy-divider'>", unsafe_allow_html=True)

    # Evidently drift
    st.markdown('<div class="section-title">Verified Data & Prediction Drift — Evidently</div>', unsafe_allow_html=True)
    st.success("Validated scope: Evidently data drift and prediction drift monitoring were implemented and tested on the monitoring branch.")
    st.info("No synthetic drift scores are displayed. Use the Evidently HTML report artifact for detailed drift values.")

    st.markdown("<hr class='fancy-divider'>", unsafe_allow_html=True)

    st.markdown('<div class="section-title">Maintenance Automation</div>', unsafe_allow_html=True)
    st.warning("No validated automatic retraining artifact is present in this repository, so no retraining graph or script is displayed.")

    st.markdown("<hr class='fancy-divider'>", unsafe_allow_html=True)

    # Documentation
    st.markdown('<div class="section-title">📚 Technical Documentation</div>', unsafe_allow_html=True)
    with st.expander("README — Setup & Architecture", expanded=True):
        st.markdown("""
## 🚦 AccidentML — Road Accident Severity Prediction

**Goal:** Predict accident severity (`grav`) in France using historical national data (2005–2020).

---

### Current Status
Verified: preprocessing, training, evaluation, XGBoost artifact saving, FastAPI endpoints, pytest, MLflow tracking/registration, Docker image build, and Evidently data/prediction drift monitoring.

Partially verified / cleanup: Docker Compose deployment stability, DVC fresh-clone reproducibility, remote artifact pull setup, and nginx reverse-proxy validation.

Not validated in this review: Airflow, Prometheus and Grafana as fully running services.

### ⚙️ Quick Start
```bash
git clone https://github.com/Megha-2023/mar26bmlops_int_accidents
cd mar26bmlops_int_accidents
pip install -r requirements.txt
dvc repro            # verified in the DVC container; fresh-clone remote setup still needs cleanup
streamlit run accidentml_streamlit_project_fit_v2.py # launch this dashboard
```

---

### 🏗️ Architecture
```
Raw Data (data.gouv.fr)
    └─▶ Ingestion → Processing → Training → Evaluation
                                     └─▶ MLflow Tracking
                                     └─▶ Model Registry
                                              └─▶ Inference API (FastAPI)
                                                       └─▶ Evidently monitoring
                                                       └─▶ nginx cleanup / deployment layer
```

---

### 📡 API Reference
| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/predict` | Predict severity from features |
| GET  | `/health`  | Health check |
| GET  | `/model-info` | Model metadata and expected features |

**POST `/predict` body:**
```json
{"mois":5,"jour":12,"hour":14,"lum":1,"int":1,"atm":1,"col":3,"catr":4,"circ":2,"nbv":2,"vosp":0,"surf":1,"infra":0,"situ":1,"lat":48.8566,"long":2.3522,"place":1,"catu":1,"sexe":1,"locp":0,"actp":0,"etatp":1,"catv":7,"victim_age":35}
```

---

        """)

    with st.expander("🔧 Open Cleanup Items"):
        cleanup_df = pd.DataFrame([
            ("DVC", "Fresh-clone remote pull/reproducibility needs cleanup."),
            ("nginx", "Reverse-proxy layer needs stable end-to-end validation."),
            ("Docker", "Build context should be reduced to speed up image builds."),
            ("Airflow / Prometheus / Grafana", "Not validated in this review."),
        ], columns=["Area", "Item"])
        st.dataframe(cleanup_df, use_container_width=True, hide_index=True)
