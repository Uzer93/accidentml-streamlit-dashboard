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
import os
from pathlib import Path

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

# ─── DATA LOADERS / GENERATORS ───────────────────────────────────────────────
APP_DIR = Path(__file__).resolve().parent
PROJECT_DIR = Path(os.getenv("ACCIDENTML_REPO_DIR", APP_DIR)).expanduser().resolve()
ACCIDENTS_FULL_PATH = PROJECT_DIR / "data" / "accidents_full.csv"
PREPROCESSED_DIR = PROJECT_DIR / "data" / "preprocessed"
DASHBOARD_ARTIFACTS_DIR = PROJECT_DIR / "dashboard_artifacts"
METRICS_PATH = PROJECT_DIR / "metrics" / "metrics.json"
CLASSIFICATION_REPORT_PATH = PROJECT_DIR / "metrics" / "classification_report.txt"
CONFUSION_MATRIX_PATH = PROJECT_DIR / "metrics" / "plots" / "confusion_matrix.png"
ROC_CURVE_PATH = PROJECT_DIR / "metrics" / "plots" / "roc_curve.png"
EVIDENTLY_REPORT_PATH = PROJECT_DIR / "metrics" / "reports" / "xtrain_vs_xtest_drift_report.html"
PREDICTION_DRIFT_REPORT_PATH = PROJECT_DIR / "metrics" / "reports" / "xtrain_vs_xtest_prediction_drift_report.html"
CI_WORKFLOW_PATH = PROJECT_DIR / ".github" / "workflows" / "ci.yml"
AIRFLOW_DOCKERFILE_PATH = PROJECT_DIR / "Dockerfile.airflow"
AIRFLOW_REQUIREMENTS_PATH = PROJECT_DIR / "requirements.airflow.txt"
TESTS_DIR = PROJECT_DIR / "tests"
SRC_DIR = PROJECT_DIR / "src"
EXECUTION_FLOW_IMAGE_CANDIDATES = [
    PROJECT_DIR / "dashboard_artifacts" / "project_execution_flow.png",
    PROJECT_DIR / "metrics" / "reports" / "project_execution_flow.png",
    PROJECT_DIR / "reports" / "project_execution_flow.png",
    PROJECT_DIR / "project_execution_flow.png",
]
EXECUTION_FLOW_IMAGE_PATH = next((path for path in EXECUTION_FLOW_IMAGE_CANDIDATES if path.exists()), None)


def rel_path(path: Path) -> str:
    try:
        return str(path.relative_to(PROJECT_DIR))
    except ValueError:
        return str(path)

@st.cache_data(show_spinner="Loading BAAC merged dataset...")
def load_eda_data():
    """Load the merged accident dataset columns used by the EDA page."""
    if not ACCIDENTS_FULL_PATH.exists():
        return pd.DataFrame(), f"Missing DVC-managed dataset file: {rel_path(ACCIDENTS_FULL_PATH)}. Run dvc pull in the AccidentML repository if the data artifacts are not present."

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
        return pd.DataFrame(), pd.Series(dtype="float64"), f"Missing preprocessed files for split '{split}'. These DVC-managed artifacts may require dvc pull or a pipeline run."
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


def test_count_for(filename: str, fallback: int) -> int:
    path = TESTS_DIR / filename
    count = count_test_functions(path)
    return count if count else fallback

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
        "📊  Dataset & EDA",
        "⚙️  Data Processing",
        "🤖  Baseline Model",
        "🔗  Services & MLflow",
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
# PAGE 1 — DATASET & EDA
# ══════════════════════════════════════════════════════════════════════════════
if page == "📊  Dataset & EDA":
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
        effects make a naive random split inappropriate. The preprocessing pipeline filters the
        modeling data to <b>2010–2016</b> and uses a <b>time-based split</b>:
        train <b>2010–2015</b>, test <b>2016</b>.
        </div>
        """, unsafe_allow_html=True)

    st.markdown('<div class="section-title">Raw BAAC Sources and Merge Logic</div>', unsafe_allow_html=True)
    baac_sources_df = pd.DataFrame([
        ("caracteristiques", "accident context", "Time, weather, location and accident-level context."),
        ("lieux", "road / environment", "Road category, surface, geometry and infrastructure fields."),
        ("usagers", "people / target `grav`", "User-level records; the severity target `grav` is defined here."),
        ("vehicules", "vehicle info", "Vehicle category and vehicle-level accident participation."),
    ], columns=["Raw table", "Role", "Main contribution"])
    st.dataframe(baac_sources_df, use_container_width=True, hide_index=True)

    merge_logic_df = pd.DataFrame([
        ("caracteristiques", "`Num_Acc`", "accident-level base table"),
        ("lieux", "`Num_Acc`", "joined by accident identifier"),
        ("usagers", "`Num_Acc`", "joined by accident identifier; provides `grav`"),
        ("vehicules", "`Num_Acc`", "joined by accident identifier"),
        ("merged dataset", "`Num_Acc`", "combined accident/user/road/vehicle feature table"),
    ], columns=["Source", "Join key", "Merge role"])
    st.markdown("**Merge logic:** the four BAAC file families are combined through the shared accident identifier `Num_Acc`.")
    st.dataframe(merge_logic_df, use_container_width=True, hide_index=True)

    st.markdown("<hr class='fancy-divider'>", unsafe_allow_html=True)

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
        if not (use_artifacts and eda_hour.empty):
            hr = eda_hour.copy() if use_artifacts else df["hour"].value_counts().sort_index().reset_index()
            if not use_artifacts:
                hr.columns = ["Hour","Count"]
            fig3 = px.bar(hr, x="Hour", y="Count", title="Accident/User Rows by Hour of Day",
                          color="Count", color_continuous_scale="Oranges")
            apply_theme(fig3)
            st.plotly_chart(fig3, use_container_width=True)

    with col_r2:
        # Missing value heatmap
        if not (use_artifacts and eda_missing.empty):
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

# ══════════════════════════════════════════════════════════════════════════════
# PAGE 2 — DATA PROCESSING
# ══════════════════════════════════════════════════════════════════════════════
elif page == "⚙️  Data Processing":
    st.markdown('<div class="page-title">Data Processing & Preparation</div>', unsafe_allow_html=True)
    st.markdown('<div class="page-subtitle">Pipeline: Merge → Clean → Feature Engineer → Time Split</div>', unsafe_allow_html=True)
    st.info("The active Airflow/DVC data pipeline starts from the already merged `data/accidents_full.csv` file. It runs `make_dataset` to build the processed train/test-ready datasets, then `validate_data` to check the generated training data. Separate scripts exist for preparing current monitoring data, but raw BAAC table merging is not part of the Airflow DAG.")

    # Pipeline steps
    steps = [
        ("01", "Load & Merge", "Join caracteristiques, lieux, usagers, vehicules on Num_Acc"),
        ("02", "Time Filter",  "Keep 2010–2016 only — matches the modeling period"),
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
    st.markdown('<div class="section-title">Before / After Transformations</div>', unsafe_allow_html=True)
    transform_df = pd.DataFrame([
        (
            "Missing value handling",
            "Source fields can contain null or unavailable values after merge and filtering.",
            "Processed feature matrices are imputed with `SimpleImputer(strategy=\"most_frequent\")`.",
            "XGBoost receives complete train/test feature tables with consistent columns.",
        ),
        (
            "Hour extraction from `hrmn`",
            "`hrmn` stores time in HHMM-style numeric/text format.",
            "`hour` is derived with integer division by 100 after numeric conversion.",
            "The model gets a compact time-of-day feature without carrying the raw time string.",
        ),
        (
            "Victim age derivation",
            "Birth year is stored as `an_nais`; accident year is stored as `an`.",
            "`victim_age` is computed as `an - an_nais`, then filtered to valid ages.",
            "Age becomes an explicit model feature while invalid age records are removed.",
        ),
        (
            "Categorical encoding",
            "Categorical BAAC fields are kept as string-like codes before modeling.",
            "`OrdinalEncoder(handle_unknown=\"use_encoded_value\", unknown_value=-1)` encodes train/test categorical columns.",
            "The feature matrix becomes numeric and can handle unknown test categories consistently.",
        ),
    ], columns=["Transformation", "Before", "After", "Why it matters"])
    st.dataframe(transform_df, use_container_width=True, hide_index=True)

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
        hyperparams_df = pd.DataFrame([
            ("objective", "multi:softprob", "Multiclass probability output for severity classes."),
            ("n_estimators", "200", "Number of boosted trees in the current params file."),
            ("max_depth", "6", "Maximum tree depth in the current params file."),
            ("learning_rate", "0.1", "Boosting step size."),
            ("subsample", "0.8", "Row sampling ratio per boosting round."),
            ("colsample_bytree", "0.8", "Feature sampling ratio per tree."),
            ("eval_metric", "logloss", "Optimization/evaluation metric configured for training."),
            ("tree_method", "hist", "Histogram-based tree construction."),
            ("random_state", "42", "Reproducibility seed."),
        ], columns=["Parameter", "Value", "Role"])
        st.markdown("**XGBoost hyperparameter summary**")
        st.dataframe(hyperparams_df, use_container_width=True, hide_index=True)
        st.caption("Values reflect the current training parameter artifact and training code defaults where applicable.")

    with st.expander("📄 Training source code"):
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
    if not metrics:
        st.warning("metrics/metrics.json is missing. Evaluation metrics are generated by the ML or DVC pipeline.")
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
            st.warning("Confusion matrix artifact is missing. Metrics and plots are generated by the ML or DVC pipeline.")

    with col_r:
        if ROC_CURVE_PATH.exists():
            st.image(str(ROC_CURVE_PATH), caption="Artifact: metrics/plots/roc_curve.png")
        else:
            st.warning("ROC curve artifact is missing. Metrics and plots are generated by the ML or DVC pipeline.")

    with st.expander("📄 Classification report artifact"):
        if CLASSIFICATION_REPORT_PATH.exists():
            st.code(CLASSIFICATION_REPORT_PATH.read_text(encoding="utf-8"), language="text")
        else:
            st.warning("classification_report.txt is missing. Run the evaluation stage through the ML/DVC pipeline to generate it.")

    st.markdown("<hr class='fancy-divider'>", unsafe_allow_html=True)

    # Unit tests
    st.markdown('<div class="section-title">Repository Test Coverage Summary</div>', unsafe_allow_html=True)
    st.info("The current PR/push CI starts with Python 3.12 unit tests using `pytest -q`.")

    api_tests = count_test_functions(TESTS_DIR / "test_api.py")
    schema_tests = count_test_functions(TESTS_DIR / "test_schemas.py")
    train_tests = count_test_functions(TESTS_DIR / "test_train_model.py")
    detected_tests = api_tests + schema_tests + train_tests

    c1, c2, c3 = st.columns(3)
    c1.metric("Test modules", "3", "plus conftest.py")
    c2.metric("Detected test cases", str(detected_tests))
    c3.metric("Coverage areas", "3")

    st.markdown("<div class='section-label'>Covered domains</div>", unsafe_allow_html=True)
    st.markdown("""
    <div style='margin:0.5rem 0 1rem 0;'>
      <span class='badge badge-green'>API</span>
      <span class='badge badge-green'>schema</span>
      <span class='badge badge-green'>training</span>
    </div>
    """, unsafe_allow_html=True)

    coverage_df = pd.DataFrame([
        ("FastAPI endpoints", "test_api.py", str(api_tests), "Checks the FastAPI service behavior."),
        ("API schemas", "test_schemas.py", str(schema_tests), "Checks the request and response contracts."),
        ("Training workflow", "test_train_model.py", str(train_tests), "Checks the model training workflow."),
    ], columns=["Area", "Evidence file", "Tests", "Validation coverage"])
    st.dataframe(coverage_df, use_container_width=True, hide_index=True)

    st.markdown("""
    <div class='metric-card'>
      <div class='section-label'>Main idea</div>
      <div style='color:#cbd5e1;line-height:1.7;'>
        <ul style='margin:0.4rem 0 0 1.1rem;padding:0;'>
          <li>The tests do not prove that the model is perfect.</li>
          <li>They show that the main system components behave as expected.</li>
          <li>The API accepts valid requests and rejects invalid ones.</li>
          <li>The schema contract is checked, and the training workflow is exercised with test fixtures.</li>
          <li>This reduces the risk of silent failures before deployment.</li>
        </ul>
      </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<hr class='fancy-divider'>", unsafe_allow_html=True)

    # Inference API
    st.markdown('<div class="section-title">Inference API (FastAPI)</div>', unsafe_allow_html=True)
    st.info("FastAPI exposes the trained model through a clear request/response interface. In the current branch, the API loads the local model artifact through `src.model_registry.load_local_model()`, which defaults to `models/xgb_model.pkl`.")

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown("""
        <div class='metric-card'>
          <div class='metric-label'>API contract status</div>
          <div style='font-family:Syne,sans-serif;font-size:1.55rem;font-weight:700;color:#f97316;line-height:1.15;'>Validated</div>
          <div style='margin-top:0.45rem;'><span class='badge badge-green'>test-covered</span></div>
        </div>
        """, unsafe_allow_html=True)
    with c2:
        st.markdown("""
        <div class='metric-card'>
          <div class='metric-label'>Model source</div>
          <div style='font-family:Syne,sans-serif;font-size:1.55rem;font-weight:700;color:#f97316;line-height:1.15;'>Local artifact</div>
          <div style='margin-top:0.45rem;'><span class='badge badge-green'>xgb_model.pkl</span></div>
        </div>
        """, unsafe_allow_html=True)
    with c3:
        st.markdown("""
        <div class='metric-card'>
          <div class='metric-label'>Request features</div>
          <div style='font-family:Syne,sans-serif;font-size:1.55rem;font-weight:700;color:#f97316;line-height:1.15;'>24</div>
          <div style='margin-top:0.45rem;'><span class='badge badge-green'>schema-aligned</span></div>
        </div>
        """, unsafe_allow_html=True)
    with c4:
        st.markdown("""
        <div class='metric-card'>
          <div class='metric-label'>Failure handling</div>
          <div style='font-family:Syne,sans-serif;font-size:1.55rem;font-weight:700;color:#f97316;line-height:1.15;'>Covered</div>
          <div style='margin-top:0.45rem;'><span class='badge badge-green'>422 / 500</span></div>
        </div>
        """, unsafe_allow_html=True)

    api_validation_df = pd.DataFrame([
        ("GET", "/", "Root endpoint", "Provides a quick API overview and endpoint references."),
        ("GET", "/health", "Health check", "Confirms that the API service is running."),
        ("GET", "/model-info", "Model metadata", "Exposes expected feature count and feature metadata."),
        ("POST", "/predict", "Prediction endpoint", "Accepts a 24-feature accident payload and returns a multiclass severity prediction."),
        ("POST", "/predict invalid payload", "Validation failure", "Bad input is rejected with HTTP 422."),
        ("POST", "/predict model failure", "Internal failure", "Model failure is handled with HTTP 500."),
        ("GET", "/monitor/report/{report_type}", "Monitoring reports", "Serves generated data drift or prediction drift reports."),
    ], columns=["Method", "Path", "Purpose", "Presentation point"])
    st.dataframe(api_validation_df, use_container_width=True, hide_index=True)

    sample_payload = {
        "mois": 5, "jour": 12, "hour": 14, "lum": 1, "int": 1, "atm": 1,
        "col": 3, "catr": 4, "circ": 2, "nbv": 2, "vosp": 0, "surf": 1,
        "infra": 0, "situ": 1, "lat": 48.8566, "long": 2.3522, "place": 1,
        "catu": 1, "sexe": 1, "locp": 0, "actp": 0, "etatp": 1, "catv": 7,
        "victim_age": 35,
    }
    req_col, res_col = st.columns(2)
    with req_col:
        st.markdown("**Example `/predict` request: 24 accident-related features**")
        st.code(json.dumps(sample_payload, indent=2), language="json")
    with res_col:
        st.markdown("**Response schema: what the service returns**")
        response_schema_df = pd.DataFrame([
            ("prediction", "integer", "Predicted severity class code."),
            ("severity", "string", "Human-readable severity label."),
            ("description", "string", "Short explanation of the predicted class."),
            ("confidence", "float", "Confidence score for the predicted class."),
            ("probabilities", "object", "Probability distribution over all severity classes."),
        ], columns=["Field", "Type", "Meaning"])
        st.dataframe(response_schema_df, use_container_width=True, hide_index=True)

    st.markdown("""
    <div class='metric-card'>
      <div class='section-label'>Main idea</div>
      <div style='color:#cbd5e1;line-height:1.7;'>
        <ul style='margin:0.4rem 0 0 1.1rem;padding:0;'>
          <li>The API is the serving layer of the project.</li>
          <li>It transforms the local trained model artifact into a callable service.</li>
          <li>It gives other systems a consistent request/response interface.</li>
          <li>The request schema uses `intersection_type` internally with the external alias `int`.</li>
          <li>It keeps model inference controlled and structured.</li>
        </ul>
      </div>
    </div>
    """, unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# PAGE 4 — SERVICES & MLFLOW
# ══════════════════════════════════════════════════════════════════════════════
elif page == "🔗  Services & MLflow":
    st.markdown('<div class="page-title">Services & MLflow</div>', unsafe_allow_html=True)
    st.markdown('<div class="page-subtitle">Service architecture for the AccidentML MLOps platform</div>', unsafe_allow_html=True)
    st.info("Broader service architecture of the platform: each component has one clear responsibility.")

    st.markdown('<div class="section-title">Project Service Architecture</div>', unsafe_allow_html=True)
    st.markdown("""
    <div style='display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:0.75rem;align-items:stretch;'>
      <div class='metric-card' style='min-height:118px;'>
        <div class='section-label'>CI validation</div>
        <div style='font-family:Syne,sans-serif;font-weight:700;color:#f1f5f9;'>GitHub Actions CI</div>
        <div style='color:#64748b;font-size:0.82rem;margin-top:0.45rem;'>Light PR/push checks, not the project entry point</div>
      </div>
      <div class='metric-card' style='min-height:118px;'>
        <div class='section-label'>Launch</div>
        <div style='font-family:Syne,sans-serif;font-weight:700;color:#f1f5f9;'>docker compose up</div>
        <div style='color:#64748b;font-size:0.82rem;margin-top:0.45rem;'>Starts the service stack; Airflow runs through its profile</div>
      </div>
      <div class='metric-card' style='min-height:118px;'>
        <div class='section-label'>Orchestration</div>
        <div style='font-family:Syne,sans-serif;font-weight:700;color:#f1f5f9;'>Airflow</div>
        <div style='color:#64748b;font-size:0.82rem;margin-top:0.45rem;'>Runs make_dataset and validate_data</div>
      </div>
      <div class='metric-card' style='min-height:118px;'>
        <div class='section-label'>Serving</div>
        <div style='font-family:Syne,sans-serif;font-weight:700;color:#f1f5f9;'>FastAPI + nginx</div>
        <div style='color:#64748b;font-size:0.82rem;margin-top:0.45rem;'>Prediction API with proxy, HTTPS and rate-limit controls</div>
      </div>
    </div>
    <div style='display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:0.75rem;align-items:stretch;margin-top:0.75rem;'>
      <div class='metric-card' style='min-height:112px;'>
        <div class='section-label'>MLflow</div>
        <div style='color:#cbd5e1;font-size:0.86rem;'>Tracks runs, metrics, artifacts and model versions</div>
      </div>
      <div class='metric-card' style='min-height:112px;'>
        <div class='section-label'>DVC</div>
        <div style='color:#cbd5e1;font-size:0.86rem;'>Defines reproducible data and model pipeline stages</div>
      </div>
      <div class='metric-card' style='min-height:112px;'>
        <div class='section-label'>Prometheus</div>
        <div style='color:#cbd5e1;font-size:0.86rem;'>Future project item; not implemented in the current repository</div>
      </div>
      <div class='metric-card' style='min-height:112px;'>
        <div class='section-label'>Grafana</div>
        <div style='color:#cbd5e1;font-size:0.86rem;'>Future project item; not implemented in the current repository</div>
      </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<hr class='fancy-divider'>", unsafe_allow_html=True)

    st.markdown('<div class="section-title">Service Responsibilities</div>', unsafe_allow_html=True)
    execution_flow_df = pd.DataFrame([
        ("CI validation", "GitHub Actions CI", "Runs the automated checks for the project."),
        ("Runtime entry point", "Docker Compose", "Starts the project services together."),
        ("Orchestration", "Airflow", "Runs `make_dataset`, then `validate_data`."),
        ("Experiment management", "MLflow", "Stores training runs, metrics, artifacts, and model versions."),
        ("Pipeline reproducibility", "DVC", "Defines the data and model pipeline steps."),
        ("Inference", "FastAPI", "Serves model predictions through an API."),
        ("Serving gateway", "nginx", "Routes outside traffic to the API."),
        ("Future project", "Prometheus + Grafana", "Planned for future monitoring dashboards."),
    ], columns=["Layer", "Service", "Role"])
    st.dataframe(execution_flow_df, use_container_width=True, hide_index=True)

    st.markdown("<hr class='fancy-divider'>", unsafe_allow_html=True)
    st.markdown('<div class="section-title">MLflow Experiment Tracking & Registry</div>', unsafe_allow_html=True)
    st.info("MLflow gives traceability: it helps answer which model was trained, with which configuration, and what performance/artifacts were produced.")
    mlflow_evidence = pd.DataFrame([
        ("Experiment tracking", "Saves each training run."),
        ("Metrics logging", "Saves model performance scores."),
        ("Artifact logging", "Saves model files, plots, and reports."),
        ("Model registration", "Stores model versions."),
    ], columns=["Capability", "Role in the project"])
    st.dataframe(mlflow_evidence, use_container_width=True, hide_index=True)

    st.markdown("<hr class='fancy-divider'>", unsafe_allow_html=True)

    st.markdown('<div class="section-title">Data Versioning — DVC Pipeline</div>', unsafe_allow_html=True)
    st.info("DVC is the reproducibility layer: it defines the project pipeline stages instead of relying on manually executed scripts.")
    dvc_stage_df = pd.DataFrame([
        ("make_dataset", "Builds the processed data used for modeling."),
        ("validate_data", "Checks that the data is usable before training."),
        ("train_model", "Trains the XGBoost severity model."),
        ("evaluate_model", "Generates evaluation metrics and reports."),
        ("track_experiment", "Logs the run and artifacts to MLflow."),
    ], columns=["DVC stage", "Purpose"])
    st.dataframe(dvc_stage_df, use_container_width=True, hide_index=True)

# ══════════════════════════════════════════════════════════════════════════════
# PAGE 5 — ORCHESTRATION & DEPLOY
# ══════════════════════════════════════════════════════════════════════════════
elif page == "🚀  Orchestration & Deploy":
    st.markdown('<div class="page-title">Orchestration & Deploy</div>', unsafe_allow_html=True)
    st.markdown('<div class="page-subtitle">Execution flow from repository trigger to orchestrated services</div>', unsafe_allow_html=True)
    st.info("This page separates lightweight GitHub Actions PR/push CI from the local project runtime. CI validates pull requests and pushes; Docker Compose starts the project services, Airflow runs the data pipeline, the ML pipeline trains/tracks the model, and Evidently handles monitoring reports.")

    st.markdown('<div class="section-title">Project Execution Flow</div>', unsafe_allow_html=True)
    if EXECUTION_FLOW_IMAGE_PATH is not None:
        st.image(
            str(EXECUTION_FLOW_IMAGE_PATH),
            caption="Project execution flow: lightweight GitHub Actions validation, Docker Compose services, Airflow data orchestration, MLflow, nginx, Evidently monitoring, and future observability components.",
            use_container_width=True,
        )

    execution_steps_df = pd.DataFrame([
        ("1", "GitHub Actions PR/push CI", "Runs lightweight validation in `.github/workflows/ci.yml`; it is not the project runtime entry point."),
        ("2", "docker compose up", "Starts the main service stack; Airflow services run when the Airflow profile is enabled."),
        ("3", "Airflow orchestration", "Starts the Airflow DAG on port 8081 and runs `make_dataset` then `validate_data`."),
        ("4", "MLflow", "Runs on port 5000 for experiment tracking and model registry work."),
        ("5", "ML pipeline", "Runs model training and experiment tracking."),
        ("6", "nginx reverse proxy", "Routes API endpoint traffic with HTTPS/SSL and rate-limit controls."),
        ("7", "Monitoring reports", "Uses API/report-generation code to create Evidently data drift and prediction drift reports."),
        ("8", "Future project", "Prometheus and Grafana are not implemented yet."),
    ], columns=["Step", "Execution layer", "Role"])
    st.dataframe(execution_steps_df, use_container_width=True, hide_index=True)

    st.markdown("<hr class='fancy-divider'>", unsafe_allow_html=True)
    st.markdown('<div class="section-title">What CI Validates</div>', unsafe_allow_html=True)
    st.info("The current workflow runs on pull requests, manual `workflow_dispatch`, and pushes to `main`, `master`, `ci`, and `feature/airflow_grafana_ci`.")
    ci_validation_df = pd.DataFrame([
        ("1", "Unit tests", "Runs the 17 current unit tests."),
        ("2", "Docker Compose config", "Checks that the service setup is valid."),
        ("3", "API Docker image", "Checks that the API image can be built."),
        ("4", "Airflow Docker image", "Checks that the Airflow image can be built."),
        ("5", "Airflow DAG import/list", "Checks that Airflow can load the project DAG."),
    ], columns=["Order", "CI check", "Current behavior"])
    st.dataframe(ci_validation_df, use_container_width=True, hide_index=True)
    st.markdown("""
    <div class='metric-card'>
      <div class='section-label'>CI boundary</div>
      <div style='color:#cbd5e1;line-height:1.7;'>
        PR/push CI intentionally does not run <code>dvc pull</code>, <code>dvc repro</code>, full model training,
        full training, experiment tracking, or full Docker Compose stack execution. Those heavier steps belong to local Docker Compose,
        Airflow, ML pipeline, or manual execution.
      </div>
      <div style='margin-top:0.8rem;'>
        <span class='badge badge-green'>17 unit tests</span>
        <span class='badge badge-green'>API image build passed</span>
        <span class='badge badge-green'>Airflow image build passed</span>
        <span class='badge badge-green'>DAG list validation passed</span>
      </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<hr class='fancy-divider'>", unsafe_allow_html=True)
    st.markdown('<div class="section-title">Airflow Orchestration Design</div>', unsafe_allow_html=True)
    st.info("Airflow now represents the lightweight data orchestration layer shown in the project flow: it builds the dataset and then validates it. Full model training and experiment tracking sit in the separate ML pipeline rather than inside this Airflow DAG description.")
    airflow_design_df = pd.DataFrame([
        ("dags/accident_pipeline_dag.py", "accident_pipeline / make_dataset", "Builds the processed train and test datasets."),
        ("dags/accident_pipeline_dag.py", "accident_pipeline / validate_data", "Checks that the processed data is valid."),
    ], columns=["DAG file", "DAG id / Task", "Purpose"])
    st.dataframe(airflow_design_df, use_container_width=True, hide_index=True)

    st.markdown("<hr class='fancy-divider'>", unsafe_allow_html=True)
    st.markdown('<div class="section-title">Serving Layer — nginx Gateway</div>', unsafe_allow_html=True)
    st.markdown("""
    <div class='metric-card'>
      <div class='section-label'>nginx role</div>
      <div style='color:#cbd5e1;line-height:1.7;'>
        nginx is the front-door gateway for the serving layer. External requests go through nginx before reaching FastAPI,
        so the API is not exposed as the only public entry point. In addition to reverse proxy routing, the nginx layer
        represents the HTTPS/SSL security boundary and applies request rate limiting to protect the prediction service.
      </div>
      <div style='margin-top:0.8rem;'>
        <span class='badge badge-blue'>reverse proxy</span>
        <span class='badge badge-blue'>HTTPS / SSL</span>
        <span class='badge badge-blue'>rate limiting</span>
        <span class='badge badge-blue'>API routing</span>
        <span class='badge badge-green'>serving layer</span>
        <span class='badge badge-blue'>controlled traffic flow</span>
      </div>
    </div>
    """, unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# PAGE 6 — MONITORING & STATUS
# ══════════════════════════════════════════════════════════════════════════════
elif page == "📡  Monitoring & Maintenance":
    st.markdown('<div class="page-title">Monitoring & Maintenance</div>', unsafe_allow_html=True)
    st.markdown('<div class="page-subtitle">Drift reporting, operational metrics and model-maintenance workflow</div>', unsafe_allow_html=True)
    st.info("Monitoring closes the MLOps loop: after deployment, we need to know whether both the model behavior and the system behavior remain healthy.")

    st.markdown('<div class="section-title">Evidently Drift Monitoring</div>', unsafe_allow_html=True)
    if EVIDENTLY_REPORT_PATH.exists():
        c1, c2 = st.columns(2)
        c1.metric("Data drift report", "present", EVIDENTLY_REPORT_PATH.name)
        c2.metric("Report size", f"{EVIDENTLY_REPORT_PATH.stat().st_size / 1024 / 1024:.1f} MB", "HTML artifact")
        st.markdown(f"Artifact path: `{rel_path(EVIDENTLY_REPORT_PATH)}`")
    drift_evidence_df = pd.DataFrame([
        ("Data drift report", "`/monitor/report/data`", "Checks if input data has changed."),
        ("Prediction drift report", "`/monitor/report/prediction`", "Checks if model predictions have changed."),
        ("Monitoring report artifacts", "metrics/reports/*.html", "Stores the generated Evidently reports."),
    ], columns=["Monitoring area", "Role", "Presentation point"])
    st.dataframe(drift_evidence_df, use_container_width=True, hide_index=True)

    st.markdown("<hr class='fancy-divider'>", unsafe_allow_html=True)
    st.markdown('<div class="section-title">Future Project: Observability Stack</div>', unsafe_allow_html=True)
    st.info("Prometheus and Grafana are not implemented in the current repository. They are kept here only as future project extensions beyond the Evidently API/report-generation code.")
    obs_col1, obs_col2 = st.columns(2)
    with obs_col1:
        st.markdown("""
        <div class='metric-card'>
          <div class='section-label'>Prometheus</div>
          <div style='font-family:Syne,sans-serif;font-weight:700;color:#f1f5f9;'>Future metrics collection</div>
          <div style='color:#64748b;font-size:0.86rem;margin-top:0.5rem;'>Future extension for service, infrastructure and runtime metrics.</div>
          <div style='margin-top:0.8rem;'><span class='badge badge-blue'>future work</span></div>
        </div>
        """, unsafe_allow_html=True)
    with obs_col2:
        st.markdown("""
        <div class='metric-card'>
          <div class='section-label'>Grafana</div>
          <div style='font-family:Syne,sans-serif;font-weight:700;color:#f1f5f9;'>Future dashboard visualization</div>
          <div style='color:#64748b;font-size:0.86rem;margin-top:0.5rem;'>Future extension for service health, latency, uptime and follow-up dashboards.</div>
          <div style='margin-top:0.8rem;'><span class='badge badge-blue'>future work</span></div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<hr class='fancy-divider'>", unsafe_allow_html=True)
    st.markdown('<div class="section-title">Closing Summary</div>', unsafe_allow_html=True)
    st.markdown("""
    <div class='metric-card'>
      <div class='section-label'>End-to-end MLOps platform</div>
      <div style='color:#cbd5e1;line-height:1.8;'>
        AccidentML is presented as more than a trained model. The system is tested with CI unit tests, served with FastAPI,
        tracked and managed with MLflow, structured with DVC, orchestrated with Airflow, launched with Docker Compose,
        routed with nginx, and monitored with Evidently reports. Prometheus and Grafana remain future project extensions.
      </div>
    </div>
    """, unsafe_allow_html=True)
