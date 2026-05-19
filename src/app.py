"""Application Streamlit — Mutual Funds Return Predictor."""

from __future__ import annotations

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from config import MODEL_METRICS_FILE, MODELS_DIR, MUTUAL_FUNDS_CSV

# ── Palette ───────────────────────────────────────────────────────────────
PRIMARY = "#1B3A6B"
ACCENT  = "#2563EB"
SUCCESS = "#059669"
DANGER  = "#DC2626"
WARNING = "#D97706"
GRAY    = "#64748B"

MODEL_PALETTE = {
    "Ridge Regression":       "#4F86C6",
    "Random Forest":          "#2E9E6B",
    "Hist Gradient Boosting": "#E07B39",
}

FEATURE_LABELS = {
    "fund_return_1year":                    "Rendement 1 an",
    "fund_return_3years":                   "Rendement 3 ans",
    "fund_yield":                           "Dividende",
    "fund_sector_technology":               "Technologie",
    "fund_sector_financial_services":       "Serv. Financiers",
    "fund_sector_healthcare":               "Sante",
    "fund_sector_energy":                   "Energie",
    "fund_sector_industrials":              "Industrie",
    "fund_annual_report_net_expense_ratio": "Frais annuels",
}


def _L(**kwargs) -> dict:
    """Construit un dict de layout Plotly — margin et tickfont jamais dupliqués."""
    _BLACK = "#000000"
    base = dict(
        font=dict(family="Inter, Arial, sans-serif", size=13, color=_BLACK),
        paper_bgcolor="white",
        plot_bgcolor="white",
        hoverlabel=dict(bgcolor="white", font=dict(size=13, color=_BLACK)),
        margin=kwargs.pop("margin", dict(t=50, b=40, l=50, r=20)),
    )
    # Axes: merge caller kwargs while forcing tickfont/title_font black
    for _ax in ("xaxis", "yaxis"):
        _defaults = dict(tickfont=dict(color=_BLACK), title_font=dict(color=_BLACK))
        if _ax in kwargs:
            _defaults.update(kwargs.pop(_ax))
        base[_ax] = _defaults
    # Title: force font color black while preserving text/size the caller provides
    _title = dict(font=dict(color=_BLACK))
    if "title" in kwargs:
        t = kwargs.pop("title")
        if isinstance(t, str):
            _title["text"] = t
        else:
            t.setdefault("font", {})
            t["font"].setdefault("color", _BLACK)
            _title.update(t)
    base["title"] = _title
    base.update(kwargs)
    return base


def _qcut_safe(series: pd.Series, q: int, labels: list) -> pd.Series:
    """pd.qcut that never raises on duplicate bin edges (ranks ties away)."""
    try:
        return pd.qcut(series, q, labels=labels)
    except ValueError:
        return pd.qcut(series.rank(method="first"), q, labels=labels)


def _hex_rgba(hex_color: str, alpha: float = 0.12) -> str:
    h = hex_color.lstrip("#")
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    return f"rgba({r},{g},{b},{alpha})"


# ── CSS ───────────────────────────────────────────────────────────────────
_CSS = """
<style>
  @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
  html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

  .app-header { padding: 28px 0 18px 0; border-bottom: 2px solid #E2E8F0; margin-bottom: 24px; }
  .app-header h1 { font-size: 26px; font-weight: 700; color: #1B3A6B; margin: 0; letter-spacing: -0.3px; }
  .app-header p  { font-size: 14px; color: #64748B; margin: 6px 0 0 0; }

  .kpi-card { background:white; border:1px solid #E2E8F0; border-left:4px solid #2563EB;
              border-radius:8px; padding:16px 20px; }
  .kpi-label { font-size:11px; font-weight:600; color:#64748B; text-transform:uppercase;
               letter-spacing:0.6px; margin:0; }
  .kpi-value { font-size:28px; font-weight:700; color:#1B3A6B; margin:4px 0 0 0; line-height:1.1; }
  .kpi-sub   { font-size:12px; color:#94A3B8; margin:3px 0 0 0; }
  .kpi-card.green  { border-left-color: #059669; }
  .kpi-card.red    { border-left-color: #DC2626; }
  .kpi-card.orange { border-left-color: #D97706; }

  .section-title { font-size:15px; font-weight:600; color:#1B3A6B; margin:28px 0 12px 0;
                   padding-bottom:6px; border-bottom:1px solid #E2E8F0; }

  .model-card  { background:white; border:1px solid #E2E8F0; border-radius:10px;
                 padding:20px; text-align:center; }
  .model-name  { font-size:12px; font-weight:600; color:#64748B; text-transform:uppercase;
                 letter-spacing:0.5px; margin:0; }
  .model-pred  { font-size:40px; font-weight:700; margin:8px 0 4px 0; line-height:1; }
  .model-badge { display:inline-block; font-size:11px; font-weight:600;
                 padding:3px 10px; border-radius:20px; }

  .info-banner { background:#EFF6FF; border:1px solid #BFDBFE; border-radius:8px;
                 padding:12px 16px; font-size:13px; color:#1E40AF; margin-bottom:16px; }

  #MainMenu, footer, header { visibility: hidden; }
  .stTabs [data-baseweb="tab-list"] { gap:4px; border-bottom:2px solid #E2E8F0; }
  .stTabs [data-baseweb="tab"] { font-size:13px; font-weight:500; padding:10px 20px;
                                  border-radius:6px 6px 0 0; color:#64748B; }
  .stTabs [aria-selected="true"] { color:#2563EB !important; font-weight:600; }
</style>
"""


def _kpi(label, value, sub="", variant=""):
    return (f'<div class="kpi-card {variant}"><p class="kpi-label">{label}</p>'
            f'<p class="kpi-value">{value}</p><p class="kpi-sub">{sub}</p></div>')


def _section(title):
    st.markdown(f'<p class="section-title">{title}</p>', unsafe_allow_html=True)


# ── Cache ─────────────────────────────────────────────────────────────────

@st.cache_data
def _load_raw_data():
    from data import _compute_target
    df = pd.read_csv(MUTUAL_FUNDS_CSV, low_memory=False)
    df["R_3_to_5"] = _compute_target(df)
    df = df.dropna(subset=["R_3_to_5", "fund_return_3years", "fund_return_5years"])
    return df[~np.isinf(df["R_3_to_5"])]


@st.cache_data
def _load_metrics():
    return pd.read_csv(MODEL_METRICS_FILE) if MODEL_METRICS_FILE.exists() else None


@st.cache_resource
def _load_rf_pipeline():
    path = MODELS_DIR / "random_forest.joblib"
    return joblib.load(path) if path.exists() else None


@st.cache_resource
def _load_all_pipelines():
    keys = {"ridge": "Ridge Regression", "random_forest": "Random Forest",
            "gradient_boosting": "Hist Gradient Boosting"}
    return {name: joblib.load(MODELS_DIR / f"{key}.joblib")
            for key, name in keys.items() if (MODELS_DIR / f"{key}.joblib").exists()}


# ── Tab 1 — Overview ──────────────────────────────────────────────────────

def _tab_overview(df):
    from data import FEATURES
    target = df["R_3_to_5"]
    feature_cols = [f for f in FEATURES if f in df.columns]
    pos_rate = float((target > 0).mean())

    # KPI cards
    cols = st.columns(4, gap="small")
    cards = [
        _kpi("Fonds analysés",   f"{len(df):,}",           "apres nettoyage des NaN"),
        _kpi("Rendement moyen",  f"{target.mean():.2f} %", f"mediane : {target.median():.2f} %"),
        _kpi("Volatilite (std)", f"{target.std():.2f} %",
             f"P5 / P95 : {target.quantile(.05):.1f} / {target.quantile(.95):.1f} %", "orange"),
        _kpi("Taux haussier",    f"{pos_rate:.1%}",        "fonds avec R3 > 0",
             "green" if pos_rate > 0.6 else "red"),
    ]
    for col, html in zip(cols, cards):
        col.markdown(html, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Distribution + Boxplot
    _section("Distribution du rendement annualise R3 vers R5")
    c1, c2 = st.columns([2, 1], gap="medium")
    with c1:
        fig = go.Figure()
        fig.add_trace(go.Histogram(x=target, nbinsx=90, marker_color=ACCENT, opacity=0.8,
                                   hovertemplate="Rendement : %{x:.1f}%<br>Fonds : %{y}<extra></extra>",
                                   name="Distribution"))
        fig.add_vline(x=float(target.mean()),   line_dash="dot",  line_color=DANGER,
                      annotation_text=f"Moy. {target.mean():.1f}%",  annotation_font_color="#000000")
        fig.add_vline(x=float(target.median()), line_dash="dash", line_color=WARNING,
                      annotation_text=f"Med. {target.median():.1f}%", annotation_font_color="#000000")
        fig.update_layout(**_L(xaxis_title="R3->5 (%)", yaxis_title="Nombre de fonds",
                               showlegend=False, height=320))
        st.plotly_chart(fig, use_container_width=True)

    with c2:
        fig2 = go.Figure()
        fig2.add_trace(go.Box(y=target, boxpoints="outliers", marker_color=ACCENT,
                              line_color=PRIMARY, fillcolor="rgba(37,99,235,0.12)",
                              hovertemplate="R3->5 : %{y:.2f}%<extra></extra>", name=""))
        fig2.update_layout(**_L(yaxis_title="R3->5 (%)", xaxis_showticklabels=False,
                                showlegend=False, height=320))
        st.plotly_chart(fig2, use_container_width=True)

    # Corrélations
    _section("Correlation des variables avec la target")
    corr = df[feature_cols + ["R_3_to_5"]].corr()["R_3_to_5"].drop("R_3_to_5").sort_values()
    labels_bar = [FEATURE_LABELS.get(f, f) for f in corr.index]
    colors_bar = [SUCCESS if v >= 0 else DANGER for v in corr.values]
    fig3 = go.Figure(go.Bar(x=corr.values, y=labels_bar, orientation="h",
                            marker_color=colors_bar,
                            text=[f"{v:+.3f}" for v in corr.values], textposition="outside",
                            hovertemplate="%{y} : %{x:.3f}<extra></extra>"))
    fig3.add_vline(x=0, line_color="#CBD5E1", line_width=1)
    fig3.update_layout(**_L(xaxis_title="Correlation de Pearson", xaxis=dict(range=[-0.6, 0.9]),
                            height=320, showlegend=False))
    st.plotly_chart(fig3, use_container_width=True)

    # Scatter top 2 features
    _section("Relation entre les variables cles et R3->5")
    top2 = corr.abs().sort_values(ascending=False).head(2).index.tolist()
    c3, c4 = st.columns(2, gap="medium")
    for col_st, feat in zip([c3, c4], top2):
        with col_st:
            sample = df[[feat, "R_3_to_5"]].dropna().sample(min(2500, len(df)), random_state=42)
            fig_sc = px.scatter(sample, x=feat, y="R_3_to_5", trendline="ols", opacity=0.35,
                                labels={feat: FEATURE_LABELS.get(feat, feat), "R_3_to_5": "R3->5 (%)"},
                                color_discrete_sequence=[ACCENT], template="plotly_white",
                                title=f"{FEATURE_LABELS.get(feat, feat)} vs R3->5")
            fig_sc.update_traces(selector=dict(mode="markers"), marker_size=4)
            fig_sc.update_layout(**_L(height=300))
            st.plotly_chart(fig_sc, use_container_width=True)


# ── Tab 2 — Models ────────────────────────────────────────────────────────

def _tab_models(df):
    from data import FEATURES
    from sklearn.model_selection import train_test_split

    metrics_df = _load_metrics()
    if metrics_df is None:
        st.warning("Aucun resultat. Lancez `python scripts/main.py`.")
        return

    available = [m for m in ["MAE", "RMSE", "R2", "Win Rate"] if m in metrics_df.columns]

    # Radar
    _section("Profil de performance — vue radar")
    radar_df = metrics_df[["model_name"] + available].copy()
    for m in available:
        col = radar_df[m]
        if m in ("MAE", "RMSE"):
            radar_df[m] = 1 - (col - col.min()) / (col.max() - col.min() + 1e-9)
        else:
            radar_df[m] = (col - col.min()) / (col.max() - col.min() + 1e-9)

    labels_map = {"MAE": "Precision MAE", "RMSE": "Precision RMSE", "R2": "R2", "Win Rate": "Win Rate"}
    cats = [labels_map.get(m, m) for m in available]
    fig_radar = go.Figure()
    for _, row in radar_df.iterrows():
        name   = row["model_name"]
        vals   = [float(row[m]) for m in available]
        vals  += [vals[0]]
        color  = MODEL_PALETTE.get(name, ACCENT)
        fig_radar.add_trace(go.Scatterpolar(
            r=vals, theta=cats + [cats[0]], fill="toself",
            fillcolor=_hex_rgba(color), line_color=color, line_width=2, name=name,
            hovertemplate="%{theta} : %{r:.2f}<extra>" + name + "</extra>"))
    fig_radar.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0, 1],
                                   tickfont=dict(size=10, color="#000000"), gridcolor="#E2E8F0"),
                   angularaxis=dict(tickfont=dict(size=12, color="#000000")), bgcolor="white"),
        showlegend=True, legend=dict(orientation="h", y=-0.15, font=dict(color="#000000")),
        height=380, paper_bgcolor="white",
        font=dict(family="Inter, Arial, sans-serif", color="#000000"),
        margin=dict(t=30, b=60, l=40, r=40))
    st.plotly_chart(fig_radar, use_container_width=True)

    # Barres métriques
    _section("Metriques detaillees")
    metric_display = {"MAE": "MAE (%)", "RMSE": "RMSE (%)", "R2": "R2", "Win Rate": "Win Rate"}
    cols_m = st.columns(len(available), gap="small")
    for col_st, metric in zip(cols_m, available):
        with col_st:
            sub = metrics_df[["model_name", metric]].copy()
            best_idx = sub[metric].idxmin() if metric in ("MAE", "RMSE") else sub[metric].idxmax()
            bar_colors = [MODEL_PALETTE.get(n, ACCENT) for n in sub["model_name"]]
            fig_m = go.Figure(go.Bar(
                x=sub["model_name"].tolist(), y=sub[metric].tolist(),
                marker_color=bar_colors,
                text=sub[metric].round(3).astype(str).tolist(),
                textposition="outside",
                hovertemplate="%{x}<br>" + metric + " : %{y:.4f}<extra></extra>"))
            fig_m.add_annotation(
                x=sub.loc[best_idx, "model_name"],
                y=float(sub.loc[best_idx, metric]),
                text="meilleur", showarrow=True, arrowhead=2,
                arrowcolor=PRIMARY, font=dict(size=10, color="#000000"), ay=-30)
            fig_m.update_layout(**_L(
                title=dict(text=metric_display.get(metric, metric), font=dict(size=13, color="#000000")),
                height=260, showlegend=False, xaxis_tickangle=-20))
            st.plotly_chart(fig_m, use_container_width=True)

    # Predicted vs Actual
    _section("Predicted vs Actual — tous les modeles")
    feature_cols = [f for f in FEATURES if f in df.columns]
    X = df[feature_cols]; y = df["R_3_to_5"]
    _, X_test, _, y_test = train_test_split(X, y, test_size=0.20, random_state=42)
    pipelines = _load_all_pipelines()
    lim = float(max(abs(y_test.min()), abs(y_test.max()))) * 1.05

    fig_pva = go.Figure()
    fig_pva.add_trace(go.Scatter(x=[-lim, lim], y=[-lim, lim], mode="lines",
                                  line=dict(color="#CBD5E1", dash="dash", width=1.5),
                                  name="Ideal (y=x)", hoverinfo="skip"))
    for name, pipeline in pipelines.items():
        y_pred = pipeline.predict(X_test)
        color  = MODEL_PALETTE.get(name, ACCENT)
        fig_pva.add_trace(go.Scatter(
            x=y_test.values.tolist(), y=y_pred.tolist(), mode="markers",
            marker=dict(color=color, size=4, opacity=0.45), name=name,
            hovertemplate="Reel : %{x:.2f}%<br>Predit : %{y:.2f}%<extra>" + name + "</extra>"))
    fig_pva.update_layout(**_L(
        xaxis_title="Rendement reel R3->5 (%)", yaxis_title="Rendement predit R3->5 (%)",
        xaxis=dict(range=[-lim, lim], zeroline=True, zerolinecolor="#E2E8F0"),
        yaxis=dict(range=[-lim, lim], zeroline=True, zerolinecolor="#E2E8F0"),
        height=440, legend=dict(orientation="h", y=-0.18, font=dict(color="#000000"))))
    st.plotly_chart(fig_pva, use_container_width=True)

    # Résidus
    _section("Distribution des residus — Hist Gradient Boosting")
    if "Hist Gradient Boosting" in pipelines:
        residuals = y_test.values - pipelines["Hist Gradient Boosting"].predict(X_test)
        fig_res = go.Figure(go.Histogram(
            x=residuals.tolist(), nbinsx=70,
            marker_color=MODEL_PALETTE["Hist Gradient Boosting"], opacity=0.75,
            hovertemplate="Residu : %{x:.2f}%<br>Fonds : %{y}<extra></extra>"))
        fig_res.add_vline(x=0, line_dash="dot", line_color=PRIMARY)
        fig_res.update_layout(**_L(xaxis_title="Erreur de prediction (%)",
                                   yaxis_title="Fonds", showlegend=False, height=280))
        st.plotly_chart(fig_res, use_container_width=True)


# ── Tab 3 — Features ──────────────────────────────────────────────────────

def _tab_features(df):
    from data import FEATURES
    pipeline = _load_rf_pipeline()
    if pipeline is None:
        st.warning("Modele Random Forest introuvable. Lancez `scripts/train_models.py`.")
        return

    feature_cols  = [f for f in FEATURES if f in df.columns]
    importances   = pipeline.named_steps["model"].feature_importances_
    labels_imp    = [FEATURE_LABELS.get(f, f) for f in feature_cols]
    imp_df = pd.DataFrame({"feature": feature_cols, "label": labels_imp,
                            "importance": importances}).sort_values("importance")

    # Importance + Donut
    _section("Importance des variables — Random Forest (MDI)")
    c1, c2 = st.columns([3, 2], gap="large")
    with c1:
        q33 = float(imp_df["importance"].quantile(0.33))
        q66 = float(imp_df["importance"].quantile(0.66))
        bar_colors = [ACCENT if v >= q66 else (GRAY if v >= q33 else "#CBD5E1")
                      for v in imp_df["importance"]]
        fig_imp = go.Figure(go.Bar(
            x=imp_df["importance"].tolist(), y=imp_df["label"].tolist(), orientation="h",
            marker_color=bar_colors,
            text=[f"{v:.1%}" for v in imp_df["importance"]], textposition="outside",
            hovertemplate="%{y} : %{x:.4f}<extra></extra>"))
        fig_imp.update_layout(**_L(xaxis_title="Importance (MDI)", xaxis_tickformat=".0%",
                                   height=340, showlegend=False))
        st.plotly_chart(fig_imp, use_container_width=True)

    with c2:
        fig_donut = go.Figure(go.Pie(
            labels=imp_df["label"].tolist(), values=imp_df["importance"].tolist(),
            hole=0.55, textinfo="percent",
            hovertemplate="%{label}<br>%{percent}<extra></extra>",
            marker=dict(colors=px.colors.sequential.Blues_r[:len(imp_df)]),
            sort=True, direction="clockwise"))
        fig_donut.update_layout(
            paper_bgcolor="white", font=dict(family="Inter, Arial", size=11, color="#000000"),
            legend=dict(font=dict(size=11, color="#000000"), orientation="v"),
            height=340, margin=dict(t=20, b=20, l=0, r=0),
            annotations=[dict(text="Poids<br>relatif", x=0.5, y=0.5,
                               font=dict(size=13, color="#000000"), showarrow=False)])
        st.plotly_chart(fig_donut, use_container_width=True)

    # Allocation sectorielle par quartile
    _section("Allocation sectorielle moyenne par quartile de rendement")
    sector_cols = [c for c in ["fund_sector_technology", "fund_sector_financial_services",
                                "fund_sector_healthcare", "fund_sector_energy",
                                "fund_sector_industrials"] if c in df.columns]
    df_q = df[sector_cols + ["R_3_to_5"]].copy().dropna()
    df_q["quartile"] = _qcut_safe(df_q["R_3_to_5"], 4, ["Q1 bas", "Q2", "Q3", "Q4 haut"])
    grp = df_q.groupby("quartile", observed=True)[sector_cols].mean().reset_index()
    grp_long = grp.melt(id_vars="quartile", var_name="secteur", value_name="poids")
    grp_long["secteur"] = grp_long["secteur"].map(FEATURE_LABELS)
    fig_sector = px.bar(grp_long, x="secteur", y="poids", color="quartile", barmode="group",
                        color_discrete_sequence=["#BFDBFE", "#60A5FA", "#2563EB", "#1D4ED8"],
                        labels={"poids": "Poids moyen", "secteur": "Secteur",
                                "quartile": "Quartile"}, template="plotly_white")
    fig_sector.update_layout(**_L(yaxis_tickformat=".1%", height=340,
                                  legend=dict(title="Quartile R3->5", orientation="h", y=-0.22, font=dict(color="#000000")),
                                  xaxis_title=""))
    st.plotly_chart(fig_sector, use_container_width=True)

    # Heatmap
    _section("Profil moyen des features par quartile de rendement")
    df_heat = df[feature_cols + ["R_3_to_5"]].copy().dropna()
    df_heat["quartile"] = _qcut_safe(df_heat["R_3_to_5"], 4, ["Q1", "Q2", "Q3", "Q4"])
    pivot = df_heat.groupby("quartile", observed=True)[feature_cols].mean()
    pivot.columns = [FEATURE_LABELS.get(c, c) for c in pivot.columns]
    pivot_norm = (pivot - pivot.min()) / (pivot.max() - pivot.min() + 1e-9)
    text_grid  = [[f"{v:.3f}" for v in row] for row in pivot.values]
    fig_heat = go.Figure(go.Heatmap(
        z=pivot_norm.values.tolist(),
        x=pivot_norm.columns.tolist(),
        y=[str(i) for i in pivot_norm.index.tolist()],
        colorscale="Blues", text=text_grid, texttemplate="%{text}",
        textfont=dict(size=10), showscale=False,
        hovertemplate="Quartile : %{y}<br>Variable : %{x}<br>Score : %{z:.2f}<extra></extra>"))
    fig_heat.update_layout(**_L(height=240, xaxis_tickangle=-25,
                                margin=dict(t=20, b=60, l=60, r=10)))
    st.plotly_chart(fig_heat, use_container_width=True)


# ── Tab 4 — Simulator ────────────────────────────────────────────────────

def _tab_simulator(df):
    pipelines = _load_all_pipelines()
    if not pipelines:
        st.error("Aucun modele charge. Lancez `python scripts/train_models.py`.")
        return

    st.markdown('<div class="info-banner">Ajustez les parametres ci-dessous pour simuler '
                'les performances futures d\'un fonds a l\'issue de sa 3e annee.</div>',
                unsafe_allow_html=True)

    col_a, col_b = st.columns(2, gap="large")
    with col_a:
        _section("Performances historiques")
        r1y     = st.slider("Rendement 1 an (%)",             -60.0, 135.0, 25.0, 0.5)
        r3y     = st.slider("Rendement annualise 3 ans (%)",  -50.0,  65.0,  9.0, 0.5)
        yld     = st.slider("Dividende (%)",                    0.01,  12.0,  1.35, 0.05)
        expense = st.slider("Frais annuels (%)",                0.01,   5.0,  0.95, 0.01)

    with col_b:
        _section("Allocation sectorielle")
        tech   = st.slider("Technologie (%)",         0.0, 100.0, 17.0, 0.5)
        fin    = st.slider("Services financiers (%)", 0.0, 100.0, 15.0, 0.5)
        health = st.slider("Sante (%)",               0.0, 100.0, 12.0, 0.5)
        energy = st.slider("Energie (%)",             0.0, 100.0,  3.0, 0.5)
        indus  = st.slider("Industrie (%)",           0.0, 100.0, 11.0, 0.5)
        total  = tech + fin + health + energy + indus
        other  = max(0.0, 100.0 - total)

        if total > 100:
            st.error(f"Total sectoriel : {total:.0f}% — depasse 100%")
        else:
            pie_data = {"Technologie": tech, "Serv. Financiers": fin, "Sante": health,
                        "Energie": energy, "Industrie": indus, "Autres": other}
            fig_pie = go.Figure(go.Pie(
                labels=list(pie_data.keys()), values=list(pie_data.values()),
                hole=0.45, textinfo="percent+label", textfont=dict(size=10),
                marker=dict(colors=["#2563EB","#1D9E89","#F59E0B","#EF4444","#8B5CF6","#E2E8F0"]),
                hovertemplate="%{label} : %{percent}<extra></extra>", sort=False))
            fig_pie.update_layout(height=220, showlegend=False, paper_bgcolor="white",
                                  margin=dict(t=10, b=10, l=0, r=0),
                                  annotations=[dict(text="Portef.", x=0.5, y=0.5,
                                                     font=dict(size=11), showarrow=False)])
            st.plotly_chart(fig_pie, use_container_width=True)

    # Prédictions
    input_df = pd.DataFrame([{
        "fund_return_1year":                    r1y    / 100,
        "fund_return_3years":                   r3y    / 100,
        "fund_yield":                           yld    / 100,
        "fund_sector_technology":               tech   / 100,
        "fund_sector_financial_services":       fin    / 100,
        "fund_sector_healthcare":               health / 100,
        "fund_sector_energy":                   energy / 100,
        "fund_sector_industrials":              indus  / 100,
        "fund_annual_report_net_expense_ratio": expense / 100,
    }])

    predictions   = {n: float(p.predict(input_df)[0]) for n, p in pipelines.items()}
    consensus     = float(np.mean(list(predictions.values())))
    median_target = float(df["R_3_to_5"].median())
    pct_rank      = float((df["R_3_to_5"] < consensus).mean() * 100)

    st.markdown("<br>", unsafe_allow_html=True)
    _section("Prediction R3->5 — resultats par modele")

    pred_cols = st.columns(3, gap="small")
    for col_st, (name, pred) in zip(pred_cols, predictions.items()):
        color = MODEL_PALETTE.get(name, ACCENT)
        if pred >= 7:
            bb, bt, bl = "#D1FAE5", "#065F46", "Excellent"
        elif pred >= 4:
            bb, bt, bl = "#DBEAFE", "#1E40AF", "Bon"
        elif pred >= 0:
            bb, bt, bl = "#FEF3C7", "#92400E", "Neutre"
        else:
            bb, bt, bl = "#FEE2E2", "#991B1B", "Negatif"
        with col_st:
            st.markdown(
                f'<div class="model-card"><p class="model-name">{name}</p>'
                f'<p class="model-pred" style="color:{color};">{pred:+.2f}%</p>'
                f'<span class="model-badge" style="background:{bb};color:{bt};">{bl}</span></div>',
                unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Jauge + barre
    p5  = float(df["R_3_to_5"].quantile(0.05))
    p95 = float(df["R_3_to_5"].quantile(0.95))

    # Gauge bounds: always include 0 so that step ranges are never descending
    # (p5 can be positive when most funds have positive returns, making [p5, 0] invalid)
    gauge_min = min(p5, 0.0, consensus) - 1.0
    gauge_max = max(p95, consensus, median_target) + 1.0

    if median_target > 0.0:
        steps_gauge = [
            dict(range=[gauge_min, 0.0],          color="#FEE2E2"),
            dict(range=[0.0, median_target],       color="#FEF3C7"),
            dict(range=[median_target, gauge_max], color="#D1FAE5"),
        ]
    else:
        steps_gauge = [
            dict(range=[gauge_min, median_target], color="#FEE2E2"),
            dict(range=[median_target, gauge_max], color="#D1FAE5"),
        ]

    c1, c2 = st.columns([1, 1], gap="large")

    with c1:
        try:
            fig_gauge = go.Figure(go.Indicator(
                mode="gauge+number+delta", value=consensus,
                number=dict(suffix=" %", font=dict(size=40, color="#000000"), valueformat="+.2f"),
                delta=dict(reference=median_target, suffix="%", valueformat="+.2f",
                           increasing=dict(color=SUCCESS), decreasing=dict(color=DANGER)),
                gauge=dict(
                    axis=dict(range=[gauge_min, gauge_max], ticksuffix="%",
                              tickfont=dict(size=11), gridcolor="#E2E8F0"),
                    bar=dict(color=ACCENT, thickness=0.28), bgcolor="white",
                    bordercolor="#E2E8F0",
                    steps=steps_gauge,
                    threshold=dict(line=dict(color=GRAY, width=2), thickness=0.7,
                                   value=median_target)),
                title=dict(text=f"Consensus — {pct_rank:.0f}e percentile<br>"
                                 f"<span style='font-size:11px;color:#64748B'>"
                                 f"mediane dataset : {median_target:.2f}%</span>",
                           font=dict(size=14, color="#000000"))))
            fig_gauge.update_layout(height=300, paper_bgcolor="white",
                                    font=dict(family="Inter, Arial", color="#000000"),
                                    margin=dict(t=70, b=20, l=20, r=20))
            st.plotly_chart(fig_gauge, use_container_width=True)
        except Exception:
            st.metric("Consensus R3→5", f"{consensus:+.2f}%",
                      delta=f"{consensus - median_target:+.2f}% vs mediane")
            st.caption(f"Percentile : {pct_rank:.0f}e  |  Mediane dataset : {median_target:.2f}%")

    with c2:
        comp_names = list(predictions.keys()) + ["Mediane dataset"]
        comp_vals  = list(predictions.values()) + [median_target]
        n_pred     = len(predictions)
        bar_colors_c = [MODEL_PALETTE.get(n, ACCENT) if i < n_pred else "#CBD5E1"
                        for i, n in enumerate(comp_names)]
        fig_bar = go.Figure(go.Bar(
            x=comp_names, y=comp_vals, marker_color=bar_colors_c,
            text=[f"{v:+.2f}%" for v in comp_vals], textposition="outside",
            hovertemplate="%{x}<br>R3->5 predit : %{y:.2f}%<extra></extra>"))
        fig_bar.add_hline(y=0, line_color="#CBD5E1", line_width=1)
        fig_bar.update_layout(**_L(yaxis_title="R3->5 predit (%)",
                                   xaxis_tickangle=-15, showlegend=False, height=300))
        st.plotly_chart(fig_bar, use_container_width=True)

    # Positionnement dans le dataset
    try:
        _section("Positionnement dans le dataset")
        fig_pos = go.Figure()
        fig_pos.add_trace(go.Histogram(x=df["R_3_to_5"].dropna().tolist(), nbinsx=80,
                                       marker_color="#BFDBFE", opacity=0.8, hoverinfo="skip",
                                       name="Distribution"))
        fig_pos.add_vline(x=consensus, line_color=ACCENT, line_width=2.5,
                          annotation_text=f"Votre fonds : {consensus:+.2f}%",
                          annotation_font_color="#000000")
        fig_pos.add_vline(x=median_target, line_color=GRAY, line_dash="dot", line_width=1.5,
                          annotation_text=f"Mediane : {median_target:.2f}%",
                          annotation_font_color="#000000",
                          annotation_position="bottom right")
        fig_pos.update_layout(**_L(xaxis_title="R3->5 (%)", yaxis_title="Nombre de fonds",
                                   showlegend=False, height=270))
        st.plotly_chart(fig_pos, use_container_width=True)
    except Exception:
        pass


# ── Entry point ───────────────────────────────────────────────────────────

def build_app():
    st.set_page_config(page_title="Mutual Funds — Return Predictor",
                       layout="wide", initial_sidebar_state="collapsed")
    st.markdown(_CSS, unsafe_allow_html=True)
    st.markdown(
        '<div class="app-header">'
        '<h1>Mutual Funds — Prediction du rendement R3 vers R5</h1>'
        '<p>Modeles ML entraines a predire le rendement annualise entre l\'an 3 et l\'an 5 '
        'd\'un fonds, a partir des seules informations disponibles a la fin de l\'an 3. '
        '17 022 fonds analyses &nbsp;·&nbsp; 9 variables &nbsp;·&nbsp; 3 modeles compares.</p>'
        '</div>', unsafe_allow_html=True)

    df = _load_raw_data()
    tab1, tab2, tab3, tab4 = st.tabs(["Exploration", "Modeles", "Variables", "Simulation"])
    with tab1: _tab_overview(df)
    with tab2: _tab_models(df)
    with tab3: _tab_features(df)
    with tab4: _tab_simulator(df)


if __name__ == "__main__":
    build_app()
