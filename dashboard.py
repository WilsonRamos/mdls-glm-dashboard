"""
Dashboard interactivo — EDA MDLS-GLM Slope Stability
Streamlit + Plotly · 5 visualizaciones en orden del notebook

Ejecutar:
    cd Dashboard
    streamlit run dashboard.py
"""
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from scipy.signal import correlate
from pathlib import Path

# ──────────────────────────────────────────────────────────────────────────────
# CONFIGURACIÓN
# ──────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="MDLS-GLM · EDA Dashboard",
    page_icon="⛰️",
    layout="wide",
)

PARQUET_PATH = Path(__file__).parent / "mdls_glm_1min.parquet"

CMAP_MES = {8: "#1f77b4", 9: "#ff7f0e", 10: "#2ca02c",
            11: "#d62728", 12: "#9467bd", 1: "#8c564b"}
NOMBRE_MES = {8: "Ago", 9: "Sep", 10: "Oct", 11: "Nov", 12: "Dic", 1: "Ene"}

# ──────────────────────────────────────────────────────────────────────────────
# CARGA Y PREPARACIÓN DE DATOS
# ──────────────────────────────────────────────────────────────────────────────
@st.cache_data
def cargar_datos():
    df = pd.read_parquet(PARQUET_PATH)
    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index)
    df = df.sort_index()
    return df


@st.cache_data
def preparar_horario(_df):
    return _df.resample("1h").mean(numeric_only=True)


@st.cache_data
def calcular_spearman(_df):
    cols = [c for c in
            ['T1','T5','H1','H5','V1','R','P','Dleft','Dright','I','Ax_rms','A1_rms']
            if c in _df.columns]
    df_h = _df[cols].resample('1h').mean().dropna()
    return df_h.corr(method='spearman'), cols, df_h


@st.cache_data
def calcular_pca(_df):
    cols = [c for c in
            ['T1','T5','H1','H5','V1','R','P','Dleft','Dright','I','Ax_rms','A1_rms']
            if c in _df.columns]
    df_h = _df[cols].resample('1h').mean().dropna()
    scaler = StandardScaler()
    X = scaler.fit_transform(df_h)
    pca = PCA(n_components=min(8, len(cols)))
    X_pca = pca.fit_transform(X)
    return pca, X_pca, df_h, cols


@st.cache_data
def lag_corr_cached(_df, xc, yc, max_lag_h=168):
    df_h = _df[[xc, yc]].resample('1h').mean().dropna()
    common = df_h.dropna()
    xv = (common[xc] - common[xc].mean()) / common[xc].std()
    yv = (common[yc] - common[yc].mean()) / common[yc].std()
    n = len(xv)
    cc = correlate(yv.values, xv.values, mode='full') / n
    lags = np.arange(-n + 1, n)
    mask = np.abs(lags) <= max_lag_h
    return lags[mask], cc[mask]


# ──────────────────────────────────────────────────────────────────────────────
# SIDEBAR — SELECCIÓN DE VISUALIZACIÓN
# ──────────────────────────────────────────────────────────────────────────────
VISUALIZACIONES = {
    "Distribuciones univariadas": "hist",
    "Correlación de Spearman": "spearman",
    "Análisis de Componentes Principales (PCA)": "pca",
    "Series de tiempo causales": "series",
    "Cross-correlación con lag": "crosscorr",
}

st.sidebar.title("⛰️ MDLS-GLM · EDA")
st.sidebar.markdown("**Monitoreo multi-sensor de talud de loess**  \nGaolan Mountain, China · 6 meses")
st.sidebar.markdown("---")

seleccion_label = st.sidebar.selectbox(
    "Seleccionar visualización",
    list(VISUALIZACIONES.keys()),
)
viz_key = VISUALIZACIONES[seleccion_label]

st.sidebar.markdown("---")
st.sidebar.markdown(
    "📓 **Notebook:** Pasos 3–7  \n"
    "📄 Li et al. (2025) · *Data in Brief* 62  \n"
    "🏫 UNSA 2026-A · Ciencia de Datos"
)

# ──────────────────────────────────────────────────────────────────────────────
# CARGA
# ──────────────────────────────────────────────────────────────────────────────
with st.spinner("Cargando Parquet (meses estables Sep–Ene)…"):
    df = cargar_datos()

st.title(seleccion_label)

# ──────────────────────────────────────────────────────────────────────────────
# VIZ 1 — HISTOGRAMAS UNIVARIADOS (PASO 3)
# ──────────────────────────────────────────────────────────────────────────────
if viz_key == "hist":
    st.markdown(
        "**Distribuciones reales sobre los 226,446 minutos** (6 meses completos). "
        "Línea roja = media · línea negra punteada = mediana."
    )

    grupos = [
        ("T1 (°C)",    "T1",     "#E74C3C"),
        ("T5 (°C)",    "T5",     "#C0392B"),
        ("H1 (V)",     "H1",     "#3498DB"),
        ("V1 (mV)",    "V1",     "#9B59B6"),
        ("R (raw)",    "R",      "#27AE60"),
        ("P (kPa)",    "P",      "#F39C12"),
        ("Dleft (mm)", "Dleft",  "#16A085"),
        ("I (lux)",    "I",      "#F1C40F"),
        ("Ax_rms (g)", "Ax_rms", "#8E44AD"),
    ]
    grupos = [(l, c, col) for l, c, col in grupos if c in df.columns]

    fig = make_subplots(rows=3, cols=3, subplot_titles=[l for l, _, _ in grupos])

    # Ejes con clip p1-p99 para que outliers extremos no aplasten la distribución
    RANGE_OVERRIDE = {
        "I":     (0, 65535),   # mantener overflow ADC visible
        "Dleft": (2160, 2200), # solo rango operativo (excluye arranque ~23mm)
    }

    for idx, (label, col, color) in enumerate(grupos):
        row, col_pos = divmod(idx, 3)
        s = df[col].dropna()
        media = s.mean()
        mediana = s.median()

        # Clip p1-p99 para visualización (los datos originales no se modifican)
        lo = RANGE_OVERRIDE.get(col, (s.quantile(0.01), None))[0]
        hi = RANGE_OVERRIDE.get(col, (None, s.quantile(0.99)))[1]
        s_plot = s[(s >= lo) & (s <= hi)] if hi is not None else s[s >= lo]

        fig.add_trace(
            go.Histogram(
                x=s_plot,
                nbinsx=60,
                marker_color=color,
                opacity=0.85,
                name=label,
                showlegend=False,
            ),
            row=row + 1, col=col_pos + 1,
        )
        # Media (sobre datos completos)
        fig.add_vline(
            x=media, line_dash="dash", line_color="red", line_width=1.5,
            annotation_text=f"μ={media:.2f}", annotation_font_size=9,
            annotation_position="top right",
            row=row + 1, col=col_pos + 1,
        )
        # Mediana (sobre datos completos)
        fig.add_vline(
            x=mediana, line_dash="dot", line_color="black", line_width=1.5,
            annotation_text=f"med={mediana:.2f}", annotation_font_size=9,
            annotation_position="top left",
            row=row + 1, col=col_pos + 1,
        )

    fig.update_layout(
        height=820,
        title_text=f"Histogramas — Distribuciones sobre {len(df):,} minutos · rango p1–p99 (outliers excluidos del gráfico, no de los datos)",
        title_font_size=13,
        bargap=0.05,
    )
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("""
**Interpretaciones clave:**

| Sensor | Forma | Observación |
|--------|-------|-------------|
| **T1** | Asimétrica negativa | Pico en 17–18°C; cola hacia frío (enero). Media < Mediana |
| **T5** | Multimodal irregular | ~5 picos a lo largo de 0–25°C; std=6.9°C (3× mayor que T1) |
| **H1** | Estrecha / sesgada | Solo 10 mV de rango total; meses cálidos en el lado derecho |
| **V1** | Bimodal clara | Reposo ~20 mV · flujo activo ~25–40 mV |
| **R**  | Sesgo extremo derecho | >99% en baseline ~0.11 mm; picos = lluvia real |
| **P**  | Campana centrada en −2.9 kPa | Distribución real visible con clip p1-p99. Sin clip el eje 0–300 aplasta todo por un outlier de 325 kPa |
| **Dleft** | Spike en borde der. | Rango operativo ~2174 mm; barra tiny en 0 = arranque |
| **I**  | J invertida | Mediana=920 lux (noche); media=9,150 lux jalada por el sol |
| **Ax_rms** | Log-normal, sesgo derecho | Ruido base ~0.006 g; cola hasta 3 g = eventos vibratorios |
""")

# ──────────────────────────────────────────────────────────────────────────────
# VIZ 2 — CORRELACIÓN SPEARMAN (PASO 5)
# ──────────────────────────────────────────────────────────────────────────────
elif viz_key == "spearman":
    st.markdown(
        "**Correlación de Spearman** sobre datos resampleados a 1 hora (6 meses). "
        "Se destacan 3 pares con interpretación física relevante."
    )

    corr, corr_cols, df_h = calcular_spearman(df)
    n_horas = len(df_h)

    # Pares a destacar con rectángulo
    pares_highlight = [
        ("T1", "H1", "T1↔H1: deriva térmica (ρ≈+0.71)"),
        ("T1", "Dleft", "T1↔Dleft: correlación estacional (ρ≈+0.59)"),
        ("T1", "Ax_rms", "T1↔Ax_rms: anti-correlación estacional (ρ≈−0.70)"),
    ]

    z = corr.values
    labels = corr_cols

    fig = go.Figure(data=go.Heatmap(
        z=z,
        x=labels,
        y=labels,
        colorscale="RdBu_r",
        zmid=0,
        zmin=-1,
        zmax=1,
        text=np.round(z, 2),
        texttemplate="%{text}",
        textfont_size=9,
        hovertemplate="<b>%{y} ↔ %{x}</b><br>ρ = %{z:.3f}<extra></extra>",
        colorbar=dict(title="ρ Spearman"),
    ))

    # Rectángulos de highlight
    for (a, b, nota) in pares_highlight:
        if a in labels and b in labels:
            ix = labels.index(a)
            iy = labels.index(b)
            for rx, ry in [(ix, iy), (iy, ix)]:
                fig.add_shape(
                    type="rect",
                    x0=rx - 0.5, x1=rx + 0.5,
                    y0=ry - 0.5, y1=ry + 0.5,
                    line=dict(color="yellow", width=2.5),
                )

    fig.update_layout(
        title=f"Correlación de Spearman — 6 meses ({n_horas:,} horas)",
        height=620,
        xaxis=dict(tickfont_size=10),
        yaxis=dict(tickfont_size=10, autorange="reversed"),
    )
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("**Pares destacados (recuadro amarillo):**")
    for a, b, nota in pares_highlight:
        if a in labels and b in labels:
            rho = corr.loc[a, b]
            st.markdown(f"- **{nota}**: ρ = {rho:+.3f}")

    st.markdown("""
**Interpretaciones clave:**

1. **Eje térmico dominante:** T1 y T5 tienen correlaciones fuertes con muchos sensores
   porque el ciclo estacional (verano↔invierno) es el driver principal en 6 meses.
2. **Dleft ↔ Dright** (ρ > 0.95): multicolinealidad — ambos extensómetros miden el mismo punto.
3. **Sensores H en Voltios:** correlación H↔T = deriva térmica del circuito, no humedad real.
4. **T1 ↔ Ax_rms** (ρ negativo): en invierno hay más lluvia → más vibración; en verano menos.
   No significa que el calor reduzca el movimiento del talud directamente.
""")

# ──────────────────────────────────────────────────────────────────────────────
# VIZ 3 — PCA SCREE + BIPLOT INTERACTIVO (PASO 6)
# ──────────────────────────────────────────────────────────────────────────────
elif viz_key == "pca":
    st.markdown(
        "**PCA sobre 12 sensores** (resample 1 h, 6 meses). "
        "Scree plot a la izquierda · Biplot interactivo coloreado por mes a la derecha."
    )

    pca_obj, X_pca, df_pca, pca_cols = calcular_pca(df)
    var = pca_obj.explained_variance_ratio_
    cum = np.cumsum(var)
    n_pcs = len(var)

    # Filtro p1-p99 para biplot
    pc1, pc2 = X_pca[:, 0], X_pca[:, 1]
    p1_lo, p1_hi = np.percentile(pc1, [1, 99])
    p2_lo, p2_hi = np.percentile(pc2, [1, 99])
    mask = (pc1 >= p1_lo) & (pc1 <= p1_hi) & (pc2 >= p2_lo) & (pc2 <= p2_hi)

    pc1_f = pc1[mask]
    pc2_f = pc2[mask]
    meses_f = df_pca.index[mask].month
    colores_f = [CMAP_MES.get(m, "#888888") for m in meses_f]
    nombres_mes_f = [NOMBRE_MES.get(m, str(m)) for m in meses_f]

    fig = make_subplots(
        rows=1, cols=2,
        column_widths=[0.38, 0.62],
        subplot_titles=[
            "Scree plot — Varianza explicada",
            f"Biplot PC1 vs PC2 (color = mes · p1–p99 · {(~mask).sum()} outliers excluidos)"
        ],
    )

    # Scree — barras
    fig.add_trace(
        go.Bar(
            x=[f"PC{i+1}" for i in range(n_pcs)],
            y=var * 100,
            marker_color="#4878D0",
            name="Varianza individual",
            showlegend=True,
        ),
        row=1, col=1,
    )
    # Scree — acumulada
    fig.add_trace(
        go.Scatter(
            x=[f"PC{i+1}" for i in range(n_pcs)],
            y=cum * 100,
            mode="lines+markers",
            marker_color="red",
            name="Acumulada",
            showlegend=True,
        ),
        row=1, col=1,
    )
    fig.add_hline(y=80, line_dash="dot", line_color="gray",
                  annotation_text="80%", row=1, col=1)

    # Biplot — puntos por mes (trazas separadas para leyenda)
    for mes_num in sorted(set(meses_f)):
        idx_m = [i for i, m in enumerate(meses_f) if m == mes_num]
        fig.add_trace(
            go.Scatter(
                x=pc1_f[idx_m],
                y=pc2_f[idx_m],
                mode="markers",
                marker=dict(size=3, color=CMAP_MES.get(mes_num, "#888"), opacity=0.4),
                name=NOMBRE_MES.get(mes_num, str(mes_num)),
                showlegend=True,
            ),
            row=1, col=2,
        )

    # Flechas de loadings
    scale = min(pc1_f.max() - pc1_f.min(), pc2_f.max() - pc2_f.min()) * 0.35
    loadings = pca_obj.components_[:2].T
    for i, col in enumerate(pca_cols):
        lx = loadings[i, 0] * scale
        ly = loadings[i, 1] * scale
        fig.add_annotation(
            ax=0, ay=0,
            x=lx, y=ly,
            xref="x2", yref="y2",
            axref="x2", ayref="y2",
            showarrow=True,
            arrowhead=2,
            arrowcolor="red",
            arrowwidth=1.5,
        )
        fig.add_annotation(
            x=lx * 1.15, y=ly * 1.15,
            xref="x2", yref="y2",
            text=f"<b>{col}</b>",
            showarrow=False,
            font=dict(size=9, color="darkred"),
        )

    fig.update_layout(height=580, legend=dict(orientation="v", x=1.02))
    fig.update_xaxes(title_text="PC", row=1, col=1)
    fig.update_yaxes(title_text="% Varianza", row=1, col=1)
    fig.update_xaxes(title_text=f"PC1 ({var[0]*100:.1f}%)", row=1, col=2)
    fig.update_yaxes(title_text=f"PC2 ({var[1]*100:.1f}%)", row=1, col=2)

    st.plotly_chart(fig, use_container_width=True)

    # Tabla de loadings
    df_load = pd.DataFrame(
        pca_obj.components_[:3].T,
        index=pca_cols,
        columns=["PC1", "PC2", "PC3"]
    ).round(3)
    col1, col2 = st.columns([1, 1])
    with col1:
        st.markdown("**Loadings PC1–PC3:**")
        st.dataframe(df_load.style.background_gradient(cmap="RdBu_r", vmin=-1, vmax=1))
    with col2:
        st.markdown("**Varianza explicada acumulada:**")
        df_var = pd.DataFrame({
            "PC": [f"PC{i+1}" for i in range(n_pcs)],
            "Individual (%)": (var * 100).round(1),
            "Acumulada (%)": (cum * 100).round(1),
        })
        st.dataframe(df_var, hide_index=True)

    st.markdown(f"""
**Interpretación:**

- **PC1 ({var[0]*100:.1f}%):** captura el ciclo estacional térmico — sensores de temperatura y H
  (deriva térmica) cargan en el mismo sentido. Es el driver dominante en los 6 meses.
- **PC2 ({var[1]*100:.1f}%):** patrón estacional secundario: Dleft crece en verano
  (expansión térmica del suelo), mientras Ax_rms crece en invierno (lluvia genera vibraciones).
  Son opuestos en PC2 porque tienen estacionalidad inversa, **no porque sean fenómenos contrarios**.
- **Se necesitan 5 PCs para superar el 80%** → el sistema tiene al menos 5 dimensiones
  independientes de variabilidad. No existe una sola variable que explique todo el comportamiento.
""")

# ──────────────────────────────────────────────────────────────────────────────
# VIZ 4 — SERIES DE TIEMPO CAUSALES (PASO 7)
# ──────────────────────────────────────────────────────────────────────────────
elif viz_key == "series":
    st.markdown(
        "**Cadena causal T1 → R → P → Dleft** sobre 6 meses (resample 1 h). "
        "Doble eje Y para comparar escalas distintas. "
        "Selector de rango temporal en la parte inferior."
    )

    df_h = preparar_horario(df)

    cols_check = [c for c in ["T1", "R", "P", "Dleft"] if c in df_h.columns]
    if len(cols_check) < 2:
        st.error("Columnas no disponibles en el Parquet.")
        st.stop()

    # ylim p2-p98 por sensor
    def ylim(s):
        lo, hi = s.quantile(0.02), s.quantile(0.98)
        pad = (hi - lo) * 0.15 if hi > lo else 1
        return lo - pad, hi + pad

    fig = make_subplots(
        rows=4, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.04,
        subplot_titles=[
            "T1 — Temperatura superficial (°C)",
            "R — Lluvia (raw mm)",
            "P — Presión de poros (kPa)",
            "Dleft — Deformación izquierda (mm)",
        ],
        specs=[[{"secondary_y": False}]] * 4,
    )

    configs = [
        ("T1",    "#E74C3C", "T1 (°C)"),
        ("R",     "#27AE60", "R (mm)"),
        ("P",     "#F39C12", "P (kPa)"),
        ("Dleft", "#16A085", "Dleft (mm)"),
    ]

    for row_i, (col, color, ylabel) in enumerate(configs, start=1):
        if col not in df_h.columns:
            continue
        s = df_h[col].dropna()
        lo, hi = ylim(s)
        fig.add_trace(
            go.Scatter(
                x=s.index,
                y=s,
                mode="lines",
                line=dict(color=color, width=0.8),
                name=ylabel,
                hovertemplate=f"<b>{col}</b><br>%{{x}}<br>%{{y:.3f}}<extra></extra>",
            ),
            row=row_i, col=1,
        )
        fig.update_yaxes(title_text=ylabel, range=[lo, hi], row=row_i, col=1,
                         title_font_size=10)

    fig.update_layout(
        height=760,
        title_text="Cadena causal: T1 → R → P → Dleft (resample 1 h · ylim p2–p98)",
        showlegend=True,
        legend=dict(orientation="h", y=-0.05),
        xaxis4=dict(
            rangeslider=dict(visible=True, thickness=0.04),
            title="Fecha",
        ),
    )
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("""
**Interpretación de la cadena causal:**

| Sensor | Patrón en 6 meses | Mecanismo |
|--------|-------------------|-----------|
| **T1** | Descenso estacional Ago→Ene (18°C → 6.5°C) | Ciclo térmico anual |
| **R**  | Picos esporádicos (Aug–Sep), baseline constante ~0.11 mm | Eventos de lluvia convectiva de verano |
| **P**  | Valores negativos (−2.9 kPa = succión mátrica) · pico puntual máx. 325 kPa | Suelo no saturado; pico = evento hidráulico extremo |
| **Dleft** | Rango operativo 2155–2198 mm con dip diario al mediodía (~12.5 mm) | Dip = artefacto mecánico de bomba de riego, NO contracción térmica |

> **Nota:** el Δ≈+427 mm que puede aparecer al inicio de la serie es el período de
> arranque del sensor (desde ~23 mm hasta el rango operativo). No es deformación real del talud.
""")

# ──────────────────────────────────────────────────────────────────────────────
# VIZ 5 — CROSS-CORRELACIÓN CON LAG (PASO 7)
# ──────────────────────────────────────────────────────────────────────────────
elif viz_key == "crosscorr":
    st.markdown(
        "**Cross-correlación con lag** (máx. ±168 h = 1 semana). "
        "La línea vertical roja marca el lag óptimo (|ρ| máximo). "
        "Lag positivo = X precede a Y."
    )

    pares_lag = [
        ("R",  "H1",    "R → H1 (lluvia → humedad)"),
        ("R",  "P",     "R → P (lluvia → presión poros)"),
        ("R",  "Dleft", "R → Dleft (lluvia → deformación)"),
        ("T1", "H1",    "T1 → H1 (deriva térmica del sensor)"),
    ]
    pares_lag = [(x, y, t) for x, y, t in pares_lag if x in df.columns and y in df.columns]

    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=[t for _, _, t in pares_lag],
        vertical_spacing=0.14,
        horizontal_spacing=0.10,
    )

    resumen = []
    for idx, (xc, yc, titulo) in enumerate(pares_lag):
        row, col_pos = divmod(idx, 2)
        lags, cc = lag_corr_cached(df, xc, yc)
        peak_idx = int(np.argmax(np.abs(cc)))
        opt_lag = int(lags[peak_idx])
        peak_rho = float(cc[peak_idx])
        resumen.append((xc, yc, opt_lag, peak_rho))

        fig.add_trace(
            go.Scatter(
                x=lags,
                y=cc,
                mode="lines",
                line=dict(color="#2980B9", width=1.0),
                name=titulo,
                showlegend=False,
                hovertemplate="lag=%{x}h<br>ρ=%{y:.3f}<extra></extra>",
            ),
            row=row + 1, col=col_pos + 1,
        )
        # Línea en lag óptimo
        fig.add_vline(
            x=opt_lag,
            line_dash="dash", line_color="red", line_width=1.5,
            annotation_text=f"lag={opt_lag:+d}h\nρ={peak_rho:+.3f}",
            annotation_font_size=9,
            annotation_position="top right",
            row=row + 1, col=col_pos + 1,
        )
        fig.add_hline(
            y=0, line_color="gray", line_width=0.5,
            row=row + 1, col=col_pos + 1,
        )
        fig.update_xaxes(title_text="Lag (horas)", row=row + 1, col=col_pos + 1)
        fig.update_yaxes(title_text="ρ", row=row + 1, col=col_pos + 1)

    fig.update_layout(
        height=600,
        title_text="Cross-correlación con lag — Cadena causal (±168 h)",
    )
    st.plotly_chart(fig, use_container_width=True)

    # Tabla de resultados
    st.markdown("**Resultados del lag óptimo:**")
    CAUSAL_INTERP = {
        ("R",  "H1"):    "Lluvia → circuito se enfría → H1 baja (termómetro indirecto)",
        ("R",  "P"):     "Lag espurio por tendencia estacional (ρ débil)",
        ("R",  "Dleft"): "Efecto térmico dominante: lluvia enfría → Dleft disminuye",
        ("T1", "H1"):    "Deriva térmica robusta: T1 → H1 con inercia de circuito",
    }
    rows_tabla = []
    for xc, yc, opt_lag, peak_rho in resumen:
        fuerza = "fuerte" if abs(peak_rho) > 0.5 else "moderada" if abs(peak_rho) > 0.3 else "débil"
        interp = CAUSAL_INTERP.get((xc, yc), f"{xc} → {yc}")
        rows_tabla.append({
            "Par": f"{xc} → {yc}",
            "Lag óptimo (h)": opt_lag,
            "ρ peak": round(peak_rho, 3),
            "Fuerza": fuerza,
            "Interpretación": interp,
        })
    st.dataframe(pd.DataFrame(rows_tabla), hide_index=True, use_container_width=True)

    st.markdown("""
**Cómo leer la cross-correlación:**

- **Lag positivo** = X precede a Y en ese número de horas (X es probable causa de Y).
- **Signo de ρ:** positivo = Y sube cuando X sube; negativo = Y baja cuando X sube.
- **T1→H1** (lag≈+29h, ρ≈+0.711): el dato más robusto → H1 es un termómetro de circuito con 29 horas de inercia térmica.
- **R→Dleft** (lag≈−2h, ρ≈−0.290): la lluvia precede a la caída de Dleft porque llega con frío, no porque haya deformación real.
- Un lag de 97 horas con ρ < 0.2 (caso R→P) es un resultado espurio dominado por tendencia estacional, no causalidad hidráulica real.
""")
