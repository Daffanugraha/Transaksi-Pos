"""
visualizer.py
Creates Plotly charts for the POS dashboard.
"""

import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots


# Color palette
PALETTE = {
    "primary":   "#00D4FF",
    "secondary": "#7B2FBE",
    "accent":    "#FF6B35",
    "success":   "#00C896",
    "warning":   "#FFD60A",
    "danger":    "#FF3B6E",
    "bg":        "#0A0E1A",
    "surface":   "#111827",
    "text":      "#E2E8F0",
    "muted":     "#64748B",
}

SEVERITY_COLORS = {
    "Critical": "#FF3B6E",
    "High":     "#FF6B35",
    "Medium":   "#FFD60A",
    "Low":      "#00C896",
    "Normal":   "#64748B",
}

_layout_base = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(family="'DM Sans', sans-serif", color=PALETTE["text"], size=12),
    margin=dict(l=20, r=20, t=40, b=20),
    legend=dict(bgcolor="rgba(255,255,255,0.05)", bordercolor="rgba(255,255,255,0.1)", borderwidth=1),
)


def _apply_base(fig: go.Figure) -> go.Figure:
    fig.update_layout(**_layout_base)
    fig.update_xaxes(gridcolor="rgba(255,255,255,0.05)", zeroline=False)
    fig.update_yaxes(gridcolor="rgba(255,255,255,0.05)", zeroline=False)
    return fig


# Chart 1: Daily revenue
def chart_daily_revenue(df: pd.DataFrame) -> go.Figure:
    daily = df.groupby("date")["total_amount"].sum().reset_index()
    anomaly_daily = df[df["is_anomaly"] == True].groupby("date")["total_amount"].sum().reset_index()

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=daily["date"], y=daily["total_amount"],
        mode="lines+markers",
        line=dict(color=PALETTE["primary"], width=2.5),
        marker=dict(size=6, color=PALETTE["primary"]),
        fill="tozeroy",
        fillcolor="rgba(0,212,255,0.08)",
        name="Total Pendapatan",
        hovertemplate="<b>%{x|%d %b}</b><br>Rp %{y:,.0f}<extra></extra>",
    ))
    
    if not anomaly_daily.empty:
        fig.add_trace(go.Scatter(
            x=anomaly_daily["date"], y=anomaly_daily["total_amount"],
            mode="markers",
            # BAGIAN YANG DIUBAH: Mengganti symbol="x" menjadi "circle" dan memperhalus styling
            marker=dict(
                size=12, 
                color=PALETTE["danger"], 
                symbol="circle", 
                line=dict(color="#ffffff", width=2) # Memberikan border putih/cerah agar rapi
            ),
            name="Indikasi Temuan",
            hovertemplate="<b>%{x|%d %b}</b><br>Nilai Temuan: Rp %{y:,.0f}<extra></extra>",
        ))
        
    fig.update_layout(title="Pendapatan Harian", **_layout_base)
    fig.update_xaxes(gridcolor="rgba(255,255,255,0.05)", zeroline=False)
    fig.update_yaxes(gridcolor="rgba(255,255,255,0.05)", zeroline=False, tickformat=",.0f", tickprefix="Rp ")
    return fig


# Chart 2: Anomaly by severity
def chart_severity_donut(df: pd.DataFrame) -> go.Figure:
    counts = df["anomaly_severity"].value_counts()
    labels = counts.index.tolist()
    values = counts.values.tolist()
    colors = [SEVERITY_COLORS.get(l, PALETTE["muted"]) for l in labels]

    fig = go.Figure(go.Pie(
        labels=labels, values=values,
        hole=0.65,  # Sedikit diperbesar agar teks di tengah lebih lega
        marker=dict(colors=colors, line=dict(color=PALETTE["bg"], width=2)),
        textinfo="percent",  # Hanya tampilkan angka persen agar tidak penuh
        textposition="outside",  # Pindahkan teks ke luar irisan grafik
        hovertemplate="<b>Tingkat %{label}</b><br>%{value} transaksi (%{percent})<extra></extra>",
    ))
    
    fig.update_layout(
        title="Proporsi Tingkat Risiko",
        # Menyembunyikan teks yang benar-benar tidak muat agar tidak tabrakan
        uniformtext_minsize=10, 
        uniformtext_mode='hide',
        annotations=[dict(
            text=f"<b>{df['is_anomaly'].sum()}</b><br>Temuan", 
            x=0.5, y=0.5,
            font_size=14, 
            showarrow=False, 
            font_color=PALETTE["text"]
        )],
        **_layout_base,
    )
    
    # Memastikan legend tertata rapi di sebelah kanan
    fig.update_layout(
        legend=dict(
            orientation="v",
            yanchor="middle",
            y=0.5,
            xanchor="left",
            x=1.1
        )
    )
    
    return fig


# Chart 3: Hourly transaction heatmap
def chart_hourly_heatmap(df: pd.DataFrame) -> go.Figure:
    if "hour" not in df.columns or "day_of_week" not in df.columns:
        return go.Figure()

    day_order = ["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"]
    pivot = df.groupby(["day_of_week","hour"])["total_amount"].sum().unstack(fill_value=0)
    pivot = pivot.reindex([d for d in day_order if d in pivot.index])

    def format_hour(val):
        try:
            return f"{int(float(val)):02d}:00"
        except (ValueError, TypeError):
            return str(val)

    fig = go.Figure(go.Heatmap(
        z=pivot.values,
        x=[format_hour(h) for h in pivot.columns],
        y=pivot.index.tolist(),
        colorscale=[[0, PALETTE["surface"]], [0.5, PALETTE["secondary"]], [1, PALETTE["primary"]]],
        hovertemplate="<b>%{y} %{x}</b><br>Rp %{z:,.0f}<extra></extra>",
        showscale=True,
        colorbar=dict(tickformat=",.0f", title="Nilai"),
    ))
    fig.update_layout(title="Intensitas Transaksi (Berdasarkan Jam & Hari)", **_layout_base)
    fig.update_xaxes(gridcolor="rgba(0,0,0,0)")
    fig.update_yaxes(gridcolor="rgba(0,0,0,0)")
    return fig


# Chart 4: Anomaly type bar chart
def chart_anomaly_types(df: pd.DataFrame) -> go.Figure:
    anomalies = df[df["is_anomaly"] == True]
    if anomalies.empty:
        return go.Figure()

    all_types = []
    for types in anomalies["anomaly_types"].dropna():
        all_types.extend([t.strip() for t in types.split(",") if t.strip()])

    type_counts = pd.Series(all_types).value_counts()

    fig = go.Figure(go.Bar(
        x=type_counts.values,
        y=type_counts.index,
        orientation="h",
        marker=dict(
            color=type_counts.values,
            colorscale=[[0, PALETTE["warning"]], [1, PALETTE["danger"]]],
            showscale=False,
        ),
        hovertemplate="<b>%{y}</b><br>%{x} kasus ditemukan<extra></extra>",
        text=type_counts.values,
        textposition="outside",
        textfont=dict(color=PALETTE["text"]),
    ))
    fig.update_layout(title="Kategori Indikasi Risiko", **_layout_base, bargap=0.3)
    fig.update_xaxes(gridcolor="rgba(255,255,255,0.05)", zeroline=False, title="Jumlah Kasus")
    fig.update_yaxes(gridcolor="rgba(0,0,0,0)")
    return fig


# Chart 5: Cashier performance
def chart_cashier_performance(df: pd.DataFrame) -> go.Figure:
    if "cashier_name" not in df.columns:
        return go.Figure()

    cashier_stats = df.groupby("cashier_name").agg(
        total_revenue=("total_amount", "sum"),
        total_transactions=("transaction_id", "count"),
        anomaly_count=("is_anomaly", "sum"),
    ).reset_index()

    cashier_stats["anomaly_rate"] = (
        cashier_stats["anomaly_count"] / cashier_stats["total_transactions"] * 100
    ).round(1)

    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(go.Bar(
        name="Total Pendapatan",
        x=cashier_stats["cashier_name"],
        y=cashier_stats["total_revenue"],
        marker_color=PALETTE["primary"],
        opacity=0.85,
        hovertemplate="<b>%{x}</b><br>Kontribusi: Rp %{y:,.0f}<extra></extra>",
    ), secondary_y=False)
    fig.add_trace(go.Scatter(
        name="% Indikasi Risiko",
        x=cashier_stats["cashier_name"],
        y=cashier_stats["anomaly_rate"],
        mode="markers+lines",
        marker=dict(size=10, color=PALETTE["danger"], symbol="diamond"),
        line=dict(color=PALETTE["danger"], width=2, dash="dot"),
        hovertemplate="<b>%{x}</b><br>Rasio Risiko: %{y:.1f}%<extra></extra>",
    ), secondary_y=True)

    fig.update_layout(title="Aktivitas & Profil Risiko Staf Kasir", **_layout_base)
    fig.update_xaxes(gridcolor="rgba(255,255,255,0.05)")
    fig.update_yaxes(title_text="Volume Pendapatan (Rp)", secondary_y=False, gridcolor="rgba(255,255,255,0.05)",
                     tickformat=",.0f", tickprefix="Rp ")
    fig.update_yaxes(title_text="Rasio Risiko (%)", secondary_y=True, gridcolor="rgba(0,0,0,0)",
                     ticksuffix="%")
    return fig


# Chart 6: Category breakdown
def chart_category_revenue(df: pd.DataFrame) -> go.Figure:
    if "category" not in df.columns:
        return go.Figure()

    cat_data = df.groupby("category")["total_amount"].sum().sort_values(ascending=True)

    colors = px.colors.sample_colorscale(
        [[0, PALETTE["secondary"]], [1, PALETTE["primary"]]], len(cat_data)
    )

    fig = go.Figure(go.Bar(
        x=cat_data.values,
        y=cat_data.index,
        orientation="h",
        marker_color=colors,
        hovertemplate="<b>%{y}</b><br>Rp %{x:,.0f}<extra></extra>",
        text=[f"Rp {v:,.0f}" for v in cat_data.values],
        textposition="outside",
        textfont=dict(color=PALETTE["text"], size=10),
    ))
    fig.update_layout(title="Distribusi Pendapatan per Kategori", **_layout_base, bargap=0.25)
    fig.update_xaxes(gridcolor="rgba(255,255,255,0.05)", tickformat=",.0f", tickprefix="Rp ")
    fig.update_yaxes(gridcolor="rgba(0,0,0,0)")
    return fig


# Chart 7: Amount distribution
def chart_amount_distribution(df: pd.DataFrame) -> go.Figure:
    normal = df[df["is_anomaly"] == False]["total_amount"]
    anomaly = df[df["is_anomaly"] == True]["total_amount"]

    fig = go.Figure()
    fig.add_trace(go.Histogram(
        x=normal, name="Wajar",
        marker_color=PALETTE["success"],
        opacity=0.7,
        nbinsx=30,
        hovertemplate="Rp %{x:,.0f}: %{y} transaksi<extra>Wajar</extra>",
    ))
    fig.add_trace(go.Histogram(
        x=anomaly, name="Perlu Verifikasi",
        marker_color=PALETTE["danger"],
        opacity=0.85,
        nbinsx=15,
        hovertemplate="Rp %{x:,.0f}: %{y} transaksi<extra>Perlu Verifikasi</extra>",
    ))
    fig.update_layout(
        title="Sebaran Nominal Transaksi",
        barmode="overlay",
        **_layout_base,
    )
    fig.update_xaxes(gridcolor="rgba(255,255,255,0.05)", tickformat=",.0f", tickprefix="Rp ")
    fig.update_yaxes(gridcolor="rgba(255,255,255,0.05)", title="Frekuensi Kemunculan")
    return fig