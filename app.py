"""
app.py  POS Anomaly Detector
Main Streamlit application entry point.
"""

import streamlit as st
import pandas as pd
import numpy as np
import io
import os
from dotenv import load_dotenv

from components.data_processor import load_csv, preprocess, get_summary_stats
from components.anomaly_detector import detect_anomalies
from components.visualizer import (
    chart_daily_revenue, chart_severity_donut, chart_hourly_heatmap,
    chart_anomaly_types, chart_cashier_performance,
    chart_category_revenue, chart_amount_distribution,
)
from components.llm_analyzer import generate_analysis, AVAILABLE_MODELS

# Load environment variables
load_dotenv()


# Helper: Templating & UI
def render_template(template_name: str, **kwargs) -> str:
    """Membaca file .html dari assets/templates dan mengganti {{ placeholder }} dengan nilai kwargs."""
    filepath = os.path.join(os.path.dirname(__file__), "assets", "templates", template_name)
    with open(filepath, "r", encoding="utf-8") as file:
        content = file.read()
    for key, value in kwargs.items():
        content = content.replace(f"{{{{ {key} }}}}", str(value))
    return content

def get_badge_html(severity: str) -> str:
    """Menghasilkan HTML badge statis berdasarkan severity."""
    cls = {
        "Critical": "critical", "High": "high", "Medium": "medium",
        "Low": "low", "Normal": "normal",
    }.get(severity, "normal")
    return f'<span class="badge badge-{cls}">{severity}</span>'

def load_css(file_name: str):
    """Membaca file .css."""
    filepath = os.path.join(os.path.dirname(__file__), file_name)
    with open(filepath, "r", encoding="utf-8") as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)


# Page config
st.set_page_config(
    page_title="POS Anomaly Detector",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Load CSS
load_css(os.path.join("assets", "styles", "main.css"))

groq_api_key = os.environ.get("GROQ_API_KEY", "")

# Sidebar
with st.sidebar:
    st.markdown("### Pengaturan Sistem")

    selected_model = st.selectbox(
        "Pilihan Mesin Analisis",
        AVAILABLE_MODELS,
        index=0,
    )

    st.markdown("### Kriteria Deteksi")

    z_threshold = st.slider(
        "Batas Toleransi Z-Score", 1.5, 4.0, 2.5, 0.1,
        help="Semakin kecil angkanya, sistem akan semakin peka dalam menangkap kejanggalan harga.",
    )
    max_discount = st.slider("Batas Wajar Diskon (%)", 10, 80, 40, 5)
    max_quantity = st.slider("Batas Wajar Jumlah Barang", 10, 200, 50, 5)
    late_night_start = st.slider("Awal Jam Rawan (Malam)", 20, 23, 23, 1)
    late_night_end = st.slider("Akhir Jam Rawan (Pagi)", 4, 8, 6, 1)


# Header
st.markdown(render_template("header.html"), unsafe_allow_html=True)


# File Upload
col_upload, col_demo = st.columns([3, 1])

with col_upload:
    uploaded_file = st.file_uploader(
        "Unggah Data Transaksi POS (CSV)",
        type=["csv"],
        help="Pastikan ada kolom: transaction_id, date, total_amount, cashier_id, dan cashier_name.",
    )

with col_demo:
    st.markdown("<br>", unsafe_allow_html=True)
    use_demo = st.button("Coba dengan Data Sampel", use_container_width=True)


# Load Data
df_raw = None

if use_demo or (not uploaded_file and st.session_state.get("demo_loaded")):
    st.session_state["demo_loaded"] = True
    demo_path = os.path.join(os.path.dirname(__file__), "data", "dummy_transactions.csv")
    df_raw, msgs = load_csv(demo_path)
    for m in msgs:
        clean_msg = m.replace("✅", "").replace("⚠️", "").strip()
        if "✅" in m:
            st.success(clean_msg + " (Data sampel berhasil dimuat)")
        else:
            st.warning(clean_msg)

elif uploaded_file:
    st.session_state["demo_loaded"] = False
    df_raw, msgs = load_csv(uploaded_file)
    for m in msgs:
        clean_msg = m.replace("✅", "").replace("⚠️", "").strip()
        if "✅" in m:
            st.success(clean_msg)
        else:
            st.warning(clean_msg)


# Main App
if df_raw is not None and not df_raw.empty:

    # Preprocess
    df = preprocess(df_raw)

    # Detect anomalies
    anomaly_config = {
        "z_threshold": z_threshold,
        "max_discount": max_discount,
        "max_quantity": max_quantity,
        "late_night_start": late_night_start,
        "late_night_end": late_night_end,
    }
    df = detect_anomalies(df, config=anomaly_config)

    stats = get_summary_stats(df)
    anomalies_df = df[df["is_anomaly"] == True].copy()
    anomaly_count = len(anomalies_df)
    anomaly_rate = anomaly_count / len(df) * 100

    # Tabs ditaruh paling atas agar tidak tabrakan dengan indikator
    st.markdown("<br>", unsafe_allow_html=True) # Jarak agar rapi
    tab_overview, tab_anomaly, tab_detail, tab_ai = st.tabs([
        "Ringkasan", "Catatan Anomali", "Data Transaksi", "Analisis Mendalam"
    ])

    # TAB 1: OVERVIEW
    with tab_overview:
        # KPI Cards dipindah ke dalam Tab Ringkasan agar tampilan utamanya luas
        kpi_html = render_template(
            "kpi_grid.html",
            total_transactions=f"{stats['total_transactions']:,}",
            date_start=stats['date_range'][0],
            date_end=stats['date_range'][1],
            total_revenue=f"{stats['total_revenue']:,.0f}",
            avg_transaction=f"{stats['avg_transaction']:,.0f}",
            anomaly_count=anomaly_count,
            anomaly_rate=f"{anomaly_rate:.1f}",
            num_cashiers=stats['num_cashiers'],
            num_products=stats['num_products']
        )
        st.markdown(kpi_html, unsafe_allow_html=True)
        st.markdown("<br>", unsafe_allow_html=True) # Jarak antara KPI dan Grafik

        col1, col2 = st.columns([3, 2])
        with col1:
            with st.container():
                st.markdown('<div class="chart-card">', unsafe_allow_html=True)
                st.plotly_chart(chart_daily_revenue(df), use_container_width=True, config={"displayModeBar": False})
                st.markdown('</div>', unsafe_allow_html=True)
        with col2:
            with st.container():
                st.markdown('<div class="chart-card">', unsafe_allow_html=True)
                st.plotly_chart(chart_severity_donut(df), use_container_width=True, config={"displayModeBar": False})
                st.markdown('</div>', unsafe_allow_html=True)

        col3, col4 = st.columns(2)
        with col3:
            st.markdown('<div class="chart-card">', unsafe_allow_html=True)
            st.plotly_chart(chart_category_revenue(df), use_container_width=True, config={"displayModeBar": False})
            st.markdown('</div>', unsafe_allow_html=True)
        with col4:
            st.markdown('<div class="chart-card">', unsafe_allow_html=True)
            st.plotly_chart(chart_amount_distribution(df), use_container_width=True, config={"displayModeBar": False})
            st.markdown('</div>', unsafe_allow_html=True)

        if "hour" in df.columns:
            st.markdown('<div class="chart-card">', unsafe_allow_html=True)
            st.plotly_chart(chart_hourly_heatmap(df), use_container_width=True, config={"displayModeBar": False})
            st.markdown('</div>', unsafe_allow_html=True)

        st.markdown('<div class="chart-card">', unsafe_allow_html=True)
        st.plotly_chart(chart_cashier_performance(df), use_container_width=True, config={"displayModeBar": False})
        st.markdown('</div>', unsafe_allow_html=True)

    # TAB 2: ANOMALY LIST
    with tab_anomaly:
        if anomalies_df.empty:
            st.success("Bagus, operasional berjalan wajar. Tidak ditemukan indikasi anomali pada data ini.")
        else:
            col_a, col_b = st.columns([2, 3])
            with col_a:
                st.markdown('<div class="chart-card">', unsafe_allow_html=True)
                st.plotly_chart(chart_anomaly_types(df), use_container_width=True, config={"displayModeBar": False})
                st.markdown('</div>', unsafe_allow_html=True)
            with col_b:
                # Severity summary
                sev_counts = anomalies_df["anomaly_severity"].value_counts()
                st.markdown('<div class="chart-card">', unsafe_allow_html=True)
                st.markdown("#### Tingkat Risiko Temuan")
                for sev in ["Critical", "High", "Medium", "Low"]:
                    cnt = sev_counts.get(sev, 0)
                    pct = cnt / anomaly_count * 100 if anomaly_count else 0
                    color = {"Critical":"#FF3B6E","High":"#FF6B35","Medium":"#FFD60A","Low":"#00C896"}[sev]
                    
                    row_html = render_template(
                        "severity_row.html",
                        badge_html=get_badge_html(sev),
                        color=color,
                        cnt=cnt,
                        pct=f"{pct:.1f}"
                    )
                    st.markdown(row_html, unsafe_allow_html=True)
                st.markdown('</div>', unsafe_allow_html=True)

            # Filter controls
            st.markdown('<div class="section-title">Rincian Temuan Anomali</div>', unsafe_allow_html=True)
            fc1, fc2 = st.columns(2)
            with fc1:
                filter_sev = st.multiselect(
                    "Saring berdasarkan Risiko", ["Critical","High","Medium","Low"],
                    default=["Critical","High","Medium","Low"]
                )
            with fc2:
                filter_type = st.text_input("Cari jenis temuan...", placeholder="contoh: Transaksi Malam")

            filtered = anomalies_df[anomalies_df["anomaly_severity"].isin(filter_sev)]
            if filter_type:
                filtered = filtered[filtered["anomaly_types"].str.contains(filter_type, case=False, na=False)]

            # Render table
            if not filtered.empty:
                rows_html = ""
                for _, row in filtered.sort_values("anomaly_score", ascending=False).iterrows():
                    sev = row.get("anomaly_severity", "Normal")
                    rows_html += render_template(
                        "anomaly_table_row.html",
                        transaction_id=row.get('transaction_id',''),
                        date_str=row.get('date_str',''),
                        cashier_name=row.get('cashier_name',''),
                        product_name=row.get('product_name',''),
                        total_amount=f"{float(row.get('total_amount',0)):,.0f}",
                        badge_html=get_badge_html(sev),
                        anomaly_types=row.get('anomaly_types',''),
                        anomaly_details=str(row.get('anomaly_details',''))[:120]
                    )
                
                table_html = render_template(
                    "anomaly_table_layout.html",
                    rows_html=rows_html,
                    filtered_count=len(filtered),
                    total_count=anomaly_count
                )
                st.markdown(table_html, unsafe_allow_html=True)
            else:
                st.info("Sistem tidak menemukan data yang cocok dengan pencarian Anda.")

            # Download button
            csv_out = filtered.to_csv(index=False).encode("utf-8")
            st.download_button(
                "Unduh Laporan Temuan (CSV)",
                data=csv_out,
                file_name="laporan_temuan_transaksi.csv",
                mime="text/csv",
            )

    # TAB 3: DETAIL DATA
    with tab_detail:
        st.markdown("### Seluruh Data Transaksi")

        show_cols = [c for c in [
            "transaction_id","date_str","time","cashier_name","product_name",
            "category","quantity","unit_price","total_amount","payment_method",
            "is_anomaly","anomaly_severity","anomaly_types"
        ] if c in df.columns]

        search_q = st.text_input("Cari Transaksi...", placeholder="Ketik nomor transaksi, nama kasir, atau produk...")
        show_df = df[show_cols].copy()

        if search_q:
            mask = show_df.apply(lambda col: col.astype(str).str.contains(search_q, case=False, na=False)).any(axis=1)
            show_df = show_df[mask]

        st.dataframe(
            show_df,
            use_container_width=True,
            height=500,
            column_config={
                "total_amount": st.column_config.NumberColumn("Total (Rp)", format="Rp %,.0f"),
                "unit_price": st.column_config.NumberColumn("Harga Satuan (Rp)", format="Rp %,.0f"),
                "is_anomaly": st.column_config.CheckboxColumn("Tanda Anomali"),
                "anomaly_severity": st.column_config.TextColumn("Status"),
            },
        )
        st.caption(f"Menampilkan {len(show_df):,} dari total {len(df):,} baris data yang ada.")

    # TAB 4: AI ANALYSIS
    with tab_ai:
        st.markdown(render_template("ai_header.html"), unsafe_allow_html=True)

        # Build anomaly summary
        sev_counts = anomalies_df["anomaly_severity"].value_counts() if not anomalies_df.empty else pd.Series()
        all_types = []
        for t in anomalies_df.get("anomaly_types", pd.Series()).dropna():
            all_types.extend([x.strip() for x in t.split(",") if x.strip()])
        top_type = pd.Series(all_types).value_counts().idxmax() if all_types else "Belum Ada"

        anomaly_summary = {
            "total_anomalies": anomaly_count,
            "anomaly_rate": anomaly_rate,
            "critical": int(sev_counts.get("Critical", 0)),
            "high": int(sev_counts.get("High", 0)),
            "medium": int(sev_counts.get("Medium", 0)),
            "low": int(sev_counts.get("Low", 0)),
            "top_type": top_type,
        }

        sample_anomalies = anomalies_df.head(10).to_dict("records")

        # Status indicator
        if groq_api_key:
            st.success("Modul analisis siap digunakan.")
        else:
            st.warning("Silakan masukkan Kunci Akses di menu samping untuk mulai menggunakan fitur analisis ini.")

        col_btn, col_info = st.columns([2, 3])
        with col_btn:
            run_analysis = st.button("Mulai Proses Analisis", use_container_width=True)
        with col_info:
            st.markdown(
                '<span style="color:#64748B;font-size:0.8rem">'
                'Sistem akan membantu Anda mengevaluasi risiko operasional dan memberikan rekomendasi strategis.'
                '</span>',
                unsafe_allow_html=True
            )

        if run_analysis:
            with st.spinner("Mohon tunggu sebentar, sistem sedang memetakan pola dan metrik operasional Anda..."):
                result = generate_analysis(
                    stats, anomaly_summary, sample_anomalies,
                    api_key=groq_api_key,
                    model=selected_model,
                )
            st.session_state["ai_result"] = result

        if "ai_result" in st.session_state:
            result_text = st.session_state["ai_result"]
            is_error = "error" in result_text.lower() or "gagal" in result_text.lower()

            if is_error:
                st.error(result_text)
            else:
                st.markdown(f'<div class="ai-box">{result_text}</div>', unsafe_allow_html=True)
                st.download_button(
                    "Unduh Ringkasan Laporan (TXT)",
                    data=result_text.encode("utf-8"),
                    file_name="ringkasan_laporan_eksekutif.txt",
                    mime="text/plain",
                )

else:
    # Landing state
    st.markdown(render_template("landing_page.html"), unsafe_allow_html=True)