"""
anomaly_detector.py
Detects abnormal POS transactions using AI judgment and statistical pre-filtering.
"""

import pandas as pd
import numpy as np
import json
import os
import streamlit as st

def _severity(score: float) -> str:
    # Digunakan hanya sebagai fallback jika AI tidak memberikan severity
    if score >= 2.5: return "Critical"
    if score >= 2.0: return "High"
    if score >= 1.0: return "Medium"
    return "Low"

@st.cache_data(show_spinner=False)
def detect_anomalies(df: pd.DataFrame, config: dict | None = None) -> pd.DataFrame:
    """
    AI-Based Anomaly Detection (Hybrid Approach).
    1. Saring data yang berpotensi anomali menggunakan threshold.
    2. Kirim data yang tersaring ke LLM untuk dinilai (Judged) secara kontekstual.
    """
    api_key = os.environ.get("GROQ_API_KEY", "")
    
    cfg = {
        "z_threshold": 2.0, # Lebih longgar untuk pre-filter agar banyak kandidat
        "late_night_start": 22,
        "late_night_end": 6,
        "max_discount": 25,
        "max_quantity": 20,
        **(config or {}),
    }

    df = df.copy()
    df["transaction_id"] = df["transaction_id"].astype(str)
    
    # ── TAHAP 1: PRE-FILTERING (Pencarian Kandidat) ──────────────────────
    suspicious_mask = pd.Series(False, index=df.index)
    
    # Rule dasar untuk mencari kandidat yang akan dihakimi AI
    if "hour" in df.columns:
        suspicious_mask |= (df["hour"] >= cfg["late_night_start"]) | (df["hour"] < cfg["late_night_end"])
    if "discount_percent" in df.columns:
        suspicious_mask |= (df["discount_percent"] > cfg["max_discount"])
    if "quantity" in df.columns:
        suspicious_mask |= (df["quantity"] > cfg["max_quantity"])
    if "is_void" in df.columns:
        suspicious_mask |= (df["is_void"] == True)
    if "total_amount" in df.columns:
        mean, std = df["total_amount"].mean(), df["total_amount"].std()
        if std > 0:
            suspicious_mask |= (((df["total_amount"] - mean) / std).abs() > cfg["z_threshold"])

    candidates_df = df[suspicious_mask].copy()

    # Siapkan struktur default jika tidak ada anomali atau API gagal
    default_anomaly_cols = pd.DataFrame(columns=[
        "transaction_id", "is_anomaly", "anomaly_types", 
        "anomaly_severity", "anomaly_details", "anomaly_score"
    ])

    if candidates_df.empty or not api_key:
        return _merge_results(df, default_anomaly_cols)

    # ── TAHAP 2: BIARKAN AI YANG MENJADI JURI ──────────────────────────────
    try:
        from groq import Groq
        client = Groq(api_key=api_key)
        
        # Ambil kolom esensial saja untuk menghemat token API
        cols = [c for c in ["transaction_id", "time", "cashier_name", "product_name", 
                            "quantity", "unit_price", "total_amount", "discount_percent", "is_void"] 
                if c in candidates_df.columns]
        
        # Batasi maksimal 60 transaksi mencurigakan sekaligus agar tidak over-token
        data_to_judge = candidates_df[cols].head(60).to_dict(orient="records")
        
        prompt = f"""
            Kamu adalah AI Auditor Keuangan untuk sistem kasir (POS). 
            Berikut adalah transaksi yang 'mencurigakan' hasil pre-filter sistem.
            Tugasmu adalah menjadi JURI untuk menentukan apakah transaksi ini BENAR-BENAR anomali yang merugikan bisnis, atau hanya operasional biasa.

            ATURAN TOKO (HARUS DIPATUHI - Berdasarkan Setting Pengguna):
            - Diskon di atas {cfg['max_discount']}% dianggap melanggar batas normal.
            - Kuantitas di atas {cfg['max_quantity']} unit dalam satu struk sangat mencurigakan.
            - Transaksi antara pukul {cfg['late_night_start']}:00 hingga {cfg['late_night_end']}:00 adalah di luar jam operasional resmi.
            - Z-Score Threshold: {cfg['z_threshold']} (Nilai nominal transaksi yang menyimpang jauh dari rata-rata).

            Gunakan logika bisnis untuk mengevaluasi data berdasarkan aturan toko di atas.
            - Jika sebuah transaksi melanggar aturan toko namun nominal akhirnya sangat kecil (misal Rp 2.000), severity-nya mungkin hanya "Low".
            - Jika pelanggarannya merugikan secara material (jutaan), beri severity "High" atau "Critical".
            - Transaksi dibatalkan (void) dengan nominal besar wajib ditandai.

            Data Transaksi:
            {json.dumps(data_to_judge, default=str)}

            Kembalikan HANYA format JSON object persis seperti ini:
            {{
                "anomalies": [
                    {{
                        "transaction_id": "...",
                        "anomaly_types": "Nama Anomali (misal: Diskon Ekstrem / Di Luar Jam Kerja)",
                        "anomaly_severity": "High", 
                        "anomaly_details": "Penjelasan logis...",
                        "anomaly_score": 2.5
                    }}
                ]
            }}
            (Score dari 0.1 sampai 3.0. Severity pilih antara: Low, Medium, High, Critical).
            Hanya masukkan ke array jika kamu YAKIN itu anomali. Jika wajar, abaikan.
            """
            # Gunakan JSON mode dari Groq agar balasan selalu valid
        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            response_format={"type": "json_object"} 
        )
        
        ai_result = json.loads(response.choices[0].message.content)
        anomaly_list = ai_result.get("anomalies", [])
        
        if anomaly_list:
            ai_df = pd.DataFrame(anomaly_list)
            ai_df["is_anomaly"] = True
        else:
            ai_df = default_anomaly_cols
            
        return _merge_results(df, ai_df)

    except Exception as e:
        # Fallback aman jika API error (rate limit / putus koneksi)
        st.warning(f"⚠️ Proses AI terganggu. Menampilkan data tanpa deteksi anomali. Error: {str(e)}")
        return _merge_results(df, default_anomaly_cols)


def _merge_results(original_df: pd.DataFrame, anomaly_df: pd.DataFrame) -> pd.DataFrame:
    """Fungsi bantuan untuk menggabungkan hasil dari AI kembali ke tabel asli."""
    if anomaly_df.empty:
        result = original_df.copy()
        result["is_anomaly"] = False
        result["anomaly_types"] = ""
        result["anomaly_severity"] = "Normal"
        result["anomaly_details"] = ""
        result["anomaly_score"] = 0.0
        return result

    # Pastikan tipe data ID sama agar bisa digabungkan
    original_df["transaction_id"] = original_df["transaction_id"].astype(str)
    anomaly_df["transaction_id"] = anomaly_df["transaction_id"].astype(str)
    
    result = original_df.merge(anomaly_df, on="transaction_id", how="left")
    result["is_anomaly"] = result["is_anomaly"].fillna(False)
    result["anomaly_types"] = result["anomaly_types"].fillna("")
    result["anomaly_severity"] = result["anomaly_severity"].fillna("Normal")
    result["anomaly_details"] = result["anomaly_details"].fillna("")
    result["anomaly_score"] = result["anomaly_score"].fillna(0.0)
    
    return result


def severity_color(severity: str) -> str:
    return {
        "Critical": "#FF3B3B",
        "High":     "#FF8C00",
        "Medium":   "#FFD700",
        "Low":      "#90EE90",
        "Normal":   "#4CAF50",
    }.get(severity, "#888888")