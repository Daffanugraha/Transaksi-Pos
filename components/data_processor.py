"""
data_processor.py
Handles loading, validating, and preprocessing POS transaction data.
"""

import pandas as pd
import numpy as np
from datetime import datetime
from typing import Tuple
import os
import streamlit as st

# Pastikan file column_mapper.py sudah Anda buat di dalam folder components
try:
    from components.column_mapper import map_columns_with_ai
except ImportError:
    map_columns_with_ai = None


REQUIRED_COLUMNS = [
    "transaction_id", "date", "total_amount", "cashier_id", "cashier_name"
]

OPTIONAL_COLUMNS = [
    "time", "product_name", "category", "quantity", "unit_price",
    "payment_method", "store_id", "customer_id", "discount_percent",
    "is_void", "product_id"
]


def standardize_columns_basic(df: pd.DataFrame) -> pd.DataFrame:
    """Membersihkan spasi dan memetakan nama kolom umum ke standar sistem."""
    df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_")
    
    # Kamus alias cepat (menghemat pemanggilan API)
    mapping = {
        "date": ["tanggal", "waktu", "transaction_date", "waktu_transaksi", "tgl", "datetime"],
        "transaction_id": ["id_transaksi", "trans_id", "order_id", "invoice", "receipt_id", "id", "no_nota"],
        "total_amount": ["total", "amount", "grand_total", "total_pembayaran", "nilai_transaksi", "revenue", "omset"],
        "cashier_id": ["id_kasir", "user_id", "staff_id"],
        "cashier_name": ["nama_kasir", "kasir", "staff_name", "user_name", "operator"],
        "product_name": ["nama_produk", "item", "produk", "product", "barang"],
        "category": ["kategori", "jenis_produk", "department", "kategori_produk", "jenis"],
        "quantity": ["qty", "jumlah", "jml", "kuantitas"],
        "unit_price": ["harga", "harga_satuan", "price", "harga_jual"]
    }
    
    rename_dict = {}
    for standard_name, aliases in mapping.items():
        for col in df.columns:
            if col in aliases and standard_name not in df.columns:
                rename_dict[col] = standard_name
                
    return df.rename(columns=rename_dict)


def load_csv(file) -> Tuple[pd.DataFrame, list]:
    """Load CSV file and return DataFrame with validation messages."""
    messages = []
    try:
        df = pd.read_csv(file)
        
        # Lakukan pemetaan dasar terlebih dahulu
        df = standardize_columns_basic(df)
        
        messages.append(f"Berhasil memuat {len(df)} baris data.")
    except Exception as e:
        return pd.DataFrame(), [f"Gagal membaca dokumen: {e}"]

    # Pengecekan kolom wajib ditiadakan di sini, dipindah ke preprocess agar AI bisa bekerja dulu
    return df, messages


def preprocess(df: pd.DataFrame) -> pd.DataFrame:
    """Clean and enrich the dataframe, applying AI column mapping if needed."""
    df = df.copy()

    # 1. Cek apakah kolom wajib masih ada yang kurang
    missing = [col for col in REQUIRED_COLUMNS if col not in df.columns]
    
    # 2. Jika ada yang kurang, panggil modul pemetaan cerdas
    if missing and map_columns_with_ai is not None:
        groq_api_key = os.environ.get("GROQ_API_KEY", "")
        if groq_api_key:
            with st.spinner("Sistem sedang menganalisis dan menyesuaikan struktur kolom Anda..."):
                ai_mapping = map_columns_with_ai(list(df.columns), groq_api_key)
                if ai_mapping:
                    df = df.rename(columns=ai_mapping)
        else:
            # Tidak menghentikan proses, hanya memberitahu
            st.info("Kunci Sistem tidak terdeteksi. Sistem menggunakan pemetaan kolom standar.")

    # 3. Validasi Akhir: Jika setelah dibantu AI masih gagal, hentikan proses
    missing_after = [col for col in REQUIRED_COLUMNS if col not in df.columns]
    if missing_after:
        st.error(f"Sistem tidak dapat mengidentifikasi kolom data berikut: {', '.join(missing_after)}. Pastikan dokumen yang diunggah adalah rekap transaksi yang valid.")
        st.stop() # Menghentikan eksekusi Streamlit di titik ini agar tidak memicu error sistem (KeyError)

    # Parse date
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df["day_of_week"] = df["date"].dt.day_name()
    df["month"] = df["date"].dt.month_name()
    df["date_str"] = df["date"].dt.strftime("%d %b %Y")

    # Parse time if available
    if "time" in df.columns:
        df["hour"] = pd.to_datetime(
            df["time"].astype(str), format="%H:%M:%S", errors="coerce"
        ).dt.hour
        # Fill fallback for 5-char time (HH:MM)
        mask = df["hour"].isna()
        df.loc[mask, "hour"] = pd.to_datetime(
            df.loc[mask, "time"].astype(str), format="%H:%M", errors="coerce"
        ).dt.hour

    # Numeric coercion (Aman dari string seperti "Rp" atau ",")
    for col in ["total_amount", "quantity", "unit_price", "discount_percent"]:
        if col in df.columns:
            if df[col].dtype == object:
                # Membuang karakter selain angka dan titik desimal
                df[col] = df[col].astype(str).str.replace(r'[^\d.]', '', regex=True)
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)

    # Boolean void flag
    if "is_void" in df.columns:
        df["is_void"] = df["is_void"].astype(str).str.strip().str.lower().isin(["yes", "true", "1"])

    return df


def get_summary_stats(df: pd.DataFrame) -> dict:
    """Return high-level summary statistics."""
    total_revenue = df["total_amount"].sum()
    avg_transaction = df["total_amount"].mean()
    num_cashiers = df["cashier_id"].nunique() if "cashier_id" in df.columns else 0
    num_products = df["product_name"].nunique() if "product_name" in df.columns else 0

    stats = {
        "total_transactions": len(df),
        "total_revenue": total_revenue,
        "avg_transaction": avg_transaction,
        "max_transaction": df["total_amount"].max(),
        "min_transaction": df["total_amount"].min(),
        "num_cashiers": num_cashiers,
        "num_products": num_products,
        "date_range": (
            df["date"].min().strftime("%d %b %Y") if not df["date"].isna().all() else "N/A",
            df["date"].max().strftime("%d %b %Y") if not df["date"].isna().all() else "N/A",
        ),
    }
    return stats