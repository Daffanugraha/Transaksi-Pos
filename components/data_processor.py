"""
data_processor.py
Handles loading, validating, and preprocessing POS transaction data.
"""

import pandas as pd
import numpy as np
from datetime import datetime
from typing import Tuple


REQUIRED_COLUMNS = [
    "transaction_id", "date", "total_amount", "cashier_id", "cashier_name"
]

OPTIONAL_COLUMNS = [
    "time", "product_name", "category", "quantity", "unit_price",
    "payment_method", "store_id", "customer_id", "discount_percent",
    "is_void", "product_id"
]


def load_csv(file) -> Tuple[pd.DataFrame, list]:
    """Load CSV file and return DataFrame with validation messages."""
    messages = []
    try:
        df = pd.read_csv(file)
        messages.append(f"✅ Berhasil memuat {len(df)} transaksi.")
    except Exception as e:
        return pd.DataFrame(), [f"❌ Gagal membaca file: {e}"]

    # Normalize column names
    df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_")

    # Check required columns
    missing = [col for col in REQUIRED_COLUMNS if col not in df.columns]
    if missing:
        return df, [f"⚠️ Kolom wajib tidak ditemukan: {missing}"]

    return df, messages


def preprocess(df: pd.DataFrame) -> pd.DataFrame:
    """Clean and enrich the dataframe."""
    df = df.copy()

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

    # Numeric coercion
    for col in ["total_amount", "quantity", "unit_price", "discount_percent"]:
        if col in df.columns:
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
