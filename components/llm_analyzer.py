"""
llm_analyzer.py
Calls Groq (Llama) API to generate intelligent summaries and analysis
of POS transaction anomalies.
"""

import os
import json
from typing import Optional

try:
    from groq import Groq
    GROQ_AVAILABLE = True
except ImportError:
    GROQ_AVAILABLE = False


SYSTEM_PROMPT = """Kamu adalah konsultan bisnis ritel dan pakar sistem POS. Tugasmu menganalisis data transaksi toko dan memberikan insight bisnis yang natural, profesional, dan langsung pada intinya.

ATURAN GAYA BAHASA:
- DILARANG KERAS menggunakan emoji dalam bentuk apa pun.
- Jangan terlalu detail, panjang lebar, atau bertele-tele. Buat penjelasan yang padat dan langsung ke poin utama.
- Gunakan Bahasa Indonesia yang luwes dan mengalir seperti manusia (hindari bahasa robotik atau pengulangan kata seperti "potensi penyebab adalah").
- Gunakan logika lapangan saat membahas anomali (misalnya: salah ketik nominal diskon, kasir lupa logout shift, atau pelanggan memborong barang).

FORMAT RESPONS:
1. Ringkasan Eksekutif
Buat 1-2 paragraf singkat yang merangkum performa omzet dan kesehatan transaksi toko. Jangan gunakan format daftar (bullet points) di bagian ini, buat mengalir seperti narasi.

2. Sorotan Anomali Lapangan
Jelaskan secara singkat apa saja keanehan yang terjadi di data kasir dan kemungkinan kejadian aslinya di lapangan. Fokus pada anomali dengan risiko tertinggi tanpa perlu merinci semua data satu per satu.

3. Rencana Tindakan (Action Plan)
Jangan gunakan format daftar panjang biasa. Susun rekomendasi ke dalam kerangka waktu operasional berikut:
- Harian (Daily): Tindakan yang harus dilakukan setiap tutup shift atau akhir hari.
- Mingguan (Weekly): Pengecekan sistem atau evaluasi kasir yang dilakukan tiap minggu.
- Bulanan (Monthly): Kebijakan atau audit menyeluruh untuk mencegah anomali berulang.

4. Fokus Investigasi
Sebutkan ID transaksi spesifik atau area yang harus segera diinvestigasi oleh pemilik toko hari ini juga sebagai penutup."""


def _build_prompt(summary_stats: dict, anomaly_summary: dict, sample_anomalies: list) -> str:
    """Construct a detailed analysis prompt."""
    anomaly_list = ""
    for i, a in enumerate(sample_anomalies[:10], 1):
        anomaly_list += (
            f"\n  {i}. [{a.get('anomaly_severity', '')}] {a.get('transaction_id', '')} – "
            f"{a.get('anomaly_types', '')} | "
            f"Rp{a.get('total_amount', 0):,.0f} | "
            f"{a.get('anomaly_details', '')[:100]}"
        )

    return f"""
Analisis data transaksi POS berikut dan berikan laporan komprehensif:

## STATISTIK UMUM
- Total Transaksi: {summary_stats.get('total_transactions', 0):,}
- Total Pendapatan: Rp {summary_stats.get('total_revenue', 0):,.0f}
- Rata-rata Transaksi: Rp {summary_stats.get('avg_transaction', 0):,.0f}
- Transaksi Tertinggi: Rp {summary_stats.get('max_transaction', 0):,.0f}
- Periode: {summary_stats.get('date_range', ('?', '?'))[0]} s/d {summary_stats.get('date_range', ('?', '?'))[1]}
- Jumlah Kasir: {summary_stats.get('num_cashiers', 0)}

## RINGKASAN ANOMALI
- Total Anomali Terdeteksi: {anomaly_summary.get('total_anomalies', 0)} transaksi ({anomaly_summary.get('anomaly_rate', 0):.1f}%)
- Distribusi Keparahan:
  * Critical: {anomaly_summary.get('critical', 0)} transaksi
  * High: {anomaly_summary.get('high', 0)} transaksi
  * Medium: {anomaly_summary.get('medium', 0)} transaksi
  * Low: {anomaly_summary.get('low', 0)} transaksi
- Jenis Anomali Terbanyak: {anomaly_summary.get('top_type', 'N/A')}

## CONTOH TRANSAKSI ANOMALI{anomaly_list}

Berikan analisis mendalam mencakup:
1. Ringkasan kondisi bisnis secara keseluruhan
2. Penilaian risiko dari anomali yang ditemukan
3. Potensi penyebab setiap jenis anomali
4. Rekomendasi tindakan konkret untuk manajemen toko
5. Hal-hal yang perlu diinvestigasi lebih lanjut

Tulis dalam format yang rapi dan mudah dibaca dengan emoji yang relevan.
"""


def generate_analysis(
    summary_stats: dict,
    anomaly_summary: dict,
    sample_anomalies: list,
    api_key: str,
    model: str = "llama-3.3-70b-versatile",
) -> str:
    """
    Generate AI analysis using Groq API.
    Returns analysis text or error message.
    """
    if not GROQ_AVAILABLE:
        return "❌ Library `groq` tidak terinstall. Jalankan: `pip install groq`"

    if not api_key or api_key.strip() == "":
        return "⚠️ API Key Groq belum dimasukkan. Masukkan API key di sidebar untuk mendapatkan analisis AI."

    try:
        client = Groq(api_key=api_key.strip())
        prompt = _build_prompt(summary_stats, anomaly_summary, sample_anomalies)

        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
            temperature=0.4,
            max_tokens=2048,
        )
        return response.choices[0].message.content

    except Exception as e:
        err = str(e)
        if "401" in err or "invalid_api_key" in err.lower():
            return "API Key tidak valid."
        if "rate_limit" in err.lower():
            return "Rate limit tercapai."
        return f"❌ Error saat menghubungi Groq API: {err}"


def get_quick_insight(transaction_row: dict, api_key: str, model: str = "llama-3.3-70b-versatile") -> str:
    """Get a quick one-sentence insight for a single anomalous transaction."""
    if not GROQ_AVAILABLE or not api_key:
        return ""

    prompt = (
        f"Transaksi anomali: {json.dumps(transaction_row, ensure_ascii=False, default=str)}. "
        "Jelaskan dalam 1-2 kalimat singkat bahasa Indonesia: apa yang mencurigakan dan apa kemungkinan penyebabnya."
    )
    try:
        client = Groq(api_key=api_key.strip())
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1,
            max_tokens=150,
        )
        return response.choices[0].message.content.strip()
    except Exception:
        return ""


AVAILABLE_MODELS = [
    "llama-3.3-70b-versatile",  
    "llama-3.1-8b-instant",     
    "openai/gpt-oss-120b",      
    "groq/compound"
]
