import pandas as pd
import json
from groq import Groq
import os
import streamlit as st

def map_columns_with_ai(df_columns: list, api_key: str, model="llama-3.3-70b-versatile") -> dict:
    """
    Menggunakan LLM untuk memetakan kolom CSV acak ke format standar sistem.
    """
    if not api_key:
        return {} # Fallback jika belum memasukkan API Key
        
    client = Groq(api_key=api_key)
    
    # Standar kolom yang dibutuhkan sistem
    standard_columns = {
        "date": "Waktu/Tanggal terjadinya transaksi",
        "transaction_id": "ID, nomor unik, atau resi transaksi",
        "total_amount": "Total nilai atau harga dari seluruh transaksi",
        "cashier_name": "Nama kasir, staf, atau operator",
        "cashier_id": "ID kasir atau staf",
        "product_name": "Nama barang atau produk yang dijual",
        "category": "Kategori atau jenis produk",
        "quantity": "Jumlah atau kuantitas barang",
        "unit_price": "Harga satuan barang"
    }

    prompt = f"""
    Kamu adalah sistem pemetaan data otomatis (Data Mapper) untuk aplikasi Point of Sales.
    Tugasmu adalah mencocokkan nama kolom dari file CSV pengguna dengan nama kolom standar sistem.
    
    Kolom CSV Pengguna:
    {df_columns}
    
    Kolom Standar Sistem beserta deskripsinya:
    {json.dumps(standard_columns, indent=2)}
    
    ATURAN:
    1. Cocokkan Kolom CSV Pengguna ke Kolom Standar Sistem yang paling relevan.
    2. Jika ada kolom pengguna yang tidak relevan dengan standar, abaikan saja.
    3. Kembalikan HASIL HANYA dalam format JSON Murni (tanpa penjelasan, tanpa markdown ```json).
    
    Format output JSON yang wajib:
    {{
      "nama_kolom_csv_pengguna": "nama_kolom_standar_sistem",
      "nm_brg": "product_name",
      "wkt_trx": "date"
    }}
    """

    try:
        response = client.chat.completions.create(
            messages=[{"role": "user", "content": prompt}],
            model=model,
            temperature=0.1, # Suhu rendah agar hasilnya logis dan tidak berhalusinasi
            response_format={"type": "json_object"}
        )
        
        mapping_result = json.loads(response.choices[0].message.content)
        return mapping_result
    except Exception as e:
        print(f"Error AI Mapping: {e}")
        return {} # Fallback ke mapping kosong jika AI gagal