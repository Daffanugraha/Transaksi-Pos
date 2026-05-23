<div align="center">

# 🔍 POS Anomaly Detector

**Sistem Deteksi Transaksi Abnormal Bertenaga AI untuk Data Point of Sale**

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.9%2B-3776AB?logo=python&logoColor=white" />
  <img src="https://img.shields.io/badge/Framework-Streamlit-FF4B4B?logo=streamlit&logoColor=white" />
  <img src="https://img.shields.io/badge/AI_Engine-Llama_3.3_70B-F54703?logo=meta&logoColor=white" />
  <img src="https://img.shields.io/badge/API-Groq_(Free)-00A67E?logo=groq&logoColor=white" />
  <img src="https://img.shields.io/badge/Visualization-Plotly-3F4F75?logo=plotly&logoColor=white" />
  <img src="https://img.shields.io/badge/Stats-Z--Score_%2B_Rule--Based-9B59B6" />
  <img src="https://img.shields.io/badge/License-MIT-22C55E" />
  <img src="https://img.shields.io/badge/Deploy-Streamlit_Cloud-FF4B4B?logo=streamlit&logoColor=white" />
</p>

> *"Data transaksimu menyimpan cerita — biarkan AI yang membacanya."*  
> Upload CSV → Deteksi Otomatis → Laporan Eksekutif dari Llama 3.3 dalam hitungan detik.

</div>

---

## 📋 Daftar Isi

- [Gambaran Umum](#-gambaran-umum)
- [Spesifikasi Teknis](#-spesifikasi-teknis)
- [Arsitektur & Alur Kerja](#-arsitektur--alur-kerja)
- [Pipeline Pengumpulan & Pemrosesan Data](#-pipeline-pengumpulan--pemrosesan-data)
- [Mekanisme Chunking untuk LLM](#-mekanisme-chunking-untuk-llm)
- [8 Jenis Deteksi Anomali](#-8-jenis-deteksi-anomali)
- [Tech Stack](#-tech-stack)
- [Instalasi & Setup](#-instalasi--setup)
- [Format CSV yang Didukung](#-format-csv-yang-didukung)
- [Model AI Tersedia](#-model-ai-tersedia-via-groq)
- [Konfigurasi Parameter](#-konfigurasi-parameter)
- [Struktur Proyek](#-struktur-proyek)
- [Deploy ke Streamlit Cloud](#-deploy-ke-streamlit-cloud)
- [Kontribusi](#-kontribusi)

---

## 🌟 Gambaran Umum

**POS Anomaly Detector** adalah aplikasi analitik berbasis web yang dirancang untuk membantu pemilik toko, manajer operasional, dan tim audit mendeteksi transaksi tidak wajar secara otomatis menggunakan kombinasi analisis statistik dan kecerdasan buatan.

Sistem ini mengolah data ekspor CSV dari mesin kasir (POS) apapun — mulai dari Moka, Majoo, iSeller, hingga sistem custom — lalu menjalankan 8 layer deteksi anomali secara paralel, memvisualisasikan hasilnya dalam dashboard interaktif, dan menghasilkan laporan eksekutif dalam Bahasa Indonesia menggunakan **Llama 3.3-70B** via **Groq API (GRATIS)**.

### Masalah yang Diselesaikan

```
❌ Tanpa Sistem Ini              ✅ Dengan POS Anomaly Detector
─────────────────────────────    ──────────────────────────────────────────
Ribuan baris data → bingung      Upload CSV → anomali terdeteksi otomatis
Review manual = jam/hari         Analisis selesai < 30 detik
Fraud terdeteksi terlambat       Real-time flagging dengan severity level
Laporan audit manual             Laporan AI siap download 1 klik
Tidak tahu pola kasir curang     Visual cashier performance per orang
```

---

## ⚙️ Spesifikasi Teknis

### Persyaratan Minimum

| Komponen | Minimum | Rekomendasi |
|----------|---------|-------------|
| **Python** | 3.9 | 3.11+ |
| **RAM** | 512 MB | 2 GB |
| **Storage** | 100 MB | 500 MB |
| **OS** | Windows / macOS / Linux | Linux/macOS |
| **Browser** | Chrome 90+ / Firefox 88+ | Chrome terbaru |
| **Internet** | Untuk Groq API | Koneksi stabil |

### Batas Data yang Diproses

| Ukuran CSV | Baris Transaksi | Estimasi Waktu Proses | Chunking LLM |
|------------|-----------------|----------------------|--------------|
| < 1 MB | ~5.000 baris | < 2 detik | 1 chunk |
| 1–10 MB | ~50.000 baris | 2–8 detik | 3–5 chunks |
| 10–50 MB | ~250.000 baris | 8–30 detik | 10+ chunks |
| > 50 MB | 500.000+ baris | > 30 detik | Pakai sampling |

### Dependensi Inti

```
streamlit>=1.35.0    → Framework UI web interaktif
pandas>=2.0.0        → Manipulasi & analisis data tabular
numpy>=1.26.0        → Komputasi statistik (Z-score, IQR)
plotly>=5.20.0       → Visualisasi chart interaktif (7 chart)
groq>=0.9.0          → Klien resmi Groq API (akses Llama 3.3)
python-dotenv>=1.0.0 → Manajemen environment variables
```

---

## 🏗️ Arsitektur & Alur Kerja

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                          POS ANOMALY DETECTOR                               │
│                          Arsitektur Sistem                                  │
└─────────────────────────────────────────────────────────────────────────────┘

  INPUT                 PROCESSING LAYER              OUTPUT
  ─────                 ────────────────              ──────

  ┌──────────┐         ┌──────────────────┐          ┌────────────────────┐
  │  CSV     │ ──────▶ │  Data Processor  │ ────────▶│  Dashboard         │
  │  Upload  │         │  (Validasi +     │          │  (7 Chart Plotly)  │
  └──────────┘         │   Preprocessing) │          └────────────────────┘
       │               └────────┬─────────┘                   │
       │                        │                              │
  ┌──────────┐         ┌────────▼─────────┐          ┌────────────────────┐
  │  Demo    │         │  Anomaly         │ ────────▶│  Tabel Anomali     │
  │  Data    │         │  Detector        │          │  (Filter + Export) │
  └──────────┘         │  (8 Rules)       │          └────────────────────┘
                       └────────┬─────────┘                   │
                                │                              │
                       ┌────────▼─────────┐          ┌────────────────────┐
                       │  Chunker &        │ ────────▶│  Laporan AI        │
                       │  LLM Analyzer    │          │  (Bahasa Indonesia) │
                       │  (Groq + Llama)  │          └────────────────────┘
                       └──────────────────┘

  ┌─────────────────────────────────────────────────────────────┐
  │  KOMPONEN FILES                                             │
  │                                                             │
  │  app.py              → Entry point + UI orchestration       │
  │  components/                                                │
  │    data_processor.py → load_csv(), preprocess(), stats()   │
  │    anomaly_detector.py → detect_anomalies() 8 rules        │
  │    llm_analyzer.py   → chunking + Groq API call            │
  │    visualizer.py     → 7 fungsi chart Plotly               │
  │  assets/             → CSS + HTML templates                 │
  │  data/               → dummy_transactions.csv (100 baris)  │
  └─────────────────────────────────────────────────────────────┘
```

### Alur Eksekusi Lengkap

```
① User upload CSV / klik "Data Sampel"
         │
         ▼
② load_csv() — baca file, validasi kolom wajib, report warning
         │
         ▼
③ preprocess() — parsing tanggal/waktu, normalisasi tipe data,
                 tambah kolom derivasi (hour, date_str, dll)
         │
         ▼
④ detect_anomalies() — jalankan 8 rule secara berurutan:
   [Z-score] → [Tengah Malam] → [Diskon] → [Kuantitas]
   → [Void] → [Kasir Void] → [Nilai Nol] → [Produk Unknown]
   Setiap baris mendapat: is_anomaly, anomaly_score,
   anomaly_severity, anomaly_types, anomaly_details
         │
         ▼
⑤ get_summary_stats() — hitung agregat KPI (revenue, rata-rata, dll)
         │
         ├──────────────────────────────────┐
         ▼                                  ▼
⑥ Render 4 Tab UI:               ⑦ Opsional: LLM Analysis
   • Ringkasan (7 charts)            chunking anomaly data
   • Catatan Anomali                 → Groq API (Llama 3.3)
   • Data Transaksi                  → render laporan AI
   • Analisis Mendalam
```

---

## 📊 Pipeline Pengumpulan & Pemrosesan Data

### Sumber Data yang Didukung

Sistem dirancang untuk menerima ekspor CSV dari berbagai platform POS:

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Moka POS      │    │   Majoo POS     │    │   iSeller       │
│   Export CSV    │    │   Export CSV    │    │   Export CSV    │
└────────┬────────┘    └────────┬────────┘    └────────┬────────┘
         │                      │                       │
         └──────────────────────┼───────────────────────┘
                                │
                    ┌───────────▼────────────┐
                    │  POS Anomaly Detector  │
                    │  (Column Auto-Mapping) │
                    └────────────────────────┘
```

### Tahapan Preprocessing Data

```python
# Tahap 1: VALIDASI KOLOM
# Cek apakah 5 kolom wajib ada (transaction_id, date,
# total_amount, cashier_id, cashier_name)

# Tahap 2: PARSING TANGGAL & WAKTU
# Otomatis mendeteksi format: YYYY-MM-DD, DD/MM/YYYY, dll
# Ekstrak: hour, day_of_week, month, is_weekend

# Tahap 3: NORMALISASI TIPE DATA
# total_amount, unit_price, quantity → numeric (coerce invalid)
# is_void → boolean
# discount_percent → float 0–100

# Tahap 4: FEATURE ENGINEERING
# date_str       → format display tanggal
# hour           → jam transaksi (0–23)
# is_late_night  → boolean berdasarkan threshold slider
# amount_zscore  → Z-score nilai transaksi per store
```

### Statistik Ringkasan yang Dihitung

| Metrik | Cara Hitung | Digunakan Untuk |
|--------|-------------|-----------------|
| `total_revenue` | `sum(total_amount)` | KPI card utama |
| `avg_transaction` | `mean(total_amount)` | Baseline normal |
| `std_transaction` | `std(total_amount)` | Z-score threshold |
| `anomaly_rate` | `anomaly_count / total * 100` | Risk indicator |
| `cashier_void_rate` | `void_count / total per kasir` | Cashier profiling |
| `peak_hour` | `mode(hour)` | Operational insight |

---

## 🔄 Mekanisme Chunking untuk LLM

Salah satu tantangan terbesar mengirim data transaksi ke LLM adalah **batas token context window**. Groq Llama 3.3-70B mendukung hingga 128K token, tetapi dataset POS bisa memiliki ribuan baris anomali.

### Strategi Chunking yang Diimplementasikan

```
DATA ANOMALI (bisa ribuan baris)
           │
           ▼
┌──────────────────────────────────────────┐
│           CHUNKING STRATEGY              │
│                                          │
│  1. PRIORITIZATION                       │
│     Sort by anomaly_score DESC           │
│     → Ambil TOP-N anomali terpenting     │
│                                          │
│  2. SUMMARIZATION (Statistical Chunk)   │
│     → total_anomalies, anomaly_rate      │
│     → distribusi severity (C/H/M/L)     │
│     → top anomaly type                  │
│     → date range, cashier stats         │
│                                          │
│  3. SAMPLE CHUNK                         │
│     → 10 baris anomali terburuk          │
│     → field: id, date, cashier,          │
│       amount, type, details             │
└──────────────────────────────────────────┘
           │
           ▼
┌──────────────────────────────────────────┐
│        PROMPT ASSEMBLY                   │
│                                          │
│  [SYSTEM PROMPT]                         │
│  "Kamu adalah analis risiko operasional  │
│   POS berpengalaman..."                  │
│                                          │
│  [STATISTICAL CHUNK]                     │
│  Ringkasan agregat anomali               │
│                                          │
│  [SAMPLE CHUNK]                          │
│  10 kasus anomali terburuk               │
│                                          │
│  [INSTRUCTION]                           │
│  "Buat laporan eksekutif dalam           │
│   Bahasa Indonesia mencakup:             │
│   temuan utama, pola mencurigakan,       │
│   rekomendasi tindakan..."               │
└──────────────────────────────────────────┘
           │
           ▼
    Groq API → Llama 3.3-70B
           │
           ▼
    Laporan Eksekutif (Markdown)
```

### Mengapa Chunking Penting?

```
❌ Tanpa Chunking (kirim semua data mentah):
   • 10.000 transaksi ≈ ~500.000 token
   • Melebihi context window
   • Biaya API tinggi
   • Respons lambat & tidak fokus

✅ Dengan Chunking (statistik + sampel):
   • Fixed size ~2.000–3.000 token
   • Selalu dalam batas context window
   • Respons cepat (< 5 detik via Groq)
   • Lebih fokus pada temuan penting
```

---

## 🔍 8 Jenis Deteksi Anomali

```
┌────┬──────────────────────────┬──────────────────┬──────────┬────────────────┐
│ No │ Jenis Anomali            │ Metode           │ Severity │ Contoh Kasus   │
├────┼──────────────────────────┼──────────────────┼──────────┼────────────────┤
│ 1  │ Jumlah Tidak Wajar       │ Z-score > 2.5σ   │ Critical │ Transaksi Rp   │
│    │                          │ (configurable)   │ / High   │ 50jt saat avg  │
│    │                          │                  │          │ Rp 200rb       │
├────┼──────────────────────────┼──────────────────┼──────────┼────────────────┤
│ 2  │ Transaksi Tengah Malam   │ Rule: jam 23:00  │ High     │ Transaksi jam  │
│    │                          │ – 06:00          │          │ 02:34 pagi     │
├────┼──────────────────────────┼──────────────────┼──────────┼────────────────┤
│ 3  │ Diskon Berlebihan        │ Threshold > 40%  │ Medium   │ Diskon 75%     │
│    │                          │ (configurable)   │          │ tanpa otorisasi│
├────┼──────────────────────────┼──────────────────┼──────────┼────────────────┤
│ 4  │ Kuantitas Ekstrem        │ Threshold > 50   │ Medium   │ 200 unit barang│
│    │                          │ unit             │          │ dalam 1 struk  │
├────┼──────────────────────────┼──────────────────┼──────────┼────────────────┤
│ 5  │ Transaksi Void           │ Flag boolean     │ Low      │ is_void = True │
│    │                          │ is_void = True   │          │ Transaksi batal│
├────┼──────────────────────────┼──────────────────┼──────────┼────────────────┤
│ 6  │ Kasir Sering Void        │ Agregasi: void   │ Medium   │ Kasir A: 15    │
│    │                          │ per kasir > avg  │          │ void dalam 1   │
│    │                          │ + 2σ             │          │ hari           │
├────┼──────────────────────────┼──────────────────┼──────────┼────────────────┤
│ 7  │ Nilai Nol / Negatif      │ total_amount ≤ 0 │ Medium   │ Transaksi Rp 0 │
│    │                          │ tanpa is_void    │          │ bukan return   │
├────┼──────────────────────────┼──────────────────┼──────────┼────────────────┤
│ 8  │ Produk Tidak Dikenal     │ Category =       │ High     │ product_name = │
│    │                          │ "unknown/other"  │          │ "tes", "xxx",  │
│    │                          │ atau null        │          │ atau kosong    │
└────┴──────────────────────────┴──────────────────┴──────────┴────────────────┘
```

### Skema Severity Scoring

```
Setiap transaksi mendapat anomaly_score dari 0–100:

  0 ──────── 25 ──────── 50 ──────── 75 ──────── 100
  │  Normal  │    Low    │  Medium  │   High   │ Critical │
  │ (hijau)  │  (biru)   │ (kuning) │ (oranye) │  (merah) │

  Score dihitung: Σ(bobot × flag) untuk setiap rule
  Contoh: Z-score critical (40) + Tengah Malam (25) = Score 65 → HIGH
```

---

## 🛠️ Tech Stack

```
┌─────────────────────────────────────────────────────────────┐
│                     TECH STACK OVERVIEW                     │
├─────────────────┬───────────────────────────────────────────┤
│  LAYER          │  TEKNOLOGI                                │
├─────────────────┼───────────────────────────────────────────┤
│  Frontend UI    │  Streamlit 1.35+                          │
│                 │  CSS Custom (Dark Theme)                  │
│                 │  HTML Templates (Jinja-like)              │
├─────────────────┼───────────────────────────────────────────┤
│  Visualisasi    │  Plotly 5.20+ (7 chart interaktif)        │
│                 │  • Line chart (daily revenue)             │
│                 │  • Donut chart (severity dist)            │
│                 │  • Heatmap (hourly pattern)               │
│                 │  • Bar chart (anomaly types)              │
│                 │  • Scatter (cashier performance)          │
│                 │  • Treemap (category revenue)             │
│                 │  • Histogram (amount distribution)        │
├─────────────────┼───────────────────────────────────────────┤
│  Data Science   │  Pandas 2.0+ (data manipulation)         │
│                 │  NumPy 1.26+ (Z-score, statistics)        │
├─────────────────┼───────────────────────────────────────────┤
│  AI / LLM       │  Groq SDK 0.9+ (klien API)               │
│                 │  Llama 3.3-70B Versatile (model utama)    │
│                 │  Llama 3.1-8B Instant (model cepat)       │
│                 │  Mixtral 8x7B (model alternatif)          │
├─────────────────┼───────────────────────────────────────────┤
│  Config         │  python-dotenv (env management)           │
│                 │  .streamlit/config.toml (theme)           │
└─────────────────┴───────────────────────────────────────────┘
```

### Kenapa Groq, bukan OpenAI?

| Aspek | Groq (Llama 3.3) | OpenAI (GPT-4o) |
|-------|------------------|-----------------|
| **Biaya** | **GRATIS** (free tier) | ~$5/1M token |
| **Kecepatan** | ~800 token/detik | ~100 token/detik |
| **Bahasa Indonesia** | Baik | Sangat baik |
| **Context Window** | 128K token | 128K token |
| **Privacy** | Data tidak disimpan | Depends on plan |

---

## 🚀 Instalasi & Setup

### 1. Clone Repository

```bash
git clone https://github.com/Daffanugraha/Transaksi-Pos.git
cd Transaksi-Pos
```

### 2. Buat Virtual Environment (Rekomendasi)

```bash
# Windows
python -m venv venv
venv\Scripts\activate

# macOS / Linux
python3 -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Setup API Key Groq (Opsional tapi Sangat Direkomendasikan)

```bash
# Daftar gratis di https://console.groq.com
# Copy API key kamu (format: gsk_xxxxxxxxxxxx)

# Cara 1: Environment Variable (via .env file)
cp .env.example .env
# Edit .env:
# GROQ_API_KEY=gsk_your_key_here

# Cara 2: Input langsung di sidebar aplikasi (tidak perlu file)
# Paling mudah untuk testing cepat!
```

### 5. Jalankan Aplikasi

```bash
streamlit run app.py
```

Buka browser: **http://localhost:8501**

### Troubleshooting Umum

```bash
# Error: ModuleNotFoundError
pip install -r requirements.txt --upgrade

# Error: Port already in use
streamlit run app.py --server.port 8502

# Error: Groq API rate limit
# Ganti ke model 'llama-3.1-8b-instant' yang lebih ringan
```

---

## 📋 Format CSV yang Didukung

### Kolom Wajib (Minimal 5 Kolom Ini)

```csv
transaction_id,date,total_amount,cashier_id,cashier_name
TRX-001,2024-01-15,150000,CSH-01,Budi Santoso
TRX-002,2024-01-15,0,CSH-02,Sari Dewi
TRX-003,2024-01-16,9999999,CSH-01,Budi Santoso
```

| Kolom | Tipe | Contoh | Keterangan |
|-------|------|--------|-----------|
| `transaction_id` | string | `TRX-001` | ID unik per transaksi |
| `date` | date | `2024-01-15` | Format YYYY-MM-DD |
| `total_amount` | numeric | `150000` | Nilai transaksi (Rupiah) |
| `cashier_id` | string | `CSH-01` | ID kasir |
| `cashier_name` | string | `Budi Santoso` | Nama kasir |

### Kolom Opsional (Untuk Analisis Lebih Kaya)

| Kolom | Tipe | Deteksi Anomali Yang Aktif |
|-------|------|---------------------------|
| `time` | HH:MM:SS | Transaksi Tengah Malam (#2) |
| `discount_percent` | float 0-100 | Diskon Berlebihan (#3) |
| `quantity` | integer | Kuantitas Ekstrem (#4) |
| `is_void` | bool/0/1 | Transaksi Void (#5 & #6) |
| `product_name` | string | Produk Tidak Dikenal (#8) |
| `category` | string | Produk Tidak Dikenal (#8) |
| `unit_price` | numeric | Validasi konsistensi harga |
| `payment_method` | string | Segmentasi laporan |

---

## 🤖 Model AI Tersedia (via Groq)

| Model | Kecepatan | Kualitas | Best For |
|-------|-----------|----------|----------|
| `llama-3.3-70b-versatile` ⭐ | ⚡⚡⚡ | ★★★★★ | Laporan lengkap & analisis mendalam |
| `llama-3.1-8b-instant` | ⚡⚡⚡⚡⚡ | ★★★ | Preview cepat, testing, low rate limit |
| `mixtral-8x7b-32768` | ⚡⚡⚡⚡ | ★★★★ | Analisis terstruktur, format tabel rapi |
| `gemma2-9b-it` | ⚡⚡⚡⚡ | ★★★ | Ringkasan singkat, beban API rendah |

---

## 🔧 Konfigurasi Parameter

Semua parameter dapat diubah **realtime** di sidebar tanpa restart aplikasi:

```
┌─────────────────────────────────────────────────────┐
│                    SIDEBAR CONTROLS                  │
│                                                      │
│  ┌─────────────────────────────────────────────┐    │
│  │ Pilihan Mesin Analisis                      │    │
│  │ [llama-3.3-70b-versatile ▼]                │    │
│  └─────────────────────────────────────────────┘    │
│                                                      │
│  ─── Kriteria Deteksi ───                           │
│                                                      │
│  Batas Toleransi Z-Score                            │
│  1.5 ──●────────────── 4.0  (default: 2.5)         │
│       ↑ Makin kecil = makin sensitif                │
│                                                      │
│  Batas Wajar Diskon (%)                             │
│  10% ────●──────────── 80%  (default: 40%)          │
│                                                      │
│  Batas Wajar Jumlah Barang                          │
│  10 ─────────●──────── 200  (default: 50 unit)      │
│                                                      │
│  Awal Jam Rawan (Malam)                             │
│  20:00 ──────────●──── 23:00  (default: 23:00)      │
│                                                      │
│  Akhir Jam Rawan (Pagi)                             │
│  4:00 ─●────────────── 8:00   (default: 6:00)       │
└─────────────────────────────────────────────────────┘
```

---

## 📂 Struktur Proyek

```
Transaksi-Pos/
│
├── 📄 app.py                    # Entry point — UI orchestration & tab routing
├── 📄 requirements.txt          # Python dependencies
├── 📄 .gitignore
│
├── 📁 .streamlit/
│   └── config.toml              # Dark theme configuration
│
├── 📁 components/               # Business logic (MVC-style separation)
│   ├── __init__.py
│   ├── data_processor.py        # load_csv(), preprocess(), get_summary_stats()
│   ├── anomaly_detector.py      # detect_anomalies() — 8 rules engine
│   ├── llm_analyzer.py          # Chunking + Groq API integration
│   └── visualizer.py            # 7 fungsi chart_*() dengan Plotly
│
├── 📁 assets/
│   ├── 📁 styles/
│   │   └── main.css             # Dark theme + badge + card styles
│   └── 📁 templates/
│       ├── header.html
│       ├── kpi_grid.html
│       ├── ai_header.html
│       ├── anomaly_table_row.html
│       ├── anomaly_table_layout.html
│       ├── severity_row.html
│       └── landing_page.html
│
└── 📁 data/
    └── dummy_transactions.csv   # 100 baris data demo (sudah ada anomali)
```

---

## ☁️ Deploy ke Streamlit Cloud

```bash
# 1. Push ke GitHub
git add .
git commit -m "ready to deploy"
git push origin main

# 2. Buka https://share.streamlit.io
# 3. Login dengan akun GitHub → New app
# 4. Pilih repo: Daffanugraha/Transaksi-Pos, file: app.py
# 5. Klik Deploy!
```

Tambahkan Groq API Key sebagai Secret di Streamlit Cloud → **Settings** → **Secrets**:

```toml
GROQ_API_KEY = "gsk_your_key_here"
```

> ⚠️ Jangan pernah commit API key ke file `.env` yang di-push ke repository publik!

---

## 🤝 Kontribusi

Pull request sangat disambut! Beberapa area yang bisa dikembangkan:

```
🔧 Ide Pengembangan
───────────────────
□  Tambah rule deteksi: pola transaksi berulang dalam interval pendek
□  Support format Excel (.xlsx) selain CSV
□  Multi-store comparison dashboard
□  Scheduler: auto-scan CSV dari folder tertentu (watchdog)
□  Export laporan ke PDF format
□  Tambah model: Claude Haiku via Anthropic API
□  Integrasi langsung dengan API Moka / Majoo
□  Notifikasi email / Telegram saat anomali critical terdeteksi
```

```bash
# Fork → buat branch → commit → pull request
git checkout -b feature/nama-fitur-kamu
git commit -m "feat: deskripsi singkat"
git push origin feature/nama-fitur-kamu
```

---

## 📄 License

MIT License — Bebas digunakan dan dimodifikasi. Lihat file [LICENSE](LICENSE) untuk detail.

---

<div align="center">
*"Deteksi fraud bukan lagi privilege enterprise besar — ini untuk semua pelaku usaha."*
</div>