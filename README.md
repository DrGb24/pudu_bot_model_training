# PUDU Robot Prediktif Bakım Sistemi

[![Python 3.10+](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![TensorFlow 2.13+](https://img.shields.io/badge/TensorFlow-2.13+-orange.svg)](https://www.tensorflow.org/)
[![HuggingFace Dataset](https://img.shields.io/badge/Dataset-HuggingFace-yellow.svg)](https://huggingface.co/datasets/Lightcap/pudu-robot-operation-logs-bau-capstone-2026)

PUDU servis robotları için çok çıktılı LSTM tabanlı prediktif bakım sistemi.
**147.488 gerçek robot hata logu** ile eğitilmiş; anlık durum, arıza şiddeti, 7 günlük öngörü ve tahmini arıza süresi çıktıları üretir.

---

## Proje Yapısı

```
project/
├── lstm_train_v2.py          # Model eğitim pipeline'ı (10 adım)
├── lstm_inference_v2.py      # Tahmin motoru + robot raporu üretimi
├── HATA_KODLARI_ROBOT.xlsx   # Hata kodu → destek tipi / çözüm tablosu (119 kayıt)
├── requirements.txt
├── src/
│   ├── lstm_models_v2.py     # MultiOutputLSTMModel mimarisi
│   └── config.py             # Merkezi konfigürasyon (HuggingFace, LSTM_V2_CONFIG)
└── models/lstm_v2/           # Eğitilmiş model dosyaları (git'e dahil değil)
    ├── lstm_v2_weights.weights.h5
    ├── lstm_v2_config.json
    └── lstm_v2_scaler.pkl
```

---

## Model Mimarisi — LSTM V2

**Tip:** Keras Functional API — Çok çıktılı LSTM
**Parametre sayısı:** 132.775

| Çıktı (Head) | Görev | Tip |
|---|---|---|
| Head 1 | Anlık arıza tespiti | Binary sınıflandırma |
| Head 2 | Arıza şiddeti (Event/Warning/Error/Fatal) | 4-sınıflı sınıflandırma |
| Head 3 | 7 günlük arıza olasılığı | Regresyon [0–1] |
| Head 4 | Tahmini arıza süresi (0–168 saat) | Regresyon |

**Eğitim Verisi:**
- Kaynak: `Lightcap/pudu-robot-operation-logs-bau-capstone-2026` (HuggingFace)
- Config: `partitioned_error_logs` — **toplam 147.488 kayıt**, 46 robot

| Split | Satır | Kullanım |
|---|---|---|
| `train` | 103.241 | Model eğitimi |
| `validation` | 22.123 | Erken durdurma / hiperparametre |
| `test` | 22.124 | Final değerlendirme |
| **Toplam** | **147.488** | |

- Giriş: 10 adımlı zaman serisi, 9 özellik
- Bölünme: %80 eğitim / %20 test

**Eğitim Sonuçları:**

| Head | Metrik | Sonuç |
|---|---|---|
| Head 1 — Anlık arıza | Accuracy / F1 / AUC | %99.4 / %94.8 / %99.9 ✅ |
| Head 2 — Şiddet | Accuracy | %98.6 ✅ |
| Head 3 — 7 günlük öngörü | AUC | %80.5 |
| Head 4 — Arıza süresi | MAE | 19.7 saat |

---

## Kurulum

```bash
# Sanal ortam oluştur ve aktif et
python -m venv venv
venv\Scripts\activate          # Windows
source venv/bin/activate       # Linux/Mac

# Bağımlılıkları yükle
pip install -r requirements.txt
```

---

## Kullanım

### Model Eğitimi

```bash
python lstm_train_v2.py
```

HuggingFace'den veriyi çeker, 10 adımlı pipeline ile eğitir, `models/lstm_v2/` altına kaydeder.

### Robot Raporu Üretimi (Tüm Filo)

```bash
# Windows
set PYTHONIOENCODING=utf-8
python lstm_inference_v2.py
```

46 robot için T-1 tarihi baz alınarak Türkçe rapor üretilir.

### Python API

```python
from lstm_inference_v2 import LSTMInferenceV2
import pandas as pd

engine = LSTMInferenceV2()  # model + hata kodu tablosunu yükler

# Tek robot tahmini
result = engine.predict_for_robot(robot_id='8110K...', robot_df=robot_df)
print(result['risk_level'])       # YUKSEK / ORTA / DUSUK
print(result['error_details'])    # Excel'den destek tipi + çözüm

# İnsan okunabilir Türkçe rapor
report = engine.robot_report(robot_id='8110K...', robot_df=robot_df)
print(report)

# Tüm filo raporu
fleet = engine.fleet_report(robot_dfs={'id1': df1, 'id2': df2})
```

---

## Robot Raporu Formatı

```
============================================================
  ROBOT DURUM RAPORU — T-1: 2026-02-25
  Robot ID   : 8110K4529050001
  Genel Risk : YÜKSEK 🔴
============================================================

  [1] ANLÍK DURUM  (T-1 gününe ait son log)
  ├─ Durum          : BAKIM GEREKTİRİYOR ⚠️
  ├─ Arıza olasılığı: %88.4
  ├─ Aktif hatalar  : CanNotReach, WheelErrorLeft
  └─ Hata kategorisi: Navigasyon, Hareket

  [HATA DETAYLARI]
  │
  ├─ Hata Kodu   : CanNotReach
  │  Sınıf       : Lokasyon Kayıpları
  │  Destek Tipi : 🔧 Yerinde destek gerekli
  │  Çözüm       : Hardware hatası, yerinde kontrol gereklidir.
  │
  ├─ Hata Kodu   : WheelErrorLeft
  │  Sınıf       : Tekerlek anormalliği
  │  Destek Tipi : 🔧 Yerinde destek gerekli
  │  Çözüm       : Robotu remote terminal üzerinden kapatıp aç...
  │

  [2] ARIZA ŞİDDETİ
  ├─ Mevcut şiddet  : Hata (Error)
  └─ Dağılım        : Event: %0 | Warning: %1 | Error: %97 | Fatal: %2

  [3] BAKIM / TAMİR GEREKSİNİMİ
  ├─ 7 günlük arıza ihtim.: %88.4
  ├─ Maksimum pencere iht.: %98.6
  └─ YÜKSEK ihtimalle bakım gerekecek

  [4] TAHMİNİ ARIZA ZAMANI (T-1'den itibaren)
  └─ ⚠️  ~18 saat içinde hata bekleniyor
============================================================
```

### Durum Etiketleri

| Etiket | Koşul |
|---|---|
| `ARIZALI ⛔` | Fatal seviyeli hata **VEYA** Error + arıza olasılığı ≥ %80 |
| `BAKIM GEREKTİRİYOR ⚠️` | Error seviyeli hata **VEYA** arıza olasılığı ≥ %55 + aktif hata |
| `TAKİPTE 🔶` | Warning seviyeli hata **VEYA** arıza olasılığı ≥ %30 + aktif hata |
| `ÇALIŞIR DURUMDA ✓` | Yukarıdaki koşulların hiçbiri |

### Genel Risk Seviyeleri

| Seviye | Eşik |
|---|---|
| `YÜKSEK 🔴` | max(anlık olasılık, 0.7 × 7 günlük olasılık) ≥ %70 |
| `ORTA 🟡` | ≥ %40 |
| `DÜŞÜK 🟢` | < %40 |

---

## Hata Kodu Tablosu (HATA_KODLARI_ROBOT.xlsx)

119 hata kodu kayıtlıdır. Her kayıt için:

- **Arıza sınıflandırması** — Başlatma, Tekerlek, Navigasyon, Temizlik vb.
- **Destek tipi** — 🔧 Yerinde destek gerekli / 📞 Uzaktan destek
- **Çözüm metodu** — Teknisyen talimatı (Türkçe)

Raporda her aktif hata kodu için otomatik olarak ilgili destek tipi ve çözüm adımları gösterilir.

---

## T-1 Referans Tarihi Mantığı

Sistem, raporları **T-1** (bir gün önceki) baz alarak üretir:

```
T   = max(task_time in dataset)   → veri son tarihi
T-1 = T − 1 gün                  → analiz referans tarihi
```

Bu sayede son günün tamamlanmış log verisi üzerinden karar üretilir.

---

## Bağımlılıklar

```
tensorflow>=2.13
keras>=3.0
scikit-learn
pandas
numpy
joblib
datasets          # HuggingFace
openpyxl          # Excel okuma (.xlsx)
```

Tam liste: `requirements.txt`

