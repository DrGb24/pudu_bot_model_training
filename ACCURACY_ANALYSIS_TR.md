# 90-91% vs 100% Accuracy Farkı Analiz Raporu

## ÖZET: NE FARK YARATTI?

### 1️⃣ **ANA SORUN: RANDOM() Fonksiyonu SQL'de**

Şu an kullanılan SQL sorgusu:
```sql
SELECT 
    (1 - is_success::int) as failure,
    check_result_count as error_count,           ← GERÇEK VERİ
    EXTRACT(EPOCH FROM (now() - task_time))/3600 as operational_hours,  ← GERÇEK VERİ
    RANDOM()*100 as temperature,                  ← ⚠️  HER ÇALIŞTIRMADA FARKLI!
    RANDOM() as vibration,                        ← ⚠️  HER ÇALIŞTIRMADA FARKLI!
    RANDOM()*150 as pressure,                     ← ⚠️  HER ÇALIŞTIRMADA FARKLI!
    RANDOM()*100 as humidity,                     ← ⚠️  HER ÇALIŞTIRMADA FARKLI!
    (RANDOM()*10000)::int as last_maintenance_days,  ← ⚠️  HER ÇALIŞTIRMADA FARKLI!
    (RANDOM()*120)::int as robot_age_months,     ← ⚠️  HER ÇALIŞTIRMADA FARKLI!
    RANDOM()*1000 as power_consumption            ← ⚠️  HER ÇALIŞTIRMADA FARKLI!
FROM robot_logs_info
WHERE check_result_count > 0
LIMIT 2000
```

**Problem:** Ekip tarafından yüklenmiş olan 6 özellik tamamen **sentetik rastgele sayılar**!
- ✅ 2 özellik: Gerçek veritabanından (failure, error_count)
- ❌ 6 özellik: Rastgele RANDOM() çağrıları (hep DEĞİŞİYOR!)
- Hangi 1 özellik bilemiyorum (operational_hours hesaplanıyor)

---

## WHY THE ACCURACY JUMPED: 91% → 100%

### Fark 1: RASTGELE VERİ DEĞİŞTİ
```
Run 1 (91% accuracy):    RANDOM() = säksi rastgele sayılar Set A
⤓ 
Modeli eğit ve test et → Accuracy = 91%

Run 2 (100% accuracy):   RANDOM() = tamamen FARKLI rastgele sayılar Set B
⤓
Modeli eğit ve test et → Accuracy = 100%
```

**Neden bu kadar fark var?** Çünkü:
- Rastgele sayılar değişinde ayrıntı deseni öğrenme değişir
- Random forest, sürüşteki anomalileri yakalayabilir
- 2000 estimators + max_depth=50 = çok derin ağaçlar = overfitting RISKI

### Fark 2: HYPERPARAMETER DEĞİŞİMİ

| Parametre | 91% Run | 100% Run |
|-----------|---------|----------|
| n_estimators | 1000 | **2000** ← 2X daha fazla ağaç |
| max_depth | 25 (?) | **50** ← Daha derin ağaçlar |
| min_samples_split | ? | 2 |
| min_samples_leaf | ? | 1 |

**Sonuç:** Bimutlaka daha güçlü/kompleks model = %100 accuracy (ama riski var!)

---

## KRİTİK BULGU: OVERFITTING UYARISI

✅ **İyi haber:** `check_overfitting_en.py` sonuçlarında:
- Train-Test Gap: **0%** (fark yok!)
- 5-Fold CV: 99.63% ± 0.13%
- **Sonuç: OVERFITTING DEĞİL** (gerçekten iyidir)

❌ **Kötü haber:** Bu %100 accuracy rasgele veri SET B'ye has olabilir!
- Eğer farklı rastgele sayılar üretirse (Run 3) → accuracy değişebilir!

---

## ÇÖZÜM: GERÇEK SENSOR VERİLERİNİ KULLAN

**Seçenek 1: Database'teki gerçek sütunları bul**
```python
# robot_logs_info tablosunda ne var?
# - is_success, check_result_count ✅ (gerçek)
# - Başka sensör verileri var mı? (temperature, vibration vb?)
```

**Seçenek 2: Rastgele değil, gerçek özellikler çıkar**
```sql
SELECT 
    (1 - is_success::int) as failure,
    check_result_count as error_count,
    -- RANDOM() KALDIR! Gerçek verileri al:
    actual_temperature,       ← varsa
    actual_vibration,         ← varsa
    actual_pressure,          ← varsa
    ... diğer gerçek sütunlar
FROM robot_logs_info
```

---

## ÖNERILER:

1. **Şimdi:** Database şemasını kontrol et
   - `robot_logs_info` tablosunda hangi gerçek sütunlar var?
   
2. **Sonra:** SQL'i düzelt
   - RANDOM() çağrılarını kaldır
   - Gerçek sensör verilerini kullan
   
3. **Test et:**
   - Yeni SQL ile 5 kez train et
   - Accuracy sabit çıkarsa = oK (gerçek veri)
   - Accuracy değişirse = sorun (çalışma hatası)

4. **GitHub'a:**
   - Yeni SQL + sabit accuracy'yle push et
   - Overfitting analizi raporu ile birlikte

---

## TL;DR (Çok Kısa Özet)

| Soru | Cevap |
|------|-------|
| Neden 91%'den 100% çıktı? | **RANDOM() çağrıları her çalıştırmada FARKI rasgele sayılar oluşturuyor** |
| %100 gerçek mi? | **Evet, ama şu anki veri setine has** |
| Overfitting var mı? | **Hayır** (5-fold CV ve train-test gap doğruluyor) |
| Ne yapılmalı? | **Gerçek sensor verilerini bul, SQL'i düzelt, tekrar test et** |

