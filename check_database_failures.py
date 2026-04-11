#!/usr/bin/env python3
"""
Database başarısızlık verisi kontrol et
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / 'src'))

from data_preparation import DataPreparation
from config import DATABASE_CONFIG

print("📊 DATABASE BAŞARISIZLIK VERİSİ KONTROLÜ")
print("="*70)

# Database bağlantısı
data_prep = DataPreparation(DATABASE_CONFIG)

# 1. Toplam kayıt sayısı
query_total = "SELECT COUNT(*) as total FROM robot_logs_info"
df_total = data_prep.load_from_database(DATABASE_CONFIG, query=query_total)
total_count = df_total['total'].values[0]
print(f"\n✅ Toplam robot_logs_info kayıtları: {total_count:,}")

# 2. Başarısızlıklar (is_success = 0)
query_failures = "SELECT COUNT(*) as failures FROM robot_logs_info WHERE is_success = 0"
df_failures = data_prep.load_from_database(DATABASE_CONFIG, query=query_failures)
failure_count = df_failures['failures'].values[0]
print(f"✅ Başarısız işlemler (is_success=0): {failure_count:,}")

failure_rate = (failure_count / total_count * 100) if total_count > 0 else 0
print(f"✅ Başarısızlık oranı: {failure_rate:.2f}%")

# 3. Başarılılar (is_success = 1)
success_count = total_count - failure_count
print(f"✅ Başarılı işlemler (is_success=1): {success_count:,}")

# 4. Tablo hakkında bilgi
print(f"\n📋 Tablo Bilgileri:")
query_info = """
SELECT 
    (SELECT COUNT(*) FROM robot_logs_info) as total_rows,
    (SELECT COUNT(DISTINCT robot_id) FROM robot_logs_info) as unique_robots,
    (SELECT COUNT(DISTINCT DATE(task_time)) FROM robot_logs_info) as unique_dates
"""
df_info = data_prep.load_from_database(DATABASE_CONFIG, query=query_info)
print(f"   Toplam satır: {df_info['total_rows'].values[0]:,}")
print(f"   Benzersiz robotlar: {df_info['unique_robots'].values[0]}")
print(f"   Benzersiz tarihler: {df_info['unique_dates'].values[0]}")

# 5. Robot başına başarısızlık dağılımı
print(f"\n🤖 Robot başına başarısızlık dağılımı:")
query_by_robot = """
SELECT 
    robot_id,
    COUNT(*) as total,
    SUM(CASE WHEN is_success=0 THEN 1 ELSE 0 END) as failures,
    ROUND(SUM(CASE WHEN is_success=0 THEN 1 ELSE 0 END)::numeric / COUNT(*) * 100, 2) as failure_pct
FROM robot_logs_info
GROUP BY robot_id
ORDER BY failures DESC
LIMIT 20
"""
df_by_robot = data_prep.load_from_database(DATABASE_CONFIG, query=query_by_robot)
print(df_by_robot.to_string(index=False))

# 6. Tarih aralığı
print(f"\n📅 Zaman Aralığı:")
query_dates = """
SELECT 
    MIN(task_time) as min_date,
    MAX(task_time) as max_date,
    MAX(task_time)::date - MIN(task_time)::date as days_span
FROM robot_logs_info
"""
df_dates = data_prep.load_from_database(DATABASE_CONFIG, query=query_dates)
print(f"   En eski: {df_dates['min_date'].values[0]}")
print(f"   En yeni: {df_dates['max_date'].values[0]}")
print(f"   Süre: {df_dates['days_span'].values[0]} gün")

print(f"\n{'='*70}")
print("✅ KONTROL TAMAMLANDI")
print("="*70)
