#!/usr/bin/env python
"""Inspect robot_logs tables and prepare training data"""

import psycopg2
import pandas as pd

try:
    print("🔌 PostgreSQL'e bağlanılıyor...")
    conn = psycopg2.connect(
        host='149.102.155.77',
        port=5433,
        database='robot_pipeline',
        user='robot_pipeline_admin',
        password='RobotPipe!2026#PG!149',
        sslmode='disable'
    )
    print("✅ Bağlantı başarılı\n")
    
    # Inspect robot_logs_info table
    print("="*60)
    print("📊 robot_logs_info Tablosu")
    print("="*60)
    
    df_info = pd.read_sql_query('SELECT * FROM robot_logs_info LIMIT 5', conn)
    print(f"\nSatır sayısı: {len(pd.read_sql_query('SELECT COUNT(*) as cnt FROM robot_logs_info', conn))}")
    print(f"Sütunlar: {list(df_info.columns)}")
    print(f"\nİlk 5 satır:")
    print(df_info.to_string())
    
    # Inspect robot_logs_error table
    print("\n" + "="*60)
    print("📊 robot_logs_error Tablosu")
    print("="*60)
    
    df_error = pd.read_sql_query('SELECT * FROM robot_logs_error LIMIT 5', conn)
    print(f"\nSatır sayısı: {len(pd.read_sql_query('SELECT COUNT(*) as cnt FROM robot_logs_error', conn))}")
    print(f"Sütunlar: {list(df_error.columns)}")
    print(f"\nİlk 5 satır:")
    print(df_error.to_string())
    
    # Get data types and info
    print("\n" + "="*60)
    print("📋 Detaylı Bilgi")
    print("="*60)
    
    cur = conn.cursor()
    
    for table in ['robot_logs_info', 'robot_logs_error']:
        print(f"\n{table}:")
        cur.execute(f"""
        SELECT column_name, data_type, is_nullable
        FROM information_schema.columns
        WHERE table_name='{table}'
        ORDER BY ordinal_position
        """)
        for col in cur.fetchall():
            print(f"  - {col[0]}: {col[1]} (nullable: {col[2]})")
    
    conn.close()
    print("\n✅ Bağlantı kapatıldı")
    
except Exception as e:
    print(f'❌ Hata: {str(e)}')
    import traceback
    traceback.print_exc()
