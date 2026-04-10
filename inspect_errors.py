#!/usr/bin/env python
"""Inspect robot_logs_error table for error codes"""

import psycopg2
import json
import sys

sys.path.insert(0, 'src')
from config import DATABASE_CONFIG

print("="*80)
print("ROBOT_LOGS_ERROR TABLE INSPECTION")
print("="*80)

try:
    conn = psycopg2.connect(
        host=DATABASE_CONFIG['host'],
        port=DATABASE_CONFIG['port'],
        database=DATABASE_CONFIG['database'],
        user=DATABASE_CONFIG['user'],
        password=DATABASE_CONFIG['password'],
        sslmode=DATABASE_CONFIG.get('ssl_mode', 'disable')
    )
    
    cursor = conn.cursor()
    
    # Get sample error records
    print("\nSample Error Records:\n")
    cursor.execute("SELECT * FROM robot_logs_error LIMIT 5")
    
    cols = [desc[0] for desc in cursor.description]
    rows = cursor.fetchall()
    
    print(f"Columns: {cols}\n")
    
    for row_num, row in enumerate(rows, 1):
        print(f"Row {row_num}:")
        for col, val in zip(cols, row):
            if isinstance(val, (dict, list)):
                val_str = str(val)[:100]
            else:
                val_str = str(val)[:80]
            print(f"  {col}: {val_str}")
        print()
    
    # Get statistics
    cursor.execute("SELECT COUNT(*) FROM robot_logs_error")
    count = cursor.fetchone()[0]
    print(f"\nTotal error records: {count}")
    
    # Check for error_code or error_type columns
    cursor.execute("""
        SELECT column_name 
        FROM information_schema.columns
        WHERE table_name = 'robot_logs_error'
        ORDER BY ordinal_position
    """)
    
    print("\nAll columns in robot_logs_error:")
    for col in cursor.fetchall():
        print(f"  - {col[0]}")
    
    cursor.close()
    conn.close()
    
    print("\n" + "="*80)
    print("Soruları cevapla:")
    print("  1. Hata kodları hangi sütunda?")
    print("  2. 'Yerinde destek gerekli' yazısı nerede?")
    print("="*80)
    
except Exception as e:
    print(f"❌ Error: {e}")
    sys.exit(1)
