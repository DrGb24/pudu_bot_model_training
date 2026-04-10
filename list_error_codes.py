#!/usr/bin/env python
"""Get all unique error types and details from robot_logs_error"""

import psycopg2
import sys

sys.path.insert(0, 'src')
from config import DATABASE_CONFIG

print("="*80)
print("TÜNN ERROR TYPES VE DETAILS")
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
    
    # Get unique error types
    print("\nUnique error_type values:")
    cursor.execute("""
        SELECT DISTINCT error_type, COUNT(*) as count
        FROM robot_logs_error
        GROUP BY error_type
        ORDER BY count DESC
    """)
    
    for error_type, count in cursor.fetchall():
        print(f"  {error_type:<40} ({count:>6} records)")
    
    # Get unique error levels
    print("\n\nUnique error_level values:")
    cursor.execute("""
        SELECT DISTINCT error_level, COUNT(*) as count
        FROM robot_logs_error
        GROUP BY error_level
        ORDER BY count DESC
    """)
    
    for error_level, count in cursor.fetchall():
        print(f"  {error_level:<40} ({count:>6} records)")
    
    # Get unique error details
    print("\n\nUnique error_detail values (first 20):")
    cursor.execute("""
        SELECT DISTINCT error_detail, COUNT(*) as count
        FROM robot_logs_error
        GROUP BY error_detail
        ORDER BY count DESC
        LIMIT 25
    """)
    
    for error_detail, count in cursor.fetchall():
        print(f"  {error_detail:<40} ({count:>6} records)")
    
    cursor.close()
    conn.close()
    
    print("\n" + "="*80)
    print("SONRAKI ADIM:")
    print("  HATA_KODLARI_ROBOT.xlsx'deki hata türlerini bu listede ara")
    print("  'Yerinde destek gerekli' hataları identifiy et")
    print("="*80)
    
except Exception as e:
    print(f"❌ Error: {e}")
    sys.exit(1)
