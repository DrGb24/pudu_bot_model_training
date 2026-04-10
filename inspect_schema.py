#!/usr/bin/env python
"""Inspect robot_logs_info table schema"""

import psycopg2
import sys

sys.path.insert(0, 'src')
from config import DATABASE_CONFIG

print("="*80)
print("ROBOT_LOGS_INFO TABLE SCHEMA")
print("="*80)

try:
    # Connect to database
    conn = psycopg2.connect(
        host=DATABASE_CONFIG['host'],
        port=DATABASE_CONFIG['port'],
        database=DATABASE_CONFIG['database'],
        user=DATABASE_CONFIG['user'],
        password=DATABASE_CONFIG['password'],
        sslmode=DATABASE_CONFIG.get('ssl_mode', 'disable')
    )
    
    cursor = conn.cursor()
    
    # Get column info
    cursor.execute("""
        SELECT 
            column_name, 
            data_type, 
            is_nullable,
            column_default
        FROM information_schema.columns
        WHERE table_name = 'robot_logs_info'
        ORDER BY ordinal_position
    """)
    
    columns = cursor.fetchall()
    
    print("\nColumns in robot_logs_info:\n")
    print(f"{'#':<3} {'Column Name':<30} {'Type':<15} {'Nullable':<8} {'Default':<20}")
    print("-" * 80)
    
    for i, (col_name, data_type, nullable, default) in enumerate(columns, 1):
        nullable_str = "YES" if nullable == "YES" else "NO"
        default_str = str(default) if default else "None"
        print(f"{i:<3} {col_name:<30} {data_type:<15} {nullable_str:<8} {default_str:<20}")
    
    print("\n" + "="*80)
    print(f"TOTAL: {len(columns)} columns")
    print("="*80)
    
    # Get sample data
    print("\nSAMPLE DATA (First 3 rows):\n")
    cursor.execute("SELECT * FROM robot_logs_info LIMIT 3")
    sample = cursor.fetchall()
    
    col_names = [desc[0] for desc in cursor.description]
    
    for row_num, row in enumerate(sample, 1):
        print(f"\nRow {row_num}:")
        for col_name, val in zip(col_names, row):
            if isinstance(val, (int, float)):
                print(f"  {col_name}: {val}")
            else:
                val_str = str(val)[:60]  # Truncate long strings
                print(f"  {col_name}: {val_str}")
    
    cursor.close()
    conn.close()
    
    print("\n" + "="*80)
    print("✅ Connection successful!")
    print("="*80)
    
except Exception as e:
    print(f"❌ Error: {e}")
    sys.exit(1)
