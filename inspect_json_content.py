#!/usr/bin/env python
"""Check what's inside check_result_json"""

import psycopg2
import json
import sys

sys.path.insert(0, 'src')
from config import DATABASE_CONFIG

print("="*80)
print("CHECK_RESULT_JSON CONTENT INSPECTION")
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
    
    # Get one record with check_result_json
    cursor.execute("""
        SELECT 
            ingest_id,
            robot_id,
            is_success,
            check_result_count,
            check_result_json
        FROM robot_logs_info 
        WHERE check_result_json IS NOT NULL
        LIMIT 5
    """)
    
    records = cursor.fetchall()
    
    for idx, (ingest_id, robot_id, is_success, check_count, json_data) in enumerate(records, 1):
        print(f"\n{'='*80}")
        print(f"Record {idx}: ingest_id={ingest_id}, robot_id={robot_id}")
        print(f"is_success={is_success}, check_result_count={check_count}")
        print(f"{'='*80}")
        
        if json_data:
            try:
                data = json.loads(json_data) if isinstance(json_data, str) else json_data
                
                if isinstance(data, list):
                    print(f"\nJSON array with {len(data)} items:\n")
                    for item_idx, item in enumerate(data[:3]):  # Show first 3 items
                        print(f"Item {item_idx}:")
                        if isinstance(item, dict):
                            for key, val in item.items():
                                val_str = str(val)[:80]  # Limit to 80 chars
                                print(f"  {key}: {val_str}")
                        else:
                            print(f"  {item}")
                        print()
                    
                    if len(data) > 3:
                        print(f"... and {len(data) - 3} more items\n")
                
                elif isinstance(data, dict):
                    print("\nJSON object:\n")
                    for key, val in data.items():
                        val_str = str(val)[:80]
                        print(f"  {key}: {val_str}")
                    
            except Exception as e:
                print(f"Error parsing JSON: {e}")
                print(f"Raw data (first 500 chars): {str(json_data)[:500]}")
    
    cursor.close()
    conn.close()
    
    print("\n" + "="*80)
    print("CONCLUSION:")
    print("="*80)
    print("If JSON contains numeric sensor data, we can extract features.")
    print("Otherwise, use: is_success, check_result_count, product_code, os_version")
    print("="*80)
    
except Exception as e:
    print(f"❌ Error: {e}")
    sys.exit(1)
