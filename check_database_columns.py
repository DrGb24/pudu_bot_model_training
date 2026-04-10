#!/usr/bin/env python
"""Check what actual columns are in the database"""

import sys
sys.path.insert(0, 'src')

from data_preparation import DataPreparation
from config import DATABASE_CONFIG

print("="*70)
print("DATABASE SCHEMA INSPECTION")
print("="*70)

dp = DataPreparation()

# Get schema info
schema = dp.get_table_schema(DATABASE_CONFIG, 'robot_logs_info')
print("\nColumns in robot_logs_info:")
print(schema[['column_name', 'data_type']].to_string(index=False))
