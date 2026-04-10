#!/usr/bin/env python
"""Read HATA_KODLARI_ROBOT.xlsx and extract critical error codes"""

import openpyxl
import pandas as pd
import sys

print("="*80)
print("READING: HATA_KODLARI_ROBOT.xlsx")
print("="*80)

try:
    # Method 1: Try with pandas (faster)
    try:
        df = pd.read_excel('HATA_KODLARI_ROBOT.xlsx', sheet_name=0)
        print(f"\nDataFrame shape: {df.shape}")
        print(f"Columns: {list(df.columns)}\n")
        print(df.to_string())
        
    except Exception as e:
        print(f"Pandas error: {e}")
        print("\nTrying with openpyxl...\n")
        
        # Method 2: openpyxl
        wb = openpyxl.load_workbook('HATA_KODLARI_ROBOT.xlsx')
        ws = wb.active
        
        print(f"Sheet name: {ws.title}")
        print(f"\nAll data:\n")
        
        for row_idx, row in enumerate(ws.iter_rows(values_only=True), 1):
            print(f"Row {row_idx}: {row}")
    
    print("\n" + "="*80)
    print("✅ File read successfully!")
    print("="*80)
    
except FileNotFoundError:
    print("❌ File not found: HATA_KODLARI_ROBOT.xlsx")
    sys.exit(1)
except Exception as e:
    print(f"❌ Error: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
