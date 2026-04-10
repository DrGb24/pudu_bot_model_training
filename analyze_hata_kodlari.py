#!/usr/bin/env python
"""Analyze HATA_KODLARI_ROBOT.xlsx for critical errors and severity"""

import pandas as pd
import sys

print("="*80)
print("ANALYZING: HATA_KODLARI_ROBOT.xlsx")
print("="*80)

try:
    # Read excel
    df = pd.read_excel('HATA_KODLARI_ROBOT.xlsx', sheet_name=0)
    
    print(f"\nDataFrame shape: {df.shape}")
    print(f"Total rows: {len(df)}")
    print(f"\nColumns: {list(df.columns)}\n")
    
    # Find critical errors (Yerinde destek gerekli)
    critical_mask = df.astype(str).apply(lambda x: x.str.contains('Yerinde destek gerekli', case=False, na=False)).any(axis=1)
    critical_errors = df[critical_mask]
    
    print("="*80)
    print("🔴 CRITICAL ERRORS (Yerinde destek gerekli) - ON-SITE SUPPORT NEEDED")
    print("="*80)
    print(f"\nTotal critical errors: {len(critical_errors)}")
    
    if len(critical_errors) > 0:
        # Try to find error type columns
        for col in df.columns:
            # Get error types from critical rows
            critical_col_values = critical_errors[col].dropna().unique()
            if len(critical_col_values) > 0 and not all(pd.isna(critical_col_values)):
                print(f"\n{col}:")
                for val in sorted([str(v) for v in critical_col_values]):
                    if str(val).strip() and str(val) != 'nan':
                        print(f"  - {val}")
    
    # Find high severity errors (Uzaktan destek)
    high_mask = df.astype(str).apply(lambda x: x.str.contains('Uzaktan destek', case=False, na=False)).any(axis=1)
    high_errors = df[high_mask]
    
    print("\n" + "="*80)
    print("🟠 HIGH SEVERITY ERRORS (Uzaktan destek) - REMOTE SUPPORT NEEDED")
    print("="*80)
    print(f"\nTotal high severity errors: {len(high_errors)}")
    
    # Severity breakdown
    print("\n" + "="*80)
    print("SEVERITY DISTRIBUTION")
    print("="*80)
    
    total = len(df)
    critical_count = len(critical_errors)
    high_count = len(high_errors)
    other_count = total - critical_count - high_count
    
    print(f"\n🔴 Critical (Yerinde destek gerekli): {critical_count:>4} ({critical_count/total*100:>5.1f}%)")
    print(f"🟠 High (Uzaktan destek):             {high_count:>4} ({high_count/total*100:>5.1f}%)")
    print(f"🟡 Other:                            {other_count:>4} ({other_count/total*100:>5.1f}%)")
    print(f"{'─'*45}")
    print(f"Total:                               {total:>4} (100.0%)")
    
    # Save mapping
    print("\n" + "="*80)
    print("GENERATING ERROR MAPPING FOR MODEL")
    print("="*80)
    
    # Create simple mapping
    mapping = {
        'critical': [],
        'high': [],
    }
    
    # Extract error codes
    for col in ['error_type', 'error_detail', 'error_code', 'ERROR_CODE', 'Hata Türü', 'Hata Kodu']:
        if col in df.columns:
            # Get critical error codes
            critical_vals = critical_errors[col].dropna().unique()
            for val in critical_vals:
                if str(val).strip() and str(val) != 'nan':
                    mapping['critical'].append(str(val))
            
            # Get high severity error codes
            high_vals = high_errors[col].dropna().unique()
            for val in high_vals:
                if str(val).strip() and str(val) != 'nan':
                    if str(val) not in mapping['critical']:
                        mapping['high'].append(str(val))

    print("\n✅ Critical Error Codes to Flag as Failure=1:")
    print(f"   Total: {len(set(mapping['critical']))}")
    for code in sorted(set(mapping['critical']))[:10]:
        print(f"   - {code}")
    if len(set(mapping['critical'])) > 10:
        print(f"   ... and {len(set(mapping['critical'])) - 10} more")
    
    print("\n⚠️  High Severity Error Codes (for attention):")
    print(f"   Total: {len(set(mapping['high']))}")
    for code in sorted(set(mapping['high']))[:10]:
        print(f"   - {code}")
    if len(set(mapping['high'])) > 10:
        print(f"   ... and {len(set(mapping['high'])) - 10} more")
    
    print("\n" + "="*80)
    
except Exception as e:
    print(f"❌ Error: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
