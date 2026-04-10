#!/usr/bin/env python
"""Train model 5 times and compare accuracy - verify data consistency"""

import subprocess
import csv
import os
from pathlib import Path
from datetime import datetime

print("="*80)
print("5-RUN ACCURACY CONSISTENCY TEST")
print("Training with NEW SQL (real extracted features, no RANDOM)")
print("="*80)

accuracies = []
reports_dir = Path('logs')
run_times = []

for run_num in range(1, 6):
    print(f"\n{'='*80}")
    print(f"RUN {run_num}/5: Training model...")
    print('='*80)
    
    # Run train.py
    result = subprocess.run(['python', 'train.py'], capture_output=True, text=True)
    
    if result.returncode != 0:
        print(f"❌ Error in run {run_num}:")
        print(result.stderr[-500:])  # Last 500 chars of error
        continue
    
    # Read accuracy from final_report.csv
    report_path = reports_dir / 'final_report.csv'
    
    if report_path.exists():
        with open(report_path, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                accuracy = float(row['accuracy'])
                accuracies.append({
                    'run': run_num,
                    'accuracy': accuracy,
                    'accuracy_pct': accuracy * 100,
                    'precision': float(row['precision']),
                    'recall': float(row['recall']),
                    'f1_score': float(row['f1_score'])
                })
                
                print(f"✅ Accuracy: {accuracy*100:.2f}%")
                print(f"   Precision: {row['precision']}")
                print(f"   Recall: {row['recall']}")
                print(f"   F1-Score: {row['f1_score']}")
                break

print("\n" + "="*80)
print("RESULTS SUMMARY")
print("="*80)

if len(accuracies) == 5:
    print("\nRun-by-Run Accuracies:")
    print(f"{'Run':<6} {'Accuracy':<15} {'Precision':<15} {'Recall':<15} {'F1-Score':<15}")
    print("-" * 80)
    
    for result in accuracies:
        print(f"{result['run']:<6} {result['accuracy_pct']:>6.2f}%          "
              f"{result['precision']:>6.4f}          "
              f"{result['recall']:>6.4f}          "
              f"{result['f1_score']:>6.4f}")
    
    # Statistics
    accs = [r['accuracy'] for r in accuracies]
    min_acc = min(accs)
    max_acc = max(accs)
    avg_acc = sum(accs) / len(accs)
    accuracy_range = max_acc - min_acc
    
    print("\n" + "="*80)
    print("CONSISTENCY ANALYSIS")
    print("="*80)
    print(f"\nMinimum Accuracy: {min_acc*100:.2f}%")
    print(f"Maximum Accuracy: {max_acc*100:.2f}%")
    print(f"Average Accuracy: {avg_acc*100:.2f}%")
    print(f"Range (Max-Min):  {accuracy_range*100:.2f}%")
    print(f"Std Deviation:    {(max(accs) - min(accs))*100 / 2:.2f}% (approx)")
    
    print("\n" + "="*80)
    print("VERDICT")
    print("="*80)
    
    if accuracy_range < 0.01:  # Less than 1% variation
        print("\n✅ EXCELLENT CONSISTENCY!")
        print(f"   Accuracy varies by only {accuracy_range*100:.3f}%")
        print("   → Data is STABLE (no RANDOM() issues!)")
        print("   → Real extracted features are working correctly")
    elif accuracy_range < 0.05:  # Less than 5% variation
        print("\n⚠️  GOOD CONSISTENCY (minor variations)")
        print(f"   Accuracy varies by {accuracy_range*100:.2f}%")
        print("   → Data is stable with expected train/test randomness")
    else:
        print("\n❌ HIGH VARIATION DETECTED!")
        print(f"   Accuracy varies by {accuracy_range*100:.2f}%")
        print("   → Possible data instability or feature issues")
    
    # Save results
    results_file = Path('test_results_5runs.csv')
    with open(results_file, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['run', 'accuracy', 'precision', 'recall', 'f1_score'])
        writer.writeheader()
        for r in accuracies:
            writer.writerow({
                'run': r['run'],
                'accuracy': f"{r['accuracy']:.4f}",
                'precision': f"{r['precision']:.4f}",
                'recall': f"{r['recall']:.4f}",
                'f1_score': f"{r['f1_score']:.4f}"
            })
    
    print(f"\n📊 Results saved to: {results_file}")
    print("="*80)

else:
    print(f"❌ Only {len(accuracies)} runs succeeded out of 5!")
