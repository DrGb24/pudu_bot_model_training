# Training Results Summary

## Model: Random Forest Binary Classifier  
- **Architecture**: 2000 estimators, max_depth=50, random_state=42
- **Training Data**: 2000 records from PostgreSQL robot_logs_info table
- **Test Split**: 70% training / 15% validation / 15% test

## Results
```
Accuracy:        99.33%  ✅ (Target: ≥85%)
Precision:      100.00%  ✅ (Target: ≥80%)
Recall:          71.43%  ⚠️  (Target: ≥85% - MISS)
F1-Score:        83.33%  ✅ (Target: ≥80%)
False Alarm Rate:  0.0%  ✅ (Target: ≤10%)
```

## Data Analysis
- **Target Distribution**: 48 failures (2.4%), 1952 successes (97.6%)
- **Class Imbalance**: Highly imbalanced dataset (1:40 ratio)
- **Data Source**: Real production data from robot_logs_info (is_success flag)
- **Features**: 9 numerical features from database

## Top Features by Importance
1. error_count (42.1%)
2. error_severity (38.6%)
3. software_version_length (6.7%)
4. task_day_of_week (5.7%)
5. task_hour (2.7%)

## Excel Severity Mapping Status
- **Critical errors** (on-site support): 33/135 errors in Excel (24.4%)
- **High severity errors** (remote support): 98/135 errors in Excel (72.6%)
- **Database actual**: 
  - Critical: 1.9% of error records
  - High: 5.2% of error records
  - Note: Most database records are informational (Event/Warning level)

## Issues Resolved
1. ✅ Database RANDOM() function removed - using actual data
2. ✅ is_success flag implemented as primary failure label
3. ✅ 2000 records loaded from PostgreSQL
4. ✅ SQL query simplified for reliability
5. ✅ Config updated with hourly_error_rate feature

## Outstanding Items
- ⏳ 5-run consistency test in progress (verify stability across multiple training iterations)
- ⏳ Recall target (71.43% vs 85% target) needs investigation - likely due to class imbalance

## Model Deployment
- Model saved: `models/random_forest_model.pkl`
- Feature names: `models/feature_names.npy`
- Reports: `logs/final_report.csv`, `logs/kpi_report.csv`
