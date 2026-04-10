#!/usr/bin/env python
"""
Create proper SQL query matching CRITICAL errors with robot_logs_error table
Based on Excel: HATA_KODLARI_ROBOT.xlsx
"""

import sys

# Critical error codes from Excel (24.4% of all errors)
CRITICAL_ERRORS = [
    'BATTRY_PORT_ERROR',
    'BusinessDefine',
    'CAMERA',
    'CanNotReach',
    'CoreNotReady',
    'DustAbsorptionError',
    'ERROR_COMM',
    'ERROR_HARDWARE_VERSION',
    'RGBD',
    'TaskCannotStart',
    'UnknownError',
    'WheelErrorRight',
    'lidar',
    'wheel',
]

# High severity error level from database
CRITICAL_ERROR_LEVELS = ['Fatal', 'Error']  # Top severity levels

print("="*80)
print("GENERATING SQL QUERY FOR HATA_KODLARI MAPPING")
print("="*80)

# Generate SQL query matching database schema
sql_query = """
-- Join robot_logs_info (success) with robot_logs_error (failures)
-- Based on HATA_KODLARI_ROBOT.xlsx severity: 24.4% Critical, 72.6% High

WITH error_classification AS (
  SELECT 
    i.ingest_id,
    i.robot_id,
    i.product_code,
    i.task_time,
    i.check_result_count,
    i.soft_version,
    i.os_version,
    
    -- Failed tasks: detected from error_logs with critical errors
    CASE 
      WHEN e.error_type IN ('BATTERY_ERROR', 'WheelErrorRight', 'WheelErrorLeft', 
                            'CAMERA', 'RGBD', 'lidar', 'ERROR_COMM',
                            'CoreNotReady', 'unknown', 'CanNotReach')
        THEN 1  -- CRITICAL FAILURE - On-site support (Yerinde destek gerekli)
      WHEN e.error_level IN ('Fatal', 'Error')
        THEN 1  -- HIGH SEVERITY - Remote support (Uzaktan destek)
      ELSE 0   -- Normal operation
    END as failure,
    
    EXTRACT(HOUR FROM i.task_time)::int as task_hour,
    EXTRACT(DAY FROM i.task_time)::int as task_day_of_month,
    EXTRACT(DOW FROM i.task_time)::int as task_day_of_week,
    LENGTH(i.robot_id)::int as robot_id_length,
    LENGTH(i.soft_version)::int as software_version_length,
    CASE WHEN i.product_code LIKE '%PuduBot%' THEN 1
         WHEN i.product_code LIKE '%KettyBot%' THEN 2
         WHEN i.product_code LIKE '%Bellabot%' THEN 3
         WHEN i.product_code LIKE '%CC%' THEN 4
         ELSE 5 END as product_code_type,
    
    COALESCE(i.check_result_count, 0) as error_severity,
    
    -- Error frequency signature
    COALESCE(e.hourly_error_count, 0) as hourly_error_rate,
    
    ROW_NUMBER() OVER (PARTITION BY i.robot_id ORDER BY i.task_time DESC) as robot_task_rank
  
  FROM robot_logs_info i
  LEFT JOIN robot_logs_error e 
    ON i.robot_id = e.robot_id 
    AND ABS(EXTRACT(EPOCH FROM i.task_time - e.task_time)) < 3600  -- Within 1 hour
)

SELECT DISTINCT
  failure,
  error_count as check_result_count,
  task_hour,
  task_day_of_month,
  task_day_of_week,
  robot_id_length,
  software_version_length,
  product_code_type,
  error_severity,
  hourly_error_rate
  
FROM error_classification 

WHERE robot_task_rank <= 1  -- Latest task per robot
  AND failure IS NOT NULL
  
LIMIT 2000
"""

print("\n📝 GENERATED SQL QUERY:")
print("="*80)
print(sql_query)

print("\n" + "="*80)
print("MAPPING RATIONALE:")
print("="*80)
print("""
⚠️  CRITICAL (Failure = 1) - 24.4% target from Excel:
   ├─ On-site support errors (Yerinde destek gerekli)
   ├─ Hardware failures: BATTERY, WHEEL, CAMERA, RGBD, lidar, ERROR_COMM
   ├─ System failures: CoreNotReady, CanNotReach, TaskCannotStart
   └─ Level: Fatal + Error

✅ NORMAL (Failure = 0) - 75.6% target:
   ├─ Remote support errors (Uzaktan destek) - 72.6%
   ├─ Warning level errors - 2.6%
   └─ Event level errors - 0.4%

📊 FEATURES USED:
   1. check_result_count (real error count from db)
   2. task_hour (temporal - when failures occur)
   3. task_day_of_month (calendar pattern)
   4. task_day_of_week (weekly pattern)
   5. robot_id_length (robot identifier)
   6. software_version_length (SW version info)
   7. product_code_type (robot model type)
   8. error_severity (raw error count when failure)
   9. hourly_error_rate (error frequency)
""")

print("="*80)
print("✅ SQL query ready for train.py")
print("="*80)
