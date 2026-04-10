import psycopg2
import sys
sys.path.insert(0, 'src')
from config import DATABASE_CONFIG

db_config = {
    'host': DATABASE_CONFIG['host'],
    'port': DATABASE_CONFIG['port'],
    'database': DATABASE_CONFIG['database'],
    'user': DATABASE_CONFIG['user'],
    'password': DATABASE_CONFIG['password']
}

try:
    conn = psycopg2.connect(**db_config)
    cur = conn.cursor()
    
    # Check error distribution by severity
    cur.execute("""
    SELECT 
        error_type,
        error_level,
        COUNT(*) as count
    FROM robot_logs_error
    GROUP BY error_type, error_level
    ORDER BY count DESC
    LIMIT 30
    """)
    
    print("=" * 80)
    print("ERROR DISTRIBUTION BY TYPE & LEVEL (Top 30)")
    print("=" * 80)
    critical_set = {'WheelErrorRight', 'WheelErrorLeft', 'CAMERA', 'RGBD', 
                    'lidar', 'ERROR_COMM', 'CoreNotReady', 'CanNotReach', 
                    'TaskCannotStart', 'DustAbsorptionError', 'UnknownError'}
    
    total_critical = 0
    total_high = 0
    total_errors = 0
    
    for row in cur.fetchall():
        error_type, error_level, count = row
        total_errors += count
        
        if error_type in critical_set:
            severity = "CRITICAL ⚠️"
            total_critical += count
        elif error_level in ('Fatal', 'Error'):
            severity = "HIGH 🔴"
            total_high += count
        else:
            severity = "OTHER ℹ️"
        
        print(f"  {error_type:25} | {error_level:10} | {count:6} | {severity}")
    
    print("\n" + "=" * 80)
    print(f"TOTALS:")
    print("=" * 80)
    print(f"  Critical errors (on-site):  {total_critical:6} ({100*total_critical/total_errors:.1f}%)")
    print(f"  High errors (remote):       {total_high:6} ({100*total_high/total_errors:.1f}%)")
    print(f"  Total error records:        {total_errors:6}")
    
    # Check if failures match errors
    print("\n" + "=" * 80)
    print("CHECKING IF SQL CLASSIFICATION WORKS:")
    print("=" * 80)
    
    cur.execute("""
    WITH critical_errors AS (
      SELECT DISTINCT robot_id
      FROM robot_logs_error
      WHERE error_type IN ('WheelErrorRight', 'WheelErrorLeft', 'CAMERA', 'RGBD', 
                           'lidar', 'ERROR_COMM', 'CoreNotReady', 'CanNotReach', 
                           'TaskCannotStart', 'DustAbsorptionError', 'UnknownError')
    ),
    high_errors AS (
      SELECT DISTINCT robot_id
      FROM robot_logs_error
      WHERE error_level IN ('Fatal', 'Error')
    )
    SELECT 
        (SELECT COUNT(DISTINCT robot_id) FROM critical_errors) as robots_with_critical,
        (SELECT COUNT(DISTINCT robot_id) FROM high_errors) as robots_with_high,
        (SELECT COUNT(DISTINCT robot_id) FROM robot_logs_info) as total_robots_in_logs
    """)
    
    result = cur.fetchone()
    robots_critical, robots_high, total_robots = result
    print(f"  Robots with critical errors: {robots_critical}")
    print(f"  Robots with high errors:     {robots_high}")
    print(f"  Total robots in logs:        {total_robots}")
    print(f"  Robots with any error:       {robots_critical + robots_high}")
    
    conn.close()
except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()
