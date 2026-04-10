#!/usr/bin/env python
"""Test PostgreSQL database connection and check available tables"""

import psycopg2
from psycopg2.extras import RealDictCursor

try:
    print("🔌 PostgreSQL'e bağlanılıyor...")
    conn = psycopg2.connect(
        host='149.102.155.77',
        port=5433,
        database='robot_pipeline',
        user='robot_pipeline_admin',
        password='RobotPipe!2026#PG!149',
        sslmode='disable'
    )
    print("✅ PostgreSQL Bağlantısı Başarılı!")
    
    cur = conn.cursor()
    
    # Get all tables
    cur.execute("SELECT table_name FROM information_schema.tables WHERE table_schema = 'public'")
    tables = cur.fetchall()
    
    print(f'\n📋 Veritabanında Bulunan Tablolar ({len(tables)} tane):')
    for table in tables:
        print(f'  - {table[0]}')
    
    # Check robots_data table
    if tables:
        cur.execute("SELECT 1 FROM information_schema.tables WHERE table_name='robots_data'")
        if cur.fetchone():
            cur.execute('SELECT COUNT(*) FROM robots_data')
            count = cur.fetchone()[0]
            print(f'\n✅ robots_data tablosu var: {count} satır')
            
            # Get column info
            cur.execute("SELECT column_name, data_type FROM information_schema.columns WHERE table_name='robots_data' ORDER BY ordinal_position")
            cols = cur.fetchall()
            print(f'\n📊 Sütunlar ({len(cols)} tane):')
            for col in cols:
                print(f'  - {col[0]}: {col[1]}')
            
            # Check sample data
            cur.execute('SELECT * FROM robots_data LIMIT 1')
            sample = cur.fetchone()
            if sample:
                print(f'\n📝 Örnek veri (ilk satır):')
                col_names = [desc[0] for desc in cur.description]
                for name, value in zip(col_names, sample):
                    print(f'  - {name}: {value}')
        else:
            print('\n❌ robots_data tablosu bulunamadı!')
            print('Mevcut tablolardan birini seçin.')
    
    cur.close()
    conn.close()
    print('\n✅ Bağlantı kapatıldı')
    
except Exception as e:
    print(f'❌ Hata: {str(e)}')
    import traceback
    traceback.print_exc()
