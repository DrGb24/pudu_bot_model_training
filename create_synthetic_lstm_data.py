"""
Sentetik LSTM Eğitim Verisi Oluşturucu
Real veri istatistikleriyle benzer sentetik robot log verileri üretir
"""

import logging
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os
import sys
from pathlib import Path

# Projeyi ekle
sys.path.insert(0, str(Path(__file__).parent))

from src.config import DATABASE_CONFIG
from src.data_preparation import DataPreparation

# Loglama ayarı
logging.basicConfig(
    level=logging.INFO,
    format='%(levelname)s:%(name)s:%(message)s'
)
logger = logging.getLogger(__name__)

class SyntheticLSTMDataGenerator:
    """Gerçek veri istatistiklerine dayalı sentetik LSTM verisi üretici"""
    
    def __init__(self):
        self.real_stats = None
        
    def load_real_data_stats(self):
        """Real veri istatistiklerini yükle"""
        logger.info("📊 Real veri istatistikleri yükleniyor...")
        
        data_prep = DataPreparation()
        
        # Eğer database mevcutsa stats'ı al
        sql_query = """
            SELECT 
              (1 - is_success::int) as failure,
              check_result_count as error_count,
              EXTRACT(HOUR FROM task_time)::int as task_hour,
              EXTRACT(DAY FROM task_time)::int as day_of_month,
              EXTRACT(DOW FROM task_time)::int as day_of_week,
              LENGTH(robot_id)::int as robot_id_length,
              LENGTH(soft_version)::int as software_version_length,
              CASE WHEN product_code LIKE '%PuduBot%' THEN 1
                   WHEN product_code LIKE '%KettyBot%' THEN 2
                   WHEN product_code LIKE '%Bellabot%' THEN 3
                   WHEN product_code LIKE '%CC%' THEN 4
                   ELSE 5 END as product_code_type,
              check_result_count as error_severity,
              COALESCE(0, 0) as hourly_error_rate
            FROM robot_logs_info
            ORDER BY RANDOM()
        """
        
        try:
            df_real = data_prep.load_from_database(DATABASE_CONFIG, query=sql_query)
            
            self.real_stats = {
                'failure_rate': df_real['failure'].mean(),
                'error_count': {
                    'mean': df_real['error_count'].mean(),
                    'std': df_real['error_count'].std(),
                    'min': df_real['error_count'].min(),
                    'max': df_real['error_count'].max()
                },
                'task_hour': {
                    'mean': df_real['task_hour'].mean(),
                    'std': df_real['task_hour'].std(),
                    'min': 0,
                    'max': 23
                },
                'day_of_month': {
                    'mean': df_real['day_of_month'].mean(),
                    'std': df_real['day_of_month'].std(),
                    'min': 1,
                    'max': 31
                },
                'day_of_week': {
                    'mean': df_real['day_of_week'].mean(),
                    'std': df_real['day_of_week'].std(),
                    'min': 0,
                    'max': 6
                },
                'robot_id_length': {
                    'mean': df_real['robot_id_length'].mean(),
                    'std': df_real['robot_id_length'].std(),
                    'min': df_real['robot_id_length'].min(),
                    'max': df_real['robot_id_length'].max()
                },
                'software_version_length': {
                    'mean': df_real['software_version_length'].mean(),
                    'std': df_real['software_version_length'].std(),
                    'min': df_real['software_version_length'].min(),
                    'max': df_real['software_version_length'].max()
                },
                'product_code_type': {
                    'mean': df_real['product_code_type'].mean(),
                    'std': df_real['product_code_type'].std(),
                    'min': df_real['product_code_type'].min(),
                    'max': df_real['product_code_type'].max()
                },
                'error_severity': {
                    'mean': df_real['error_severity'].mean(),
                    'std': df_real['error_severity'].std(),
                    'min': df_real['error_severity'].min(),
                    'max': df_real['error_severity'].max()
                },
                'hourly_error_rate': {
                    'mean': df_real['hourly_error_rate'].mean(),
                    'std': df_real['hourly_error_rate'].std(),
                    'min': 0,
                    'max': 1
                },
                'num_robots': 48
            }
            
            logger.info(f"✅ Real veri istatistikleri yüklendi")
            logger.info(f"   Real failure rate: {self.real_stats['failure_rate']:.4f}")
            logger.info(f"   Error count mean: {self.real_stats['error_count']['mean']:.2f}")
            
            return df_real
            
        except Exception as e:
            logger.error(f"❌ Database bağlantı hatası: {e}")
            raise
    
    def generate_synthetic_data(self, num_samples=11100, enhanced_failure_rate=0.12):
        """
        Sentetik veri oluştur
        
        Args:
            num_samples: Oluşturulacak örnek sayısı
            enhanced_failure_rate: Sentetik veriye failure rate (default: 0.12 = %12)
                                   Real: %5.24, Synthetic: %12 (class imbalance azaltmak için)
        """
        logger.info(f"🔄 {num_samples:,} sentetik örnek oluşturuluyor...")
        logger.info(f"   Target failure rate: {enhanced_failure_rate:.2%}")
        
        np.random.seed(42)
        
        # Num_samples'ın %enhanced_failure_rate kadarı failure, gerisi normal
        num_failures = int(num_samples * enhanced_failure_rate)
        num_normal = num_samples - num_failures
        
        synthetic_data = []
        
        # Normal örnekler oluştur
        for i in range(num_normal):
            sample = self._generate_normal_sample()
            synthetic_data.append(sample)
        
        # Failure örnekler oluştur (yüksek hata sayısı, yüksek severity)
        for i in range(num_failures):
            sample = self._generate_failure_sample()
            synthetic_data.append(sample)
        
        df_synthetic = pd.DataFrame(synthetic_data)
        
        # Karıştır
        df_synthetic = df_synthetic.sample(frac=1).reset_index(drop=True)
        
        failure_count = (df_synthetic['failure'] == 1).sum()
        logger.info(f"✅ Sentetik veri oluşturuldu: {len(df_synthetic):,} örnek")
        logger.info(f"   Oluşturulan failure'lar: {failure_count:,} ({failure_count/len(df_synthetic):.2%})")
        
        return df_synthetic
    
    def _generate_normal_sample(self):
        """Normal işlem örneği oluştur (failure=0)"""
        stats = self.real_stats
        
        return {
            'failure': 0,
            'error_count': max(0, int(np.random.normal(
                stats['error_count']['mean'],
                stats['error_count']['std']
            ))),
            'task_hour': int(np.clip(np.random.normal(
                stats['task_hour']['mean'],
                stats['task_hour']['std']
            ), 0, 23)),
            'day_of_month': int(np.clip(np.random.normal(
                stats['day_of_month']['mean'],
                stats['day_of_month']['std']
            ), 1, 31)),
            'day_of_week': int(np.clip(np.random.normal(
                stats['day_of_week']['mean'],
                stats['day_of_week']['std']
            ), 0, 6)),
            'robot_id_length': int(np.clip(np.random.normal(
                stats['robot_id_length']['mean'],
                stats['robot_id_length']['std']
            ), stats['robot_id_length']['min'], stats['robot_id_length']['max'])),
            'software_version_length': int(np.clip(np.random.normal(
                stats['software_version_length']['mean'],
                stats['software_version_length']['std']
            ), stats['software_version_length']['min'], stats['software_version_length']['max'])),
            'product_code_type': int(np.clip(np.random.normal(
                stats['product_code_type']['mean'],
                stats['product_code_type']['std']
            ), stats['product_code_type']['min'], stats['product_code_type']['max'])),
            'error_severity': int(np.clip(np.random.normal(
                stats['error_severity']['mean'],
                stats['error_severity']['std'] * 0.5  # Normal'lar daha düşük severity
            ), stats['error_severity']['min'], stats['error_severity']['max'])),
            'hourly_error_rate': int(np.random.choice([0, 1], p=[0.7, 0.3]))
        }
    
    def _generate_failure_sample(self):
        """Failure örneği oluştur (failure=1, yüksek risk göstergeleri)"""
        stats = self.real_stats
        
        return {
            'failure': 1,
            'error_count': int(np.clip(
                np.random.normal(
                    stats['error_count']['mean'] * 1.5,  # Yüksek hata sayısı
                    stats['error_count']['std']
                ),
                stats['error_count']['min'],
                stats['error_count']['max']
            )),
            'task_hour': int(np.clip(np.random.normal(
                stats['task_hour']['mean'],
                stats['task_hour']['std'] * 1.2
            ), 0, 23)),
            'day_of_month': int(np.clip(np.random.normal(
                stats['day_of_month']['mean'],
                stats['day_of_month']['std']
            ), 1, 31)),
            'day_of_week': int(np.clip(np.random.normal(
                stats['day_of_week']['mean'],
                stats['day_of_week']['std']
            ), 0, 6)),
            'robot_id_length': int(np.clip(np.random.normal(
                stats['robot_id_length']['mean'],
                stats['robot_id_length']['std']
            ), stats['robot_id_length']['min'], stats['robot_id_length']['max'])),
            'software_version_length': int(np.clip(np.random.normal(
                stats['software_version_length']['mean'],
                stats['software_version_length']['std']
            ), stats['software_version_length']['min'], stats['software_version_length']['max'])),
            'product_code_type': int(np.clip(np.random.normal(
                stats['product_code_type']['mean'],
                stats['product_code_type']['std'] * 1.3  # Daha yüksek varyasyon
            ), stats['product_code_type']['min'], stats['product_code_type']['max'])),
            'error_severity': int(np.clip(
                np.random.normal(
                    stats['error_severity']['mean'] * 1.5,  # Yüksek severity
                    stats['error_severity']['std']
                ),
                stats['error_severity']['min'],
                stats['error_severity']['max']
            )),
            'hourly_error_rate': int(np.random.choice([0, 1], p=[0.2, 0.8]))  # Yüksek error rate
        }
    
    def merge_real_and_synthetic(self, df_real, df_synthetic):
        """Real ve sentetik veriyi birleştir"""
        logger.info("🔗 Real ve sentetik veriler birleştiriliyor...")
        
        # Real veriyi gerekli formata dönüştür
        if 'failure' not in df_real.columns:
            df_real['failure'] = 1 - df_real.get('is_success', 0)
        
        # Gerekli sütunları seç
        required_cols = ['failure', 'error_count', 'task_hour', 'day_of_month', 
                        'day_of_week', 'robot_id_length', 'software_version_length',
                        'product_code_type', 'error_severity', 'hourly_error_rate']
        
        df_real_subset = df_real[required_cols].copy()
        
        # Birleştir
        df_combined = pd.concat([df_real_subset, df_synthetic], ignore_index=True)
        df_combined = df_combined.sample(frac=1).reset_index(drop=True)
        
        real_failures = (df_real_subset['failure'] == 1).sum()
        synth_failures = (df_synthetic['failure'] == 1).sum()
        total_failures = (df_combined['failure'] == 1).sum()
        
        logger.info(f"✅ Birleştirme tamamlandı:")
        logger.info(f"   Real examples: {len(df_real_subset):,} ({real_failures:,} failures, {real_failures/len(df_real_subset):.2%})")
        logger.info(f"   Synthetic examples: {len(df_synthetic):,} ({synth_failures:,} failures, {synth_failures/len(df_synthetic):.2%})")
        logger.info(f"   Total examples: {len(df_combined):,} ({total_failures:,} failures, {total_failures/len(df_combined):.2%})")
        
        return df_combined


def main():
    """Main execution"""
    logger.info("=" * 80)
    logger.info("SENTETIK LSTM EĞİTİM VERİSİ OLUŞTURUCUSU")
    logger.info("=" * 80)
    
    try:
        # Generator oluştur
        generator = SyntheticLSTMDataGenerator()
        
        # Real veri stats yükle
        df_real = generator.load_real_data_stats()
        
        # Sentetik veri oluştur (11,100 örnek, %12 failure rate)
        df_synthetic = generator.generate_synthetic_data(
            num_samples=11100,
            enhanced_failure_rate=0.12
        )
        
        # Birleştir
        df_combined = generator.merge_real_and_synthetic(df_real, df_synthetic)
        
        # Dosyaya kaydet
        output_path = Path('data/lstm_combined_15k.csv')
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        df_combined.to_csv(output_path, index=False)
        logger.info(f"💾 Birleştirilmiş veriler kaydedildi: {output_path}")
        
        # İstatistikler
        logger.info("\n📊 Son İstatistikler:")
        logger.info(f"   Total samples: {len(df_combined):,}")
        logger.info(f"   Total failures: {(df_combined['failure'] == 1).sum():,}")
        logger.info(f"   Failure rate: {df_combined['failure'].mean():.2%}")
        
        logger.info("\n" + "=" * 80)
        logger.info("✅ SENTETIK VERİ OLUŞTURMA TAMAMLANDI")
        logger.info("=" * 80)
        
    except Exception as e:
        logger.error(f"❌ Hata: {e}", exc_info=True)
        sys.exit(1)


if __name__ == '__main__':
    main()
