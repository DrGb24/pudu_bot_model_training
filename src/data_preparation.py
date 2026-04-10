"""
Data Preparation Module for Predictive Maintenance System
Prepared data for tree-based ML models
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
import logging
from pathlib import Path

# Database imports
try:
    import psycopg2
    from psycopg2.extras import RealDictCursor
    PSYCOPG2_AVAILABLE = True
except ImportError:
    PSYCOPG2_AVAILABLE = False
    logger = logging.getLogger(__name__)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DataPreparation:
    """Handle data loading, cleaning, and preprocessing"""
    
    def __init__(self, random_state=42):
        self.random_state = random_state
        self.scaler = StandardScaler()
        self.label_encoders = {}
        self.feature_columns = None
        self.target_column = None
        
    def load_data(self, filepath):
        """Load data from CSV file"""
        try:
            data = pd.read_csv(filepath)
            logger.info(f"Data loaded successfully. Shape: {data.shape}")
            return data
        except FileNotFoundError:
            logger.error(f"File not found: {filepath}")
            raise
    
    def load_from_database(self, db_config, query=None, table_name=None):
        """
        Load data from PostgreSQL database
        
        Parameters:
        - db_config: Dictionary with connection details
          {
            'host': 'xxx',
            'port': 5433,
            'database': 'xxx',
            'user': 'xxx',
            'password': 'xxx',
            'ssl_mode': 'disable'
          }
        - query: Custom SQL query (Optional)
        - table_name: Table name to load (if no query)
        
        Returns:
        - DataFrame with loaded data
        """
        
        if not PSYCOPG2_AVAILABLE:
            logger.error("psycopg2 not installed. Install with: pip install psycopg2-binary")
            raise ImportError("psycopg2-binary required for database connections")
        
        try:
            # Build connection string
            conn_string = f"postgresql://{db_config['user']}:{db_config['password']}" \
                         f"@{db_config['host']}:{db_config['port']}/" \
                         f"{db_config['database']}"
            
            # Connect using psycopg2
            conn = psycopg2.connect(
                host=db_config['host'],
                port=db_config['port'],
                database=db_config['database'],
                user=db_config['user'],
                password=db_config['password'],
                sslmode=db_config.get('ssl_mode', 'disable')
            )
            
            logger.info(f"✅ Connected to PostgreSQL: {db_config['database']}@{db_config['host']}")
            
            # Load data
            if query:
                logger.info(f"Executing custom query...")
                df = pd.read_sql_query(query, conn)
            elif table_name:
                query = f"SELECT * FROM {table_name}"
                logger.info(f"Loading table: {table_name}")
                df = pd.read_sql_query(query, conn)
            else:
                raise ValueError("Either 'query' or 'table_name' must be provided")
            
            conn.close()
            logger.info(f"✅ Data loaded successfully. Shape: {df.shape}")
            
            return df
            
        except psycopg2.Error as e:
            logger.error(f"❌ Database error: {e}")
            raise
        except Exception as e:
            logger.error(f"❌ Error loading data: {e}")
            raise
            
    def get_database_tables(self, db_config):
        """Get list of all tables in PostgreSQL database"""
        if not PSYCOPG2_AVAILABLE:
            raise ImportError("psycopg2-binary required")
        
        try:
            conn = psycopg2.connect(
                host=db_config['host'],
                port=db_config['port'],
                database=db_config['database'],
                user=db_config['user'],
                password=db_config['password'],
                sslmode=db_config.get('ssl_mode', 'disable')
            )
            
            query = """
            SELECT table_name FROM information_schema.tables 
            WHERE table_schema = 'public'
            """
            tables = pd.read_sql_query(query, conn)
            conn.close()
            
            logger.info(f"✅ Available tables: {tables['table_name'].tolist()}")
            return tables['table_name'].tolist()
            
        except Exception as e:
            logger.error(f"❌ Error fetching tables: {e}")
            raise
            
    def get_table_schema(self, db_config, table_name):
        """Get column names and types from a table"""
        if not PSYCOPG2_AVAILABLE:
            raise ImportError("psycopg2-binary required")
        
        try:
            conn = psycopg2.connect(
                host=db_config['host'],
                port=db_config['port'],
                database=db_config['database'],
                user=db_config['user'],
                password=db_config['password'],
                sslmode=db_config.get('ssl_mode', 'disable')
            )
            
            query = f"""
            SELECT column_name, data_type FROM information_schema.columns 
            WHERE table_name = '{table_name}'
            """
            schema = pd.read_sql_query(query, conn)
            conn.close()
            
            logger.info(f"✅ Schema for '{table_name}':")
            for idx, row in schema.iterrows():
                logger.info(f"   - {row['column_name']}: {row['data_type']}")
            
            return schema
            
        except Exception as e:
            logger.error(f"❌ Error fetching schema: {e}")
            raise
            
    def handle_missing_values(self, df, strategy='mean'):
        """Handle missing values in the dataset"""
        missing_cols = df.columns[df.isnull().any()].tolist()
        
        if missing_cols:
            logger.info(f"Handling missing values for columns: {missing_cols}")
            
            if strategy == 'mean':
                df[missing_cols] = df[missing_cols].fillna(df[missing_cols].mean())
            elif strategy == 'median':
                df[missing_cols] = df[missing_cols].fillna(df[missing_cols].median())
            elif strategy == 'forward_fill':
                df[missing_cols] = df[missing_cols].fillna(method='ffill')
                
        return df
    
    def encode_categorical_features(self, df, categorical_cols):
        """Encode categorical features"""
        df_encoded = df.copy()
        
        for col in categorical_cols:
            if col not in self.label_encoders:
                self.label_encoders[col] = LabelEncoder()
                df_encoded[col] = self.label_encoders[col].fit_transform(df[col].astype(str))
            else:
                df_encoded[col] = self.label_encoders[col].transform(df[col].astype(str))
                
        logger.info(f"Encoded {len(categorical_cols)} categorical features")
        return df_encoded
    
    def remove_outliers(self, df, columns, threshold=3):
        """Remove outliers using z-score method"""
        from scipy import stats
        
        df_clean = df.copy()
        for col in columns:
            z_scores = np.abs(stats.zscore(df_clean[col].dropna()))
            mask = z_scores < threshold
            initial_len = len(df_clean)
            df_clean = df_clean[(z_scores < threshold) | (df_clean[col].isnull())]
            removed = initial_len - len(df_clean)
            if removed > 0:
                logger.info(f"Removed {removed} outliers from {col}")
                
        return df_clean
    
    def scale_features(self, X_train, X_test=None):
        """Scale numerical features"""
        X_train_scaled = self.scaler.fit_transform(X_train)
        
        if X_test is not None:
            X_test_scaled = self.scaler.transform(X_test)
            return X_train_scaled, X_test_scaled
            
        return X_train_scaled
    
    def prepare_data(self, filepath, target_column, categorical_cols=None, 
                    numerical_cols=None, test_size=0.2, validation_size=None, 
                    return_validation=False):
        """
        Complete pipeline for data preparation
        
        Parameters:
        - filepath: Path to data file
        - target_column: Name of target column
        - categorical_cols: List of categorical column names
        - numerical_cols: List of numerical column names
        - test_size: Test set size ratio (or use train/validation/test split if validation_size provided)
        - validation_size: If provided, splits into train/validation/test (e.g., 0.15 for 70/15/15 split)
        - return_validation: If True, returns validation set as well
        
        Returns:
        - If return_validation=False: X_train, X_test, y_train, y_test, feature_names
        - If return_validation=True: X_train, X_val, X_test, y_train, y_val, y_test, feature_names
        """
        
        # Load data
        df = self.load_data(filepath)
        
        # Handle missing values
        df = self.handle_missing_values(df, strategy='mean')
        
        # Encode categorical features
        if categorical_cols:
            df = self.encode_categorical_features(df, categorical_cols)
        
        # Note: Outlier removal disabled for bounded integer features (task_hour, task_day, etc)
        # These features have limited ranges (0-23 for hour, 0-31 for day) where z-score outlier detection fails
        # if numerical_cols:
        #     df = self.remove_outliers(df, numerical_cols, threshold=10)
        
        # Prepare features and target
        self.target_column = target_column
        y = df[target_column]
        
        # Drop target and non-feature columns
        feature_cols = [col for col in df.columns if col != target_column]
        X = df[feature_cols]
        self.feature_columns = feature_cols
        
        # Split data - with or without validation set
        if validation_size is not None and return_validation:
            # Split: train (70%), temp (30%)
            X_train, X_temp, y_train, y_temp = train_test_split(
                X, y, test_size=(validation_size + validation_size), 
                random_state=self.random_state, stratify=y
            )
            
            # Split temp into validation (50%) and test (50%)
            val_test_ratio = validation_size / (validation_size + validation_size)
            X_val, X_test, y_val, y_test = train_test_split(
                X_temp, y_temp, test_size=0.5,
                random_state=self.random_state, stratify=y_temp
            )
            
            # Scale features
            X_train_scaled = self.scaler.fit_transform(X_train)
            X_val_scaled = self.scaler.transform(X_val)
            X_test_scaled = self.scaler.transform(X_test)
            
            logger.info(f"Data preparation complete!")
            logger.info(f"Training set size: {X_train.shape} (70%)")
            logger.info(f"Validation set size: {X_val.shape} (15%)")
            logger.info(f"Test set size: {X_test.shape} (15%)")
            
            return X_train_scaled, X_val_scaled, X_test_scaled, y_train, y_val, y_test, feature_cols
        else:
            # Original split: train and test only
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, random_state=self.random_state, stratify=y
            )
            
            # Scale features
            X_train_scaled, X_test_scaled = self.scale_features(X_train, X_test)
            
            logger.info(f"Data preparation complete!")
            logger.info(f"Training set size: {X_train.shape}")
            logger.info(f"Test set size: {X_test.shape}")
            
            return X_train_scaled, X_test_scaled, y_train, y_test, feature_cols


def create_synthetic_data(n_samples=1000):
    """Create synthetic predictive maintenance data"""
    np.random.seed(42)
    
    data = {
        'temperature': np.random.normal(70, 15, n_samples),
        'vibration': np.random.normal(0.5, 0.2, n_samples),
        'pressure': np.random.normal(100, 20, n_samples),
        'humidity': np.random.normal(45, 15, n_samples),
        'operational_hours': np.random.uniform(0, 10000, n_samples),
        'error_count': np.random.poisson(5, n_samples),
        'last_maintenance_days': np.random.uniform(0, 365, n_samples),
        'robot_age_months': np.random.uniform(0, 120, n_samples),
        'power_consumption': np.random.normal(500, 100, n_samples),
    }
    
    df = pd.DataFrame(data)
    
    # Create target based on features (failure is more likely with high temp, vibration, etc)
    df['failure'] = (
        (df['temperature'] > 85) |
        (df['vibration'] > 0.8) |
        (df['pressure'] > 120) |
        (df['error_count'] > 10) |
        (df['last_maintenance_days'] > 300)
    ).astype(int)
    
    # Add some noise
    noise_idx = np.random.choice(df.index, size=int(0.1 * len(df)), replace=False)
    df.loc[noise_idx, 'failure'] = 1 - df.loc[noise_idx, 'failure']
    
    return df
