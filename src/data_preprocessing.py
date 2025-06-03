"""
Data preprocessing module for Industrial Machine Failure Prediction
Handles data cleaning, feature engineering, and preparation for ML models
Compatible with existing project structure
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
import joblib
import os
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DataPreprocessor:
    """Class to handle all data preprocessing operations"""
    
    def __init__(self):
        self.scaler = StandardScaler()
        self.label_encoders = {}
        self.imputers = {}
        
    def load_data(self, file_path='data/raw/industrial_iot_dataset.csv'):
        """Load the dataset from CSV file"""
        try:
            if os.path.exists(file_path):
                df = pd.read_csv(file_path)
                logger.info(f"Data loaded successfully from {file_path}. Shape: {df.shape}")
                return df
            else:
                logger.warning(f"File {file_path} not found. Creating synthetic data for testing...")
                return self.create_synthetic_dataset()
        except Exception as e:
            logger.error(f"Error loading data: {e}")
            raise
    
    def create_synthetic_dataset(self):
        """
        Create a synthetic dataset matching the project specifications
        Based on the IoT Industrial Machine requirements from the PDF
        """
        np.random.seed(42)
        n_samples = 10000
        
        # Machine types as specified in the PDF
        common_machines = [
            '3D_Printer', 'AGV', 'Automated_Screwdriver', 'CMM', 'Carton_Former',
            'Compressor', 'Conveyor_Belt', 'Crane', 'Dryer', 'Forklift_Electric',
            'Grinder', 'Labeler', 'Mixer', 'Palletizer', 'Pick_and_Place',
            'Press_Brake', 'Pump', 'Robot_Arm', 'Shrink_Wrapper', 'Shuttle_System',
            'Vacuum_Packer', 'Valve_Controller', 'Vision_System', 'XRay_Inspector'
        ]
        
        special_machines = [
            'Laser_Cutter',  # + Laser_Intensity
            'Hydraulic_Press', 'Injection_Molder',  # + Hydraulic_Pressure_bar
            'CNC_Lathe', 'CNC_Mill', 'Industrial_Chiller',  # + Coolant_Flow_L_min
            'Boiler', 'Furnace', 'Heat_Exchanger'  # + Heat_Index
        ]
        
        all_machines = common_machines + special_machines
        
        # Generate base data with exact column names from PDF
        data = {
            'Machine_ID': [f'M_{i:05d}' for i in range(1, n_samples + 1)],
            'Machine_Type': np.random.choice(all_machines, n_samples),
            'Installation_Year': np.random.randint(2010, 2024, n_samples),
            'Operational_Hours': np.random.normal(5000, 2000, n_samples).clip(100, 15000),
            'Temperature_C': np.random.normal(65, 15, n_samples).clip(20, 120),
            'Vibration_mms': np.random.exponential(2.5, n_samples).clip(0.1, 15),
            'Sound_dB': np.random.normal(70, 12, n_samples).clip(40, 100),
            'Oil_Level_pct': np.random.normal(75, 20, n_samples).clip(10, 100),
            'Coolant_Level_pct': np.random.normal(80, 15, n_samples).clip(20, 100),
            'Power_Consumption_kW': np.random.lognormal(3.5, 0.6, n_samples).clip(5, 150),
            'Last_Maintenance_Days_Ago': np.random.exponential(35, n_samples).clip(0, 365),
            'Maintenance_History_Count': np.random.poisson(4, n_samples),
            'Failure_History_Count': np.random.poisson(1.5, n_samples),
            'AI_Supervision': np.random.choice(['Yes', 'No'], n_samples, p=[0.75, 0.25]),
            'Error_Codes_Last_30_Days': np.random.poisson(2.5, n_samples),
            'AI_Override_Events': np.random.poisson(0.8, n_samples)
        }
        
        df = pd.DataFrame(data)
        
        # Add special features for specific machine types (as per PDF specifications)
        df['Laser_Intensity'] = np.where(
            df['Machine_Type'] == 'Laser_Cutter',
            np.random.uniform(60, 150, n_samples),
            np.nan
        )
        
        df['Hydraulic_Pressure_bar'] = np.where(
            df['Machine_Type'].isin(['Hydraulic_Press', 'Injection_Molder']),
            np.random.uniform(120, 350, n_samples),
            np.nan
        )
        
        df['Coolant_Flow_L_min'] = np.where(
            df['Machine_Type'].isin(['CNC_Lathe', 'CNC_Mill', 'Industrial_Chiller']),
            np.random.uniform(15, 60, n_samples),
            np.nan
        )
        
        df['Heat_Index'] = np.where(
            df['Machine_Type'].isin(['Boiler', 'Furnace', 'Heat_Exchanger']),
            np.random.uniform(0.2, 0.9, n_samples),
            np.nan
        )
        
        # Create realistic failure indicators
        # Calculate failure probability based on multiple factors
        failure_risk = (
            (df['Operational_Hours'] > 10000) * 0.25 +
            (df['Temperature_C'] > 85) * 0.35 +
            (df['Vibration_mms'] > 6) * 0.30 +
            (df['Last_Maintenance_Days_Ago'] > 90) * 0.25 +
            (df['Oil_Level_pct'] < 40) * 0.20 +
            (df['Coolant_Level_pct'] < 50) * 0.15 +
            (df['Error_Codes_Last_30_Days'] > 4) * 0.20 +
            (df['AI_Supervision'] == 'No') * 0.15 +
            (df['Failure_History_Count'] > 3) * 0.30
        ).clip(0, 1)
        
        # Apply some randomness to make it more realistic
        failure_risk = failure_risk * np.random.uniform(0.7, 1.3, n_samples)
        failure_risk = failure_risk.clip(0, 0.95)
        
        df['Failure_Within_7_Days'] = np.random.binomial(1, failure_risk, n_samples)
        
        # Calculate remaining useful life
        # Machines that will fail within 7 days have low remaining life
        base_life = np.random.exponential(120, n_samples).clip(8, 500)
        
        # Adjust based on failure within 7 days
        df['Remaining_Useful_Life_days'] = np.where(
            df['Failure_Within_7_Days'] == 1,
            np.random.randint(1, 8, n_samples),  # 1-7 days if failing soon
            base_life.astype(int)
        )
        
        logger.info(f"Synthetic dataset created with shape: {df.shape}")
        logger.info(f"Failure rate: {df['Failure_Within_7_Days'].mean():.2%}")
        
        return df
    
    def clean_data(self, df):
        """Clean the dataset and handle missing values"""
        logger.info("Starting data cleaning...")
        
        # Remove duplicates if any
        initial_shape = df.shape
        df = df.drop_duplicates()
        if initial_shape[0] != df.shape[0]:
            logger.info(f"Removed {initial_shape[0] - df.shape[0]} duplicate rows")
        
        # Handle outliers using IQR method for numerical columns
        numerical_cols = df.select_dtypes(include=[np.number]).columns
        outlier_cols = [col for col in numerical_cols 
                       if col not in ['Machine_ID', 'Installation_Year', 'Failure_Within_7_Days', 'Remaining_Useful_Life_days']]
        
        for col in outlier_cols:
            if col in df.columns:
                Q1 = df[col].quantile(0.25)
                Q3 = df[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                
                outliers_before = ((df[col] < lower_bound) | (df[col] > upper_bound)).sum()
                df[col] = df[col].clip(lower_bound, upper_bound)
                
                if outliers_before > 0:
                    logger.info(f"Clipped {outliers_before} outliers in {col}")
        
        # Handle missing values
        for col in df.columns:
            missing_count = df[col].isnull().sum()
            if missing_count > 0:
                logger.info(f"Found {missing_count} missing values in {col}")
                
                if df[col].dtype == 'object':
                    # For categorical data, use mode or 'Unknown'
                    mode_value = df[col].mode()[0] if not df[col].mode().empty else 'Unknown'
                    df[col] = df[col].fillna(mode_value)
                else:
                    # For numerical data, use median
                    median_value = df[col].median()
                    df[col] = df[col].fillna(median_value)
        
        logger.info("Data cleaning completed")
        return df
    
    def feature_engineering(self, df):
        """Create new features and encode categorical variables"""
        logger.info("Starting feature engineering...")
        
        # Create machine age feature
        current_year = 2024
        df['Machine_Age_Years'] = current_year - df['Installation_Year']
        
        # Create efficiency and risk indicators
        df['Maintenance_Efficiency'] = df['Maintenance_History_Count'] / (df['Failure_History_Count'] + 1)
        df['Hours_Per_Year'] = df['Operational_Hours'] / (df['Machine_Age_Years'] + 1)
        
        # Create binary risk indicators
        df['High_Temperature_Risk'] = (df['Temperature_C'] > df['Temperature_C'].quantile(0.75)).astype(int)
        df['High_Vibration_Risk'] = (df['Vibration_mms'] > df['Vibration_mms'].quantile(0.75)).astype(int)
        df['Low_Oil_Risk'] = (df['Oil_Level_pct'] < df['Oil_Level_pct'].quantile(0.25)).astype(int)
        df['Overdue_Maintenance'] = (df['Last_Maintenance_Days_Ago'] > 60).astype(int)
        df['High_Error_Rate'] = (df['Error_Codes_Last_30_Days'] > df['Error_Codes_Last_30_Days'].quantile(0.75)).astype(int)
        
        # Create composite risk score
        df['Total_Risk_Score'] = (
            df['High_Temperature_Risk'] + 
            df['High_Vibration_Risk'] + 
            df['Low_Oil_Risk'] + 
            df['Overdue_Maintenance'] +
            df['High_Error_Rate']
        )
        
        # Encode categorical variables
        categorical_cols = df.select_dtypes(include=['object']).columns
        for col in categorical_cols:
            if col not in ['Machine_ID']:  # Don't encode Machine_ID
                le = LabelEncoder()
                df[col + '_encoded'] = le.fit_transform(df[col])
                self.label_encoders[col] = le
                logger.info(f"Encoded {col} with {len(le.classes_)} unique values")
        
        logger.info("Feature engineering completed")
        return df
    
    def prepare_features(self, df):
        """Prepare features for machine learning"""
        logger.info("Preparing features for ML...")
        
        # Define feature columns (excluding targets and ID)
        exclude_cols = ['Machine_ID', 'Failure_Within_7_Days', 'Remaining_Useful_Life_days']
        feature_cols = [col for col in df.columns if col not in exclude_cols]
        
        # Remove original categorical columns if encoded versions exist
        categorical_cols = df.select_dtypes(include=['object']).columns
        for col in categorical_cols:
            if col + '_encoded' in df.columns and col != 'Machine_ID':
                feature_cols.remove(col)
        
        # Prepare feature matrix
        X = df[feature_cols].copy()
        
        # Ensure all features are numerical
        for col in X.columns:
            if X[col].dtype == 'object':
                logger.warning(f"Found non-encoded categorical column {col}, removing...")
                X = X.drop(col, axis=1)
        
        # Prepare targets
        y_classification = df['Failure_Within_7_Days'].copy()
        y_regression = df['Remaining_Useful_Life_days'].copy()
        
        # Scale features
        X_scaled = pd.DataFrame(
            self.scaler.fit_transform(X),
            columns=X.columns,
            index=X.index
        )
        
        logger.info(f"Features prepared: {X_scaled.shape[1]} features, {X_scaled.shape[0]} samples")
        return X_scaled, y_classification, y_regression
    
    def split_data(self, X, y_class, y_reg, test_size=0.2, random_state=42):
        """Split data into train and test sets"""
        # Use stratification for classification target
        X_train, X_test, y_class_train, y_class_test = train_test_split(
            X, y_class, 
            test_size=test_size, 
            random_state=random_state, 
            stratify=y_class
        )
        
        # Split regression target with same indices
        y_reg_train = y_reg.loc[X_train.index]
        y_reg_test = y_reg.loc[X_test.index]
        
        logger.info(f"Data split completed:")
        logger.info(f"  Training set: {X_train.shape[0]} samples")
        logger.info(f"  Test set: {X_test.shape[0]} samples")
        logger.info(f"  Failure rate in train: {y_class_train.mean():.2%}")
        logger.info(f"  Failure rate in test: {y_class_test.mean():.2%}")
        
        return X_train, X_test, y_class_train, y_class_test, y_reg_train, y_reg_test
    
    def save_preprocessor(self, models_dir='data/models'):
        """Save the preprocessor objects"""
        # Create models directory if it doesn't exist
        os.makedirs(models_dir, exist_ok=True)
        
        # Save scaler and encoders
        joblib.dump(self.scaler, os.path.join(models_dir, 'scaler.pkl'))
        joblib.dump(self.label_encoders, os.path.join(models_dir, 'label_encoders.pkl'))
        
        logger.info(f"Preprocessor objects saved to {models_dir}")
    
    def load_preprocessor(self, models_dir='data/models'):
        """Load the preprocessor objects"""
        try:
            self.scaler = joblib.load(os.path.join(models_dir, 'scaler.pkl'))
            self.label_encoders = joblib.load(os.path.join(models_dir, 'label_encoders.pkl'))
            logger.info(f"Preprocessor objects loaded from {models_dir}")
        except FileNotFoundError as e:
            logger.error(f"Preprocessor files not found: {e}")
            raise


def main():
    """Main preprocessing pipeline"""
    logger.info("Starting data preprocessing pipeline...")
    
    # Initialize preprocessor
    preprocessor = DataPreprocessor()
    
    # Load data (will create synthetic if original not found)
    df = preprocessor.load_data()
    
    # Clean data
    df_clean = preprocessor.clean_data(df)
    
    # Feature engineering
    df_engineered = preprocessor.feature_engineering(df_clean)
    
    # Prepare features for ML
    X, y_class, y_reg = preprocessor.prepare_features(df_engineered)
    
    # Split data
    X_train, X_test, y_class_train, y_class_test, y_reg_train, y_reg_test = preprocessor.split_data(
        X, y_class, y_reg
    )
    
    # Save processed data to existing structure
    # Note: Only save to data/processed/ if the directories exist
    processed_dir = 'data/processed'
    if os.path.exists('data') and not os.path.exists(processed_dir):
        os.makedirs(processed_dir, exist_ok=True)
    
    if os.path.exists(processed_dir):
        # Save cleaned and engineered data
        df_engineered.to_csv(os.path.join(processed_dir, 'cleaned_data.csv'), index=False)
        
        # Save train/test splits
        train_data = pd.concat([
            X_train.reset_index(drop=True), 
            y_class_train.reset_index(drop=True), 
            y_reg_train.reset_index(drop=True)
        ], axis=1)
        
        test_data = pd.concat([
            X_test.reset_index(drop=True), 
            y_class_test.reset_index(drop=True), 
            y_reg_test.reset_index(drop=True)
        ], axis=1)
        
        train_data.to_csv(os.path.join(processed_dir, 'train_data.csv'), index=False)
        test_data.to_csv(os.path.join(processed_dir, 'test_data.csv'), index=False)
        
        logger.info(f"Processed data saved to {processed_dir}")
    
    # Save preprocessor objects
    preprocessor.save_preprocessor()
    
    logger.info("Data preprocessing pipeline completed successfully!")
    
    # Print summary statistics
    logger.info(f"\n=== PREPROCESSING SUMMARY ===")
    logger.info(f"Total samples: {len(df_engineered)}")
    logger.info(f"Total features: {X.shape[1]}")
    logger.info(f"Failure rate: {y_class.mean():.2%}")
    logger.info(f"Average remaining life: {y_reg.mean():.1f} days")
    logger.info(f"Machine types: {df_engineered['Machine_Type'].nunique()}")
    
    return preprocessor, X_train, X_test, y_class_train, y_class_test, y_reg_train, y_reg_test


if __name__ == "__main__":
    main()