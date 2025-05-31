"""
Data preprocessing module for Industrial Machine Failure Prediction
Handles data cleaning, feature engineering, and preparation for ML models
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
        
    def load_data(self, file_path):
        """Load the dataset from CSV file"""
        try:
            df = pd.read_csv(file_path)
            logger.info(f"Data loaded successfully. Shape: {df.shape}")
            return df
        except Exception as e:
            logger.error(f"Error loading data: {e}")
            raise
    
    def create_synthetic_dataset(self):
        """
        Create a synthetic dataset matching the specified requirements
        Based on the IoT Industrial Machine specifications provided
        """
        np.random.seed(42)
        n_samples = 10000
        
        # Machine types as specified
        common_machines = [
            '3D_Printer', 'AGV', 'Automated_Screwdriver', 'CMM', 'Carton_Former',
            'Compressor', 'Conveyor_Belt', 'Crane', 'Dryer', 'Forklift_Electric',
            'Grinder', 'Labeler', 'Mixer', 'Palletizer', 'Pick_and_Place',
            'Press_Brake', 'Pump', 'Robot_Arm', 'Shrink_Wrapper', 'Shuttle_System',
            'Vacuum_Packer', 'Valve_Controller', 'Vision_System', 'XRay_Inspector'
        ]
        
        special_machines = {
            'Laser_Cutter': 'Laser_Intensity',
            'Hydraulic_Press': 'Hydraulic_Pressure_bar',
            'Injection_Molder': 'Hydraulic_Pressure_bar',
            'CNC_Lathe': 'Coolant_Flow_L_min',
            'CNC_Mill': 'Coolant_Flow_L_min',  
            'Industrial_Chiller': 'Coolant_Flow_L_min',
            'Boiler': 'Heat_Index',
            'Furnace': 'Heat_Index',
            'Heat_Exchanger': 'Heat_Index'
        }
        
        all_machines = common_machines + list(special_machines.keys())
        
        # Generate base data
        data = {
            'Machine_ID': [f'M_{i:05d}' for i in range(1, n_samples + 1)],
            'Machine_Type': np.random.choice(all_machines, n_samples),
            'Installation_Year': np.random.randint(2010, 2024, n_samples),
            'Operational_Hours': np.random.normal(5000, 2000, n_samples).clip(0, 15000),
            'Temperature_C': np.random.normal(65, 15, n_samples).clip(20, 120),
            'Vibration_mms': np.random.exponential(2, n_samples).clip(0, 15),
            'Sound_dB': np.random.normal(70, 10, n_samples).clip(40, 100),
            'Oil_Level_pct': np.random.normal(75, 20, n_samples).clip(0, 100),
            'Coolant_Level_pct': np.random.normal(80, 15, n_samples).clip(0, 100),
            'Power_Consumption_kW': np.random.lognormal(3, 0.5, n_samples).clip(1, 100),
            'Last_Maintenance_Days_Ago': np.random.exponential(30, n_samples).clip(0, 365),
            'Maintenance_History_Count': np.random.poisson(5, n_samples),
            'Failure_History_Count': np.random.poisson(2, n_samples),
            'AI_Supervision': np.random.choice(['Yes', 'No'], n_samples, p=[0.7, 0.3]),
            'Error_Codes_Last_30_Days': np.random.poisson(3, n_samples),
            'AI_Override_Events': np.random.poisson(1, n_samples)
        }
        
        df = pd.DataFrame(data)
        
        # Add special features for specific machines
        df['Laser_Intensity'] = np.where(
            df['Machine_Type'] == 'Laser_Cutter',
            np.random.uniform(50, 100, n_samples),
            np.nan
        )
        
        df['Hydraulic_Pressure_bar'] = np.where(
            df['Machine_Type'].isin(['Hydraulic_Press', 'Injection_Molder']),
            np.random.uniform(100, 300, n_samples),
            np.nan
        )
        
        df['Coolant_Flow_L_min'] = np.where(
            df['Machine_Type'].isin(['CNC_Lathe', 'CNC_Mill', 'Industrial_Chiller']),
            np.random.uniform(10, 50, n_samples),
            np.nan
        )
        
        df['Heat_Index'] = np.where(
            df['Machine_Type'].isin(['Boiler', 'Furnace', 'Heat_Exchanger']),
            np.random.uniform(0.1, 1.0, n_samples),
            np.nan
        )
        
        # Create failure indicators based on realistic conditions
        failure_probability = (
            (df['Operational_Hours'] > 8000) * 0.3 +
            (df['Temperature_C'] > 90) * 0.4 +
            (df['Vibration_mms'] > 8) * 0.3 +
            (df['Last_Maintenance_Days_Ago'] > 90) * 0.2 +
            (df['Oil_Level_pct'] < 30) * 0.3 +
            (df['Error_Codes_Last_30_Days'] > 5) * 0.2
        ).clip(0, 1)
        
        df['Failure_Within_7_Days'] = np.random.binomial(1, failure_probability, n_samples)
        
        # Remaining useful life (inversely related to failure probability)
        df['Remaining_Useful_Life_days'] = np.where(
            df['Failure_Within_7_Days'] == 1,
            np.random.randint(1, 8, n_samples),
            np.random.exponential(60, n_samples).clip(8, 365)
        ).astype(int)
        
        logger.info(f"Synthetic dataset created with shape: {df.shape}")
        return df
    
    def clean_data(self, df):
        """Clean the dataset and handle missing values"""
        logger.info("Starting data cleaning...")
        
        # Remove duplicates
        initial_shape = df.shape
        df = df.drop_duplicates()
        logger.info(f"Removed {initial_shape[0] - df.shape[0]} duplicate rows")
        
        # Handle outliers using IQR method for numerical columns
        numerical_cols = df.select_dtypes(include=[np.number]).columns
        for col in numerical_cols:
            if col not in ['Machine_ID', 'Installation_Year', 'Failure_Within_7_Days']:
                Q1 = df[col].quantile(0.25)
                Q3 = df[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                
                outliers_count = ((df[col] < lower_bound) | (df[col] > upper_bound)).sum()
                df[col] = df[col].clip(lower_bound, upper_bound)
                
                if outliers_count > 0:
                    logger.info(f"Clipped {outliers_count} outliers in {col}")
        
        # Handle missing values
        for col in df.columns:
            missing_count = df[col].isnull().sum()
            if missing_count > 0:
                logger.info(f"Found {missing_count} missing values in {col}")
                
                if df[col].dtype in ['object']:
                    # For categorical data, use mode
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
        
        # Create age of machine
        current_year = 2024
        df['Machine_Age_Years'] = current_year - df['Installation_Year']
        
        # Create efficiency ratios
        df['Maintenance_Efficiency'] = df['Maintenance_History_Count'] / (df['Failure_History_Count'] + 1)
        df['Operating_Efficiency'] = df['Operational_Hours'] / (df['Machine_Age_Years'] + 1)
        
        # Create risk indicators
        df['High_Temperature_Risk'] = (df['Temperature_C'] > 80).astype(int)
        df['High_Vibration_Risk'] = (df['Vibration_mms'] > 5).astype(int)
        df['Low_Oil_Risk'] = (df['Oil_Level_pct'] < 40).astype(int)
        df['Overdue_Maintenance'] = (df['Last_Maintenance_Days_Ago'] > 60).astype(int)
        
        # Combine risk factors
        df['Total_Risk_Score'] = (
            df['High_Temperature_Risk'] + 
            df['High_Vibration_Risk'] + 
            df['Low_Oil_Risk'] + 
            df['Overdue_Maintenance']
        )
        
        # Encode categorical variables
        categorical_cols = df.select_dtypes(include=['object']).columns
        for col in categorical_cols:
            if col not in ['Machine_ID']:  # Don't encode ID
                le = LabelEncoder()
                df[col + '_encoded'] = le.fit_transform(df[col])
                self.label_encoders[col] = le
        
        logger.info("Feature engineering completed")
        return df
    
    def prepare_features(self, df):
        """Prepare features for machine learning"""
        # Define feature columns (excluding target and ID columns)
        feature_cols = [col for col in df.columns if col not in [
            'Machine_ID', 'Failure_Within_7_Days', 'Remaining_Useful_Life_days'
        ]]
        
        # Separate features and targets
        X = df[feature_cols]
        y_classification = df['Failure_Within_7_Days']
        y_regression = df['Remaining_Useful_Life_days']
        
        # Handle any remaining categorical columns
        for col in X.select_dtypes(include=['object']).columns:
            if col not in self.label_encoders:
                le = LabelEncoder()
                X[col + '_encoded'] = le.fit_transform(X[col])
                self.label_encoders[col] = le
            X = X.drop(col, axis=1)
        
        # Scale numerical features
        X_scaled = self.scaler.fit_transform(X)
        X_scaled = pd.DataFrame(X_scaled, columns=X.columns, index=X.index)
        
        return X_scaled, y_classification, y_regression
    
    def split_data(self, X, y_class, y_reg, test_size=0.2, random_state=42):
        """Split data into train and test sets"""
        X_train, X_test, y_class_train, y_class_test = train_test_split(
            X, y_class, test_size=test_size, random_state=random_state, stratify=y_class
        )
        
        _, _, y_reg_train, y_reg_test = train_test_split(
            X, y_reg, test_size=test_size, random_state=random_state, stratify=y_class
        )
        
        logger.info(f"Data split - Train: {X_train.shape}, Test: {X_test.shape}")
        return X_train, X_test, y_class_train, y_class_test, y_reg_train, y_reg_test
    
    def save_preprocessor(self, save_dir):
        """Save the preprocessor objects"""
        os.makedirs(save_dir, exist_ok=True)
        
        joblib.dump(self.scaler, os.path.join(save_dir, 'scaler.pkl'))
        joblib.dump(self.label_encoders, os.path.join(save_dir, 'label_encoders.pkl'))
        
        logger.info(f"Preprocessor saved to {save_dir}")
    
    def load_preprocessor(self, save_dir):
        """Load the preprocessor objects"""
        self.scaler = joblib.load(os.path.join(save_dir, 'scaler.pkl'))
        self.label_encoders = joblib.load(os.path.join(save_dir, 'label_encoders.pkl'))
        
        logger.info(f"Preprocessor loaded from {save_dir}")


def main():
    """Main preprocessing pipeline"""
    preprocessor = DataPreprocessor()
    
    # Create synthetic dataset (since we can't download from Kaggle directly)
    df = preprocessor.create_synthetic_dataset()
    
    # Save raw data
    os.makedirs('data/raw', exist_ok=True)
    df.to_csv('data/raw/industrial_iot_dataset.csv', index=False)
    
    # Clean and preprocess data
    df_clean = preprocessor.clean_data(df)
    df_engineered = preprocessor.feature_engineering(df_clean)
    
    # Prepare features
    X, y_class, y_reg = preprocessor.prepare_features(df_engineered)
    
    # Split data
    X_train, X_test, y_class_train, y_class_test, y_reg_train, y_reg_test = preprocessor.split_data(
        X, y_class, y_reg
    )
    
    # Save processed data
    os.makedirs('data/processed', exist_ok=True)
    df_engineered.to_csv('data/processed/cleaned_data.csv', index=False)
    
    train_data = pd.concat([X_train, y_class_train, y_reg_train], axis=1)
    test_data = pd.concat([X_test, y_class_test, y_reg_test], axis=1)
    
    train_data.to_csv('data/processed/train_data.csv', index=False)
    test_data.to_csv('data/processed/test_data.csv', index=False)
    
    # Save preprocessor
    preprocessor.save_preprocessor('data/models')
    
    logger.info("Preprocessing pipeline completed successfully!")


if __name__ == "__main__":
    main()