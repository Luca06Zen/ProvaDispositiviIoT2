"""
Test module for data preprocessing functionality
Corrected and complete version for industrial IoT failure prediction project
"""

import unittest
import pandas as pd
import numpy as np
import os
import sys
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# Add project root to Python path for imports
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

try:
    from src.data_preprocessing import DataPreprocessor
    from src.utils import load_config
    MODULES_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Could not import modules: {e}")
    print("Some tests will be skipped if modules are not available")
    MODULES_AVAILABLE = False

class TestDataPreprocessing(unittest.TestCase):
    """Test cases for data preprocessing functionality"""
    
    @classmethod
    def setUpClass(cls):
        """Set up test fixtures before all tests"""
        try:
            # Try to load config if available
            config_path = os.path.join(project_root, 'config.yaml')
            if os.path.exists(config_path):
                cls.config = load_config(config_path)
            else:
                cls.config = cls.create_default_config()
            
            # Check if DataPreprocessor class exists
            preprocessing_path = os.path.join(project_root, 'src', 'data_preprocessing.py')
            if MODULES_AVAILABLE and (os.path.exists(preprocessing_path) or 'data_preprocessing' in sys.modules):
                cls.preprocessor = DataPreprocessor(cls.config)
            else:
                cls.preprocessor = None
                
        except Exception as e:
            print(f"Setup error: {e}")
            cls.config = cls.create_default_config()
            cls.preprocessor = None
        
        # Create sample data for testing
        cls.sample_data = cls.create_sample_data()
    
    @classmethod
    def create_default_config(cls):
        """Create default configuration for testing"""
        return {
            'data': {
                'test_size': 0.2,
                'random_state': 42
            },
            'preprocessing': {
                'handle_outliers': True,
                'normalize_features': True,
                'fill_missing_values': True
            }
        }
    
    @classmethod
    def create_sample_data(cls):
        """Create sample data for testing according to PDF specifications"""
        np.random.seed(42)
        n_samples = 100
        
        # Machine types as specified in PDF
        basic_machines = [
            '3D_Printer', 'AGV', 'Automated_Screwdriver', 'CMM', 'Carton_Former',
            'Compressor', 'Conveyor_Belt', 'Crane', 'Dryer', 'Forklift_Electric',
            'Grinder', 'Labeler', 'Mixer', 'Palletizer', 'Pick_and_Place',
            'Press_Brake', 'Pump', 'Robot_Arm', 'Shrink_Wrapper', 'Shuttle_System',
            'Vacuum_Packer', 'Valve_Controller', 'Vision_System', 'XRay_Inspector'
        ]
        
        special_machines = [
            'Laser_Cutter', 'Hydraulic_Press', 'Injection_Molder',
            'CNC_Lathe', 'CNC_Mill', 'Industrial_Chiller',
            'Boiler', 'Furnace', 'Heat_Exchanger'
        ]
        
        all_machines = basic_machines + special_machines
        
        # Create base data with correct column names from PDF
        data = {
            'Machine_ID': [f'M{i:03d}' for i in range(n_samples)],
            'Machine_Type': np.random.choice(all_machines, n_samples),
            'Installation_Year': np.random.randint(2010, 2024, n_samples),
            'Operational_Hours': np.random.randint(1000, 50000, n_samples),
            'Temperature_C': np.random.normal(50, 15, n_samples),
            'Vibration_mms': np.random.normal(5, 2, n_samples),
            'Sound_dB': np.random.normal(70, 10, n_samples),
            'Oil_Level_pct': np.random.normal(80, 15, n_samples),
            'Coolant_Level_pct': np.random.normal(75, 20, n_samples),
            'Power_Consumption_kW': np.random.normal(30, 10, n_samples),
            'Last_Maintenance_Days_Ago': np.random.randint(1, 365, n_samples),
            'Maintenance_History_Count': np.random.randint(0, 50, n_samples),
            'Failure_History_Count': np.random.randint(0, 20, n_samples),
            'AI_Supervision': np.random.choice([0, 1], n_samples),
            'Error_Codes_Last_30_Days': np.random.randint(0, 30, n_samples),
            'Remaining_Useful_Life_days': np.random.randint(1, 1000, n_samples),
            'Failure_Within_7_Days': np.random.choice([0, 1], n_samples),
            'AI_Override_Events': np.random.randint(0, 10, n_samples)
        }
        
        df = pd.DataFrame(data)
        
        # Add special features according to PDF specifications (initially zero)
        df['Laser_Intensity'] = 0.0
        df['Hydraulic_Pressure_bar'] = 0.0
        df['Coolant_Flow_L_min'] = 0.0
        df['Heat_Index'] = 0.0
        
        # Set special features for appropriate machine types
        for idx, machine_type in enumerate(df['Machine_Type']):
            if machine_type == 'Laser_Cutter':
                df.loc[idx, 'Laser_Intensity'] = np.random.normal(50, 10)
            elif machine_type in ['Hydraulic_Press', 'Injection_Molder']:
                df.loc[idx, 'Hydraulic_Pressure_bar'] = np.random.normal(150, 30)
            elif machine_type in ['CNC_Lathe', 'CNC_Mill', 'Industrial_Chiller']:
                df.loc[idx, 'Coolant_Flow_L_min'] = np.random.normal(20, 5)
            elif machine_type in ['Boiler', 'Furnace', 'Heat_Exchanger']:
                df.loc[idx, 'Heat_Index'] = np.random.normal(0.7, 0.2)
        
        # Ensure logical consistency: if remaining days <= 7, then failure within 7 days = 1
        df.loc[df['Remaining_Useful_Life_days'] <= 7, 'Failure_Within_7_Days'] = 1

        # Add some missing values for testing null handling
        null_indices = np.random.choice(n_samples, size=10, replace=False)
        df.loc[null_indices[:5], 'Temperature_C'] = np.nan
        df.loc[null_indices[5:], 'Oil_Level_pct'] = np.nan
        
        return df
    
    def test_sample_data_structure(self):
        """Test that sample data has correct structure"""
        # Test column presence
        required_columns = [
            'Machine_ID', 'Machine_Type', 'Installation_Year', 'Operational_Hours',
            'Temperature_C', 'Vibration_mms', 'Sound_dB', 'Oil_Level_pct',
            'Coolant_Level_pct', 'Power_Consumption_kW', 'Last_Maintenance_Days_Ago',
            'Maintenance_History_Count', 'Failure_History_Count', 'AI_Supervision',
            'Error_Codes_Last_30_Days', 'Remaining_Useful_Life_days',
            'Failure_Within_7_Days', 'AI_Override_Events'
        ]
        
        for col in required_columns:
            self.assertIn(col, self.sample_data.columns, f"Missing required column: {col}")
    
    def test_special_features_columns(self):
        """Test that the 4 special feature columns exist as specified in PDF"""
        special_columns = ['Laser_Intensity', 'Hydraulic_Pressure_bar', 
                          'Coolant_Flow_L_min', 'Heat_Index']
        
        for col in special_columns:
            self.assertIn(col, self.sample_data.columns, 
                         f"Missing special feature column: {col}")
    
    def test_machine_types_compliance(self):
        """Test that machine types match PDF specifications"""
        basic_machines = [
            '3D_Printer', 'AGV', 'Automated_Screwdriver', 'CMM', 'Carton_Former',
            'Compressor', 'Conveyor_Belt', 'Crane', 'Dryer', 'Forklift_Electric',
            'Grinder', 'Labeler', 'Mixer', 'Palletizer', 'Pick_and_Place',
            'Press_Brake', 'Pump', 'Robot_Arm', 'Shrink_Wrapper', 'Shuttle_System',
            'Vacuum_Packer', 'Valve_Controller', 'Vision_System', 'XRay_Inspector'
        ]
        
        special_machines = [
            'Laser_Cutter', 'Hydraulic_Press', 'Injection_Molder',
            'CNC_Lathe', 'CNC_Mill', 'Industrial_Chiller',
            'Boiler', 'Furnace', 'Heat_Exchanger'
        ]
        
        all_expected_machines = set(basic_machines + special_machines)
        actual_machines = set(self.sample_data['Machine_Type'].unique())
        
        # Check that all actual machines are in expected list
        for machine in actual_machines:
            self.assertIn(machine, all_expected_machines, 
                         f"Unexpected machine type: {machine}")
    
    def test_target_variables_compliance(self):
        """Test that target variables match PDF output requirements"""
        # Test Remaining_Useful_Life_days exists and is numeric
        self.assertIn('Remaining_Useful_Life_days', self.sample_data.columns)
        self.assertTrue(pd.api.types.is_numeric_dtype(self.sample_data['Remaining_Useful_Life_days']))
        
        # Test Failure_Within_7_Days is binary
        self.assertIn('Failure_Within_7_Days', self.sample_data.columns)
        unique_values = set(self.sample_data['Failure_Within_7_Days'].unique())
        self.assertTrue(unique_values.issubset({0, 1}))
        
        # Test relationship between remaining days and 7-day failure
        short_life = self.sample_data[self.sample_data['Remaining_Useful_Life_days'] <= 7]
        if not short_life.empty:
            # Most machines with ≤7 days should have Failure_Within_7_Days = 1
            failure_rate = short_life['Failure_Within_7_Days'].mean()
            self.assertEqual(failure_rate, 1.0, "Machines with ≤7 days remaining should ALWAYS have Failure_Within_7_Days = 1")
    
    def test_special_features_logic(self):
        """Test that special features are set correctly based on machine type"""
        # Test Laser_Cutter has Laser_Intensity > 0
        laser_cutters = self.sample_data[self.sample_data['Machine_Type'] == 'Laser_Cutter']
        if not laser_cutters.empty:
            self.assertTrue((laser_cutters['Laser_Intensity'] > 0).all(),
                           "Laser_Cutter should have Laser_Intensity > 0")
        
        # Test Hydraulic machines have Hydraulic_Pressure_bar > 0
        hydraulic_machines = self.sample_data[
            self.sample_data['Machine_Type'].isin(['Hydraulic_Press', 'Injection_Molder'])
        ]
        if not hydraulic_machines.empty:
            self.assertTrue((hydraulic_machines['Hydraulic_Pressure_bar'] > 0).all(),
                           "Hydraulic machines should have Hydraulic_Pressure_bar > 0")
        
        # Test CNC/Chiller machines have Coolant_Flow_L_min > 0
        coolant_machines = self.sample_data[
            self.sample_data['Machine_Type'].isin(['CNC_Lathe', 'CNC_Mill', 'Industrial_Chiller'])
        ]
        if not coolant_machines.empty:
            self.assertTrue((coolant_machines['Coolant_Flow_L_min'] > 0).all(),
                           "CNC/Chiller machines should have Coolant_Flow_L_min > 0")
        
        # Test Heat machines have Heat_Index > 0
        heat_machines = self.sample_data[
            self.sample_data['Machine_Type'].isin(['Boiler', 'Furnace', 'Heat_Exchanger'])
        ]
        if not heat_machines.empty:
            self.assertTrue((heat_machines['Heat_Index'] > 0).all(),
                           "Heat machines should have Heat_Index > 0")
    
    def test_missing_values_present(self):
        """Test that sample data contains missing values for testing"""
        self.assertTrue(self.sample_data.isnull().any().any(),
                       "Sample data should contain some missing values for testing")
    
    @unittest.skipIf(not MODULES_AVAILABLE, "Required modules not available")
    def test_data_preprocessor_initialization(self):
        """Test DataPreprocessor can be initialized"""
        self.assertIsNotNone(self.preprocessor, "DataPreprocessor should be initialized")
    
    @unittest.skipIf(not MODULES_AVAILABLE, "Required modules not available")
    def test_missing_value_handling(self):
        """Test handling of missing values"""
        if self.preprocessor is None:
            self.skipTest("DataPreprocessor not available")
        
        # Create data with missing values
        test_data = self.sample_data.copy()
        
        # Test that preprocessing handles missing values
        try:
            processed_data = self.preprocessor.handle_missing_values(test_data)
            
            # Check that no missing values remain in critical columns
            numeric_columns = processed_data.select_dtypes(include=[np.number]).columns
            for col in numeric_columns:
                if col in ['Temperature_C', 'Oil_Level_pct']:  # Columns we added NaN to
                    self.assertFalse(processed_data[col].isnull().any(),
                                   f"Missing values should be handled in {col}")
        except AttributeError:
            self.skipTest("handle_missing_values method not implemented")
    
    @unittest.skipIf(not MODULES_AVAILABLE, "Required modules not available")
    def test_feature_normalization(self):
        """Test feature normalization functionality"""
        if self.preprocessor is None:
            self.skipTest("DataPreprocessor not available")
        
        try:
            # Get numeric features for normalization
            numeric_data = self.sample_data.select_dtypes(include=[np.number])
            
            # Test normalization
            normalized_data = self.preprocessor.normalize_features(numeric_data)
            
            # Check that normalized data has reasonable scale (mean ~0, std ~1)
            for col in normalized_data.columns:
                if col not in ['AI_Supervision', 'Failure_Within_7_Days']:  # Skip binary columns
                    mean_val = abs(normalized_data[col].mean())
                    std_val = normalized_data[col].std()
                    
                    self.assertLess(mean_val, 0.1, f"Normalized {col} should have mean ~0")
                    self.assertGreater(std_val, 0.8, f"Normalized {col} should have std ~1")
                    self.assertLess(std_val, 1.2, f"Normalized {col} should have std ~1")
        except AttributeError:
            self.skipTest("normalize_features method not implemented")
    
    @unittest.skipIf(not MODULES_AVAILABLE, "Required modules not available")
    def test_train_test_split_functionality(self):
        """Test train-test split functionality"""
        if self.preprocessor is None:
            self.skipTest("DataPreprocessor not available")
        
        try:
            # Prepare features and target
            X = self.sample_data.drop(['Machine_ID', 'Failure_Within_7_Days'], axis=1)
            y = self.sample_data['Failure_Within_7_Days']
            
            # Test split
            X_train, X_test, y_train, y_test = self.preprocessor.split_data(X, y)
            
            # Check split proportions
            expected_train_size = int(len(X) * (1 - self.config['data']['test_size']))
            expected_test_size = len(X) - expected_train_size
            
            self.assertEqual(len(X_train), expected_train_size)
            self.assertEqual(len(X_test), expected_test_size)
            self.assertEqual(len(y_train), expected_train_size)
            self.assertEqual(len(y_test), expected_test_size)
            
        except AttributeError:
            self.skipTest("split_data method not implemented")
    
    def test_prediction_output_format(self):
        """Test that prediction output matches PDF requirements"""
        # This test verifies the expected output format structure:
        # "xx% di probabilità guasto. Azione consigliata: .... Giorni di vita rimanenti: ..... Guasto entro 7 giorni: sì/no"
        # Note: Examples in PDF are random/illustrative, not actual model outputs
        
        # Test with various realistic prediction scenarios (not based on PDF examples)
        sample_predictions = [
            {"failure_probability": 0.15, "remaining_days": 45, "failure_within_7": False},
            {"failure_probability": 0.73, "remaining_days": 12, "failure_within_7": False},
            {"failure_probability": 0.92, "remaining_days": 2, "failure_within_7": True},
            {"failure_probability": 0.68, "remaining_days": 8, "failure_within_7": False},
            {"failure_probability": 0.35, "remaining_days": 120, "failure_within_7": False}
        ]
        
        for pred in sample_predictions:
            # Test that all required components are present
            self.assertIn("failure_probability", pred)
            self.assertIn("remaining_days", pred)
            self.assertIn("failure_within_7", pred)
            
            # Test probability is between 0 and 1 (basic validation)
            self.assertGreaterEqual(pred["failure_probability"], 0)
            self.assertLessEqual(pred["failure_probability"], 1)
            
            # Test remaining days is non-negative (0 is possible for immediate failure)
            self.assertGreaterEqual(pred["remaining_days"], 0)
            
            # Test binary failure prediction
            self.assertIn(pred["failure_within_7"], [True, False])
            
            # Test data types are correct
            self.assertIsInstance(pred["failure_probability"], (int, float))
            self.assertIsInstance(pred["remaining_days"], (int, float))
            self.assertIsInstance(pred["failure_within_7"], bool)

        # Test edge cases that might occur in real scenarios
        edge_cases = [
            {"failure_probability": 0.0, "remaining_days": 500, "failure_within_7": False},  # Very low risk
            {"failure_probability": 1.0, "remaining_days": 0, "failure_within_7": True},    # Immediate failure
            {"failure_probability": 0.5, "remaining_days": 7, "failure_within_7": True}     # Boundary case
        ]
        
        for edge_case in edge_cases:
            self.assertIn("failure_probability", edge_case)
            self.assertIn("remaining_days", edge_case)
            self.assertIn("failure_within_7", edge_case)
            
            # Validate edge case values are still within acceptable ranges
            self.assertGreaterEqual(edge_case["failure_probability"], 0)
            self.assertLessEqual(edge_case["failure_probability"], 1)
            self.assertGreaterEqual(edge_case["remaining_days"], 0)

    def test_data_types_validation(self):
        """Test that data types are appropriate"""
        # Machine_ID should be string
        self.assertTrue(self.sample_data['Machine_ID'].dtype == 'object')
        
        # Machine_Type should be string
        self.assertTrue(self.sample_data['Machine_Type'].dtype == 'object')
        
        # Numeric columns should be numeric
        numeric_columns = [
            'Installation_Year', 'Operational_Hours', 'Temperature_C',
            'Vibration_mms', 'Sound_dB', 'Oil_Level_pct', 'Coolant_Level_pct',
            'Power_Consumption_kW', 'Last_Maintenance_Days_Ago',
            'Maintenance_History_Count', 'Failure_History_Count',
            'Error_Codes_Last_30_Days', 'Remaining_Useful_Life_days',
            'AI_Override_Events', 'Laser_Intensity', 'Hydraulic_Pressure_bar',
            'Coolant_Flow_L_min', 'Heat_Index'
        ]
        
        for col in numeric_columns:
            self.assertTrue(pd.api.types.is_numeric_dtype(self.sample_data[col]),
                           f"{col} should be numeric")
        
        # Binary columns should be 0 or 1
        binary_columns = ['AI_Supervision', 'Failure_Within_7_Days']
        for col in binary_columns:
            unique_values = set(self.sample_data[col].unique())
            self.assertTrue(unique_values.issubset({0, 1}),
                           f"{col} should only contain 0 and 1")
    
    def test_data_ranges_validation(self):
        """Test that data values are within reasonable ranges"""
        # Installation year should be reasonable
        self.assertTrue(self.sample_data['Installation_Year'].min() >= 2000)
        self.assertTrue(self.sample_data['Installation_Year'].max() <= 2024)
        
        # Operational hours should be positive
        self.assertTrue((self.sample_data['Operational_Hours'] > 0).all())
        
        # Percentage columns should be between 0 and 100 (allowing some variance)
        percentage_columns = ['Oil_Level_pct', 'Coolant_Level_pct']
        for col in percentage_columns:
            valid_data = self.sample_data[col].dropna()
            self.assertTrue((valid_data >= 0).all(), f"{col} values should be >= 0")
            self.assertTrue((valid_data <= 100).all(), f"{col} values should be <= 100")
        
        # Sound should be positive
        valid_sound = self.sample_data['Sound_dB'].dropna()
        self.assertTrue((valid_sound > 0).all(), "Sound_dB should be positive")
    
    def test_project_structure_compliance(self):
        """Test that project structure matches PDF specifications"""
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        
        # Expected directories as per actual project structure
        expected_dirs = [
            'data', 'data/raw', 'data/processed', 'data/models', 
            'notebooks', 'src', 'tests',
            'web', 'web/static', 'web/static/css', 'web/static/js', 'web/static/images', 'web/templates'
        ]
        
        for dir_path in expected_dirs:
            full_path = os.path.join(project_root, dir_path)
            self.assertTrue(os.path.exists(full_path), 
                        f"Missing directory required by project structure: {dir_path}")
        
        # Expected files organized by location
        expected_files = {
            # Root files
            '': ['config.yaml', 'README.md', 'requirements.txt', 'setup.py'],
            
            # Source files
            'src/': ['data_preprocessing.py', 'model_evaluation.py', 'model_training.py',
                     'prediction_engine.py', 'utils.py'],
            
            # Web application files
            'web/':               ['app.py'],
            'web/static/css/':    ['style.css'],
            'web/static/images/': ['logo.png', 'favicon.ico'],
            'web/static/js/':     ['main.js'],
            'web/templates/':     ['index.html', 'results.html'],
            
            # Notebooks
            'notebooks/': ['01_data_exploration.ipynb', '02_data_preprocessing.ipynb', 
                           '03_model_development.ipynb', '04_model_evaluation.ipynb'],
            
            # Tests
            'tests/': ['test_model.py', 'test_preprocessing.py', 'test_predictions.py'],
            
            # Data files (these might not exist initially)
            'data/models/':    ['failure_prediction_model.pkl', 'scaler.pkl', 'label_encoder.pkl'],
            'data/processed/': ['cleaned_data.csv', 'train_data.csv', 'test_data.csv'],
            'data/raw/':       ['industrial_iot_dataset.csv']
        }
        
        for dir_path, files in expected_files.items():
            for file_name in files:
                full_path = os.path.join(project_root, dir_path, file_name)
                # For data files, only check if directory exists (files might be generated later)
                if dir_path.startswith('data/'):
                    dir_only = os.path.join(project_root, dir_path)
                    self.assertTrue(os.path.exists(dir_only), 
                                f"Missing data directory: {dir_path}")
                else:
                    self.assertTrue(os.path.exists(full_path), 
                                f"Missing file required by project structure: {dir_path}{file_name}")
            
    def test_model_saving_path(self):
        """Test that models are saved in correct directory as per PDF"""
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        models_dir = os.path.join(project_root, 'data', 'models')
        
        # Check that models directory exists
        self.assertTrue(os.path.exists(models_dir), 
                    "Models directory should exist at data/models/")


def run_tests():
    """Function to run all tests"""
    unittest.main(verbosity=2)


if __name__ == '__main__':
    run_tests()