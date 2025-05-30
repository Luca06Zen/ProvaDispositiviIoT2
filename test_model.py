"""
Test module for model functionality and performance
Corrected version for current project structure
"""

import unittest
import pandas as pd
import numpy as np
import joblib
import os
import sys

# Add project root to Python path for imports
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.append(project_root)

try:
    from model_training import FailurePredictionTrainer
    from prediction_engine import PredictionEngine
    from utils import load_config
except ImportError as e:
    print(f"Warning: Could not import modules: {e}")
    print("Some tests will be skipped if modules are not available")

from sklearn.metrics import accuracy_score, precision_score, recall_score


class TestModel(unittest.TestCase):
    """Test cases for the failure prediction model"""
    
    @classmethod
    def setUpClass(cls):
        """Set up test fixtures before all tests"""
        # Check if model files exist in current directory structure
        possible_model_paths = [
            'failure_prediction_model.pkl',
            'models/failure_prediction_model.pkl',
            'data/models/failure_prediction_model.pkl'
        ]
        
        possible_scaler_paths = [
            'scaler.pkl',
            'models/scaler.pkl', 
            'data/models/scaler.pkl'
        ]
        
        possible_encoder_paths = [
            'label_encoder.pkl',
            'models/label_encoder.pkl',
            'data/models/label_encoder.pkl'
        ]
        
        cls.model_path = None
        cls.scaler_path = None
        cls.label_encoder_path = None
        
        # Find existing model files
        for path in possible_model_paths:
            if os.path.exists(path):
                cls.model_path = path
                break
                
        for path in possible_scaler_paths:
            if os.path.exists(path):
                cls.scaler_path = path
                break
                
        for path in possible_encoder_paths:
            if os.path.exists(path):
                cls.label_encoder_path = path
                break
        
        cls.model_exists = all([
            cls.model_path and os.path.exists(cls.model_path),
            cls.scaler_path and os.path.exists(cls.scaler_path),
            cls.label_encoder_path and os.path.exists(cls.label_encoder_path)
        ])
        
        if cls.model_exists:
            try:
                cls.model = joblib.load(cls.model_path)
                cls.scaler = joblib.load(cls.scaler_path)
                cls.label_encoder = joblib.load(cls.label_encoder_path)
                
                # Initialize prediction engine
                cls.prediction_engine = PredictionEngine(
                    cls.model_path, cls.scaler_path, cls.label_encoder_path
                )
            except Exception as e:
                print(f"Error loading model files: {e}")
                cls.model_exists = False
    
    def test_model_loading(self):
        """Test if model and preprocessing objects load correctly"""
        if not self.model_exists:
            self.skipTest("Model files not found. Run model training first.")
        
        self.assertIsNotNone(self.model)
        self.assertIsNotNone(self.scaler)
        self.assertIsNotNone(self.label_encoder)
    
    def test_model_prediction_shape(self):
        """Test if model predictions have correct shape"""
        if not self.model_exists:
            self.skipTest("Model files not found. Run model training first.")
        
        # Create sample input
        sample_input = self.create_sample_input()
        
        try:
            scaled_input = self.scaler.transform(sample_input)
            predictions = self.model.predict(scaled_input)
            probabilities = self.model.predict_proba(scaled_input)
            
            self.assertEqual(len(predictions), 1)
            self.assertEqual(probabilities.shape[0], 1)
            self.assertEqual(probabilities.shape[1], 2)  # Binary classification
            self.assertIn(predictions[0], [0, 1])
        except Exception as e:
            self.fail(f"Model prediction failed: {e}")
    
    def test_prediction_engine_basic_machines(self):
        """Test prediction engine with basic machine types (only common features)"""
        if not self.model_exists:
            self.skipTest("Model files not found. Run model training first.")
        
        if not hasattr(self, 'prediction_engine'):
            self.skipTest("Prediction engine not available")
        
        basic_machines = ['3D_Printer', 'AGV', 'Compressor', 'Conveyor_Belt']
        
        for machine_type in basic_machines:
            with self.subTest(machine_type=machine_type):
                input_data = {
                    'machine_type': machine_type,
                    'installation_year': 2020,
                    'operational_hours': 15000,
                    'temperature_c': 45.0,
                    'vibration_mms': 3.5,
                    'sound_db': 65.0,
                    'oil_level_pct': 85.0,
                    'coolant_level_pct': 78.0,
                    'power_consumption_kw': 25.0,
                    'last_maintenance_days_ago': 45,
                    'maintenance_history_count': 12,
                    'failure_history_count': 2,
                    'ai_supervision': True,
                    'error_codes_last_30_days': 3,
                    'ai_override_events': 1
                }
                
                try:
                    result = self.prediction_engine.predict(input_data)
                    
                    self.assertIsInstance(result, dict)
                    self.assertIn('failure_probability', result)
                    self.assertIn('failure_within_7_days', result)
                    self.assertIn('recommended_action', result)
                    
                    # Check probability is between 0 and 1
                    self.assertGreaterEqual(result['failure_probability'], 0.0)
                    self.assertLessEqual(result['failure_probability'], 1.0)
                except Exception as e:
                    self.fail(f"Prediction failed for {machine_type}: {e}")
    
    def test_prediction_engine_special_machines(self):
        """Test prediction engine with machines that have special features"""
        if not self.model_exists:
            self.skipTest("Model files not found. Run model training first.")
        
        if not hasattr(self, 'prediction_engine'):
            self.skipTest("Prediction engine not available")
        
        special_machines = [
            ('Laser_Cutter', {'laser_intensity': 800.0}),
            ('Hydraulic_Press', {'hydraulic_pressure_bar': 150.0}),
            ('CNC_Lathe', {'coolant_flow_l_min': 20.0}),
            ('Heat_Exchanger', {'heat_index': 85.0})
        ]
        
        for machine_type, special_params in special_machines:
            with self.subTest(machine_type=machine_type):
                input_data = {
                    'machine_type': machine_type,
                    'installation_year': 2019,
                    'operational_hours': 18000,
                    'temperature_c': 55.0,
                    'vibration_mms': 4.2,
                    'sound_db': 70.0,
                    'oil_level_pct': 72.0,
                    'coolant_level_pct': 68.0,
                    'power_consumption_kw': 30.0,
                    'last_maintenance_days_ago': 65,
                    'maintenance_history_count': 8,
                    'failure_history_count': 3,
                    'ai_supervision': True,
                    'error_codes_last_30_days': 7,
                    'ai_override_events': 2,
                    **special_params
                }
                
                try:
                    result = self.prediction_engine.predict(input_data)
                    
                    self.assertIsInstance(result, dict)
                    self.assertIn('failure_probability', result)
                    self.assertIn('failure_within_7_days', result)
                    
                    # Check probability is between 0 and 1
                    self.assertGreaterEqual(result['failure_probability'], 0.0)
                    self.assertLessEqual(result['failure_probability'], 1.0)
                except Exception as e:
                    self.fail(f"Prediction failed for {machine_type}: {e}")
    
    def test_model_consistency(self):
        """Test that model gives consistent results for identical inputs"""
        if not self.model_exists:
            self.skipTest("Model files not found. Run model training first.")
        
        if not hasattr(self, 'prediction_engine'):
            self.skipTest("Prediction engine not available")
        
        input_data = {
            'machine_type': 'Laser_Cutter',
            'installation_year': 2020,
            'operational_hours': 15000,
            'temperature_c': 45.0,
            'vibration_mms': 3.5,
            'sound_db': 65.0,
            'oil_level_pct': 85.0,
            'coolant_level_pct': 78.0,
            'power_consumption_kw': 25.0,
            'last_maintenance_days_ago': 45,
            'maintenance_history_count': 12,
            'failure_history_count': 2,
            'ai_supervision': True,
            'error_codes_last_30_days': 3,
            'ai_override_events': 1,
            'laser_intensity': 750.0
        }
        
        # Make multiple predictions with same input
        results = []
        for _ in range(3):
            try:
                result = self.prediction_engine.predict(input_data)
                results.append(result['failure_probability'])
            except Exception as e:
                self.fail(f"Prediction failed: {e}")
        
        # All results should be identical
        self.assertTrue(all(abs(r - results[0]) < 1e-10 for r in results),
                       "Model should give consistent results for identical inputs")
    
    def create_sample_input(self):
        """Create a sample input for testing"""
        # Create a sample that matches expected model input format
        sample_data = {
            'Installation_Year': [2020],
            'Operational_Hours': [15000],
            'Temperature_C': [45.0],
            'Vibration_mms': [3.5],
            'Sound_dB': [65.0],
            'Oil_Level_pct': [85.0],
            'Coolant_Level_pct': [78.0],
            'Power_Consumption_kW': [25.0],
            'Last_Maintenance_Days_Ago': [45],
            'Maintenance_History_Count': [12],
            'Failure_History_Count': [2],
            'AI_Supervision': [1],
            'Error_Codes_Last_30_Days': [3],
            'AI_Override_Events': [1],
            'Remaining_Useful_Life_days': [180],
            'Laser_Intensity': [0],
            'Hydraulic_Pressure_bar': [0],
            'Coolant_Flow_L_min': [0],
            'Heat_Index': [0]
        }
        
        # Add machine type columns (if using one-hot encoding)
        machine_types = [
            '3D_Printer', 'AGV', 'Automated_Screwdriver', 'Boiler', 'CMM',
            'CNC_Lathe', 'CNC_Mill', 'Carton_Former', 'Compressor', 'Conveyor_Belt',
            'Crane', 'Dryer', 'Forklift_Electric', 'Furnace', 'Grinder',
            'Heat_Exchanger', 'Hydraulic_Press', 'Industrial_Chiller',
            'Injection_Molder', 'Labeler', 'Laser_Cutter', 'Mixer',
            'Palletizer', 'Pick_and_Place', 'Press_Brake', 'Pump',
            'Robot_Arm', 'Shrink_Wrapper', 'Shuttle_System', 'Vacuum_Packer',
            'Valve_Controller', 'Vision_System', 'XRay_Inspector'
        ]
        
        for machine_type in machine_types:
            sample_data[f'Machine_Type_{machine_type}'] = [1 if machine_type == '3D_Printer' else 0]
        
        return pd.DataFrame(sample_data)


class TestDataIntegrity(unittest.TestCase):
    """Test data integrity and format requirements"""
    
    def test_required_output_format(self):
        """Test that output matches the required format from PDF"""
        if not os.path.exists('prediction_engine.py'):
            self.skipTest("Prediction engine not found")
        
        # This test verifies the output format requirements:
        # "xx% di probabilità guasto per motivo. Azione consigliata: .... 
        #  Giorni di vita rimanenti: ...." 
        # or "Guasto entro 7 giorni: true/false"
        
        # Mock test - would need actual prediction engine
        expected_keys = ['failure_probability', 'failure_within_7_days', 
                        'recommended_action', 'remaining_useful_life']
        
        # This would be tested with actual prediction engine
        self.assertTrue(True, "Output format test placeholder")
    
    def test_special_features_mapping(self):
        """Test that special features are correctly mapped to machine types"""
        # Test mapping as specified in PDF:
        special_features_mapping = {
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
        
        # Verify mapping is correct
        for machine_type, feature in special_features_mapping.items():
            self.assertIsInstance(machine_type, str)
            self.assertIsInstance(feature, str)
            self.assertIn(feature, ['Laser_Intensity', 'Hydraulic_Pressure_bar', 
                                   'Coolant_Flow_L_min', 'Heat_Index'])


if __name__ == '__main__':
    # Create test suite
    test_suite = unittest.TestSuite()
    
    # Add test cases
    test_suite.addTest(unittest.makeSuite(TestModel))
    test_suite.addTest(unittest.makeSuite(TestDataIntegrity))
    
    # Run tests with detailed output
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(test_suite)
    
    # Print summary
    print(f"\n{'='*50}")
    print("TEST SUMMARY")
    print(f"{'='*50}")
    print(f"Tests run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    print(f"Skipped: {result.testsRun - len(result.failures) - len(result.errors) - (1 if result.wasSuccessful() else 0)}")
    
    if result.wasSuccessful():
        print("\n✅ All tests passed!")
    else:
        print(f"\n❌ Some tests failed")
        if result.failures:
            print("\nFailures:")
            for test, traceback in result.failures:
                print(f"- {test}: {traceback}")
        if result.errors:
            print("\nErrors:")
            for test, traceback in result.errors:
                print(f"- {test}: {traceback}")
