"""
Prediction engine for Industrial Machine Failure Prediction
Handles real-time predictions and result interpretation
"""

import pandas as pd
import numpy as np
import joblib
import os
import logging
from typing import Dict, Any, Tuple
import json

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class PredictionEngine:
    """Class to handle predictions and result interpretation"""
    
    def __init__(self, model_dir='data/models'):
        self.model_dir = model_dir
        self.classification_model = None
        self.regression_model = None
        self.scaler = None
        self.label_encoders = None
        self.model_info = None
        self.feature_columns = None
        
        # Machine type mappings for special features
        self.special_features_mapping = {
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
        
        # Failure cause mapping based on input patterns
        self.failure_causes = {
            'high_temperature': 'Surriscaldamento del sistema',
            'high_vibration': 'Vibrazioni eccessive - possibili problemi meccanici',
            'low_oil': 'Livello olio insufficiente',
            'low_coolant': 'Livello refrigerante insufficiente',
            'overdue_maintenance': 'Manutenzione in ritardo',
            'high_errors': 'Frequenti errori di sistema',
            'combined_factors': 'Combinazione di fattori di rischio multipli'
        }
        
        self.load_models()
    
    def load_models(self):
        """Load all required models and preprocessors"""
        try:
            # Load models
            self.classification_model = joblib.load(
                os.path.join(self.model_dir, 'best_classification_model.pkl')
            )
            self.regression_model = joblib.load(
                os.path.join(self.model_dir, 'best_regression_model.pkl')
            )
            
            # Load preprocessors
            self.scaler = joblib.load(
                os.path.join(self.model_dir, 'scaler.pkl')
            )
            self.label_encoders = joblib.load(
                os.path.join(self.model_dir, 'label_encoders.pkl')
            )
            
            # Load model info
            with open(os.path.join(self.model_dir, 'model_info.json'), 'r') as f:
                self.model_info = json.load(f)
            
            # Get feature columns from a sample processed data file
            try:
                sample_data = pd.read_csv('data/processed/train_data.csv')
                self.feature_columns = [col for col in sample_data.columns if col not in [
                    'Failure_Within_7_Days', 'Remaining_Useful_Life_days'
                ]]
            except:
                # Fallback: define expected feature columns
                self.feature_columns = self._get_expected_feature_columns()
            
            logger.info("Models and preprocessors loaded successfully")
            
        except Exception as e:
            logger.error(f"Error loading models: {e}")
            raise
    
    def _get_expected_feature_columns(self):
        """Get the expected feature columns based on preprocessing"""
        base_features = [
            'Machine_Type_encoded', 'Installation_Year', 'Operational_Hours',
            'Temperature_C', 'Vibration_mms', 'Sound_dB', 'Oil_Level_pct',
            'Coolant_Level_pct', 'Power_Consumption_kW', 'Last_Maintenance_Days_Ago',
            'Maintenance_History_Count', 'Failure_History_Count', 'AI_Supervision_encoded',
            'Error_Codes_Last_30_Days', 'AI_Override_Events', 'Laser_Intensity',
            'Hydraulic_Pressure_bar', 'Coolant_Flow_L_min', 'Heat_Index',
            'Machine_Age_Years', 'Maintenance_Efficiency', 'Operating_Efficiency',
            'High_Temperature_Risk', 'High_Vibration_Risk', 'Low_Oil_Risk',
            'Overdue_Maintenance', 'Total_Risk_Score'
        ]
        return base_features
    
    def preprocess_input(self, input_data: Dict[str, Any]) -> pd.DataFrame:
        """Preprocess input data for prediction"""
        # Create base dataframe
        df = pd.DataFrame([input_data])
        
        # Feature engineering (same as in preprocessing)
        current_year = 2024
        df['Machine_Age_Years'] = current_year - df['Installation_Year']
        df['Maintenance_Efficiency'] = df['Maintenance_History_Count'] / (df['Failure_History_Count'] + 1)
        df['Operating_Efficiency'] = df['Operational_Hours'] / (df['Machine_Age_Years'] + 1)
        
        # Risk indicators
        df['High_Temperature_Risk'] = (df['Temperature_C'] > 80).astype(int)
        df['High_Vibration_Risk'] = (df['Vibration_mms'] > 5).astype(int)
        df['Low_Oil_Risk'] = (df['Oil_Level_pct'] < 40).astype(int)
        df['Overdue_Maintenance'] = (df['Last_Maintenance_Days_Ago'] > 60).astype(int)
        df['Total_Risk_Score'] = (
            df['High_Temperature_Risk'] + df['High_Vibration_Risk'] + 
            df['Low_Oil_Risk'] + df['Overdue_Maintenance']
        )
        
        # Handle special features (set to NaN if not applicable)
        machine_type = input_data['Machine_Type']
        
        special_features = ['Laser_Intensity', 'Hydraulic_Pressure_bar', 'Coolant_Flow_L_min', 'Heat_Index']
        for feature in special_features:
            if feature not in df.columns:
                df[feature] = np.nan
        
        # Set special feature values based on machine type
        if machine_type not in self.special_features_mapping:
            # Machine with common features only
            for feature in special_features:
                df[feature] = np.nan
        else:
            # Machine with special feature
            special_feature = self.special_features_mapping[machine_type]
            for feature in special_features:
                if feature != special_feature:
                    df[feature] = np.nan
        
        # Encode categorical variables
        for col in ['Machine_Type', 'AI_Supervision']:
            if col in self.label_encoders:
                try:
                    df[col + '_encoded'] = self.label_encoders[col].transform([input_data[col]])[0]
                except ValueError:
                    # Handle unseen categories
                    df[col + '_encoded'] = 0
                    logger.warning(f"Unseen category {input_data[col]} for {col}, using default encoding")
        
        # Select and reorder features to match training data
        feature_df = pd.DataFrame()
        for col in self.feature_columns:
            if col in df.columns:
                feature_df[col] = df[col]
            else:
                feature_df[col] = np.nan
        
        # Handle missing values (fill with median/mode as in preprocessing)
        for col in feature_df.columns:
            if feature_df[col].isnull().any():
                if feature_df[col].dtype in ['object']:
                    feature_df[col] = feature_df[col].fillna('Unknown')
                else:
                    feature_df[col] = feature_df[col].fillna(feature_df[col].median())
        
        # Scale features
        feature_scaled = self.scaler.transform(feature_df)
        feature_df_scaled = pd.DataFrame(feature_scaled, columns=feature_df.columns)
        
        return feature_df_scaled
    
    def predict(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Make prediction for given input data"""
        try:
            # Preprocess input
            processed_data = self.preprocess_input(input_data)
            
            # Make predictions
            failure_probability = self.classification_model.predict_proba(processed_data)[0][1]
            failure_prediction = self.classification_model.predict(processed_data)[0]
            remaining_days = max(1, int(self.regression_model.predict(processed_data)[0]))
            
            # Determine failure cause
            failure_cause = self._determine_failure_cause(input_data)
            
            # Generate recommendations
            recommendations = self._generate_recommendations(input_data, failure_probability)
            
            # Format results
            results = {
                'failure_probability': round(failure_probability * 100, 1),
                'failure_within_7_days': bool(failure_prediction),
                'remaining_useful_life_days': remaining_days,
                'failure_cause': failure_cause,
                'recommendations': recommendations,
                'risk_level': self._get_risk_level(failure_probability),
                'input_summary': self._summarize_input(input_data)
            }

            # Generate formatted output string as required
            formatted_output = self._format_output_string(results)
            results['formatted_output'] = formatted_output
            
            return results
            
        except Exception as e:
            logger.error(f"Error making prediction: {e}")
            raise
    
    def _determine_failure_cause(self, input_data: Dict[str, Any]) -> str:
        """Determine the most likely cause of failure based on input patterns"""
        causes = []
        
        if input_data.get('Temperature_C', 0) > 90:
            causes.append('high_temperature')
        
        if input_data.get('Vibration_mms', 0) > 8:
            causes.append('high_vibration')
        
        if input_data.get('Oil_Level_pct', 100) < 30:
            causes.append('low_oil')
        
        if input_data.get('Coolant_Level_pct', 100) < 30:
            causes.append('low_coolant')
        
        if input_data.get('Last_Maintenance_Days_Ago', 0) > 90:
            causes.append('overdue_maintenance')
        
        if input_data.get('Error_Codes_Last_30_Days', 0) > 5:
            causes.append('high_errors')
        
        if len(causes) > 1:
            return self.failure_causes['combined_factors']
        elif len(causes) == 1:
            return self.failure_causes[causes[0]]
        else:
            return "Usura normale del macchinario"
    
    def _generate_recommendations(self, input_data: Dict[str, Any], failure_probability: float) -> list:
        """Generate actionable recommendations based on input data and prediction"""
        recommendations = []
        
        if failure_probability > 0.7:
            recommendations.append("⚠️ URGENTE: Fermare immediatamente la macchina per ispezione")
        
        if input_data.get('Temperature_C', 0) > 90:
            recommendations.append("🌡️ Verificare sistema di raffreddamento")
        
        if input_data.get('Vibration_mms', 0) > 8:
            recommendations.append("🔧 Controllare allineamento e bilanciamento meccanico")
        
        if input_data.get('Oil_Level_pct', 100) < 30:
            recommendations.append("🛢️ Rabboccare olio lubrificante immediatamente")
        
        if input_data.get('Coolant_Level_pct', 100) < 30:
            recommendations.append("💧 Aggiungere liquido refrigerante")
        
        if input_data.get('Last_Maintenance_Days_Ago', 0) > 90:
            recommendations.append("🔨 Programmare manutenzione ordinaria")
        
        if input_data.get('Error_Codes_Last_30_Days', 0) > 5:
            recommendations.append("💻 Analizzare log degli errori per identificare pattern")
        
        if input_data.get('AI_Override_Events', 0) > 3:
            recommendations.append("🤖 Rivedere procedure di supervisione AI")
        
        if not recommendations:
            recommendations.append("✅ Macchina in condizioni normali, continuare monitoraggio")
        
        return recommendations
    
    def _get_risk_level(self, failure_probability: float) -> str:
        """Determine risk level based on failure probability"""
        if failure_probability > 0.8:
            return "CRITICO"
        elif failure_probability > 0.6:
            return "ALTO"
        elif failure_probability > 0.3:
            return "MEDIO"
        else:
            return "BASSO"
    
    def _summarize_input(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Create a summary of the input data for display"""
        return {
            'machine_type': input_data.get('Machine_Type', 'Unknown'),
            'machine_age': 2024 - input_data.get('Installation_Year', 2024),
            'operational_hours': input_data.get('Operational_Hours', 0),
            'temperature': input_data.get('Temperature_C', 0),
            'last_maintenance': input_data.get('Last_Maintenance_Days_Ago', 0)
        }
    
    def batch_predict(self, input_list: list) -> list:
        """Make predictions for multiple inputs"""
        results = []
        for input_data in input_list:
            try:
                result = self.predict(input_data)
                results.append(result)
            except Exception as e:
                logger.error(f"Error predicting for input {input_data}: {e}")
                results.append({'error': str(e)})
        
        return results
    
    def _format_output_string(self, results: Dict[str, Any]) -> str:
        """Format output string as required: 'xx% probabilità guasto. Azione consigliata: .... Giorni di vita rimanenti: ..... Guasto entro 7 giorni: sì/no'"""
        
        # Determine recommended action based on probability and risk level
        probability = results['failure_probability']
        days_remaining = results['remaining_useful_life_days']
        failure_in_7_days = "sì" if results['failure_within_7_days'] else "no"
        
        # Determine action based on probability
        if probability >= 80:
            action = "sostituzione immediata"
        elif probability >= 60:
            action = "manutenzione urgente"
        elif probability >= 30:
            action = "manutenzione programmata"
        elif probability >= 10:
            action = "monitoraggio intensivo"
        else:
            action = "nessuna"
        
        # Format the output string
        formatted_output = (f"{probability}% di probabilità guasto. "
                        f"Azione consigliata: {action}. "
                        f"Giorni di vita rimanenti: {days_remaining}. "
                        f"Guasto entro 7 giorni: {failure_in_7_days}")
        
        return formatted_output


def main():
    """Test the prediction engine with sample data"""
    # Initialize prediction engine
    predictor = PredictionEngine()
    
    # Test cases as specified in requirements
    test_cases = [
        {
            # Test case 1: High risk machine
            'Machine_Type': 'CNC_Lathe',
            'Installation_Year': 2015,
            'Operational_Hours': 12000,
            'Temperature_C': 95,
            'Vibration_mms': 10,
            'Sound_dB': 85,
            'Oil_Level_pct': 25,
            'Coolant_Level_pct': 30,
            'Power_Consumption_kW': 45,
            'Last_Maintenance_Days_Ago': 120,
            'Maintenance_History_Count': 8,
            'Failure_History_Count': 3,
            'AI_Supervision': 'Yes',
            'Error_Codes_Last_30_Days': 7,
            'AI_Override_Events': 2,
            'Coolant_Flow_L_min': 25
        },
        {
            # Test case 2: Medium risk machine
            'Machine_Type': 'Robot_Arm',
            'Installation_Year': 2020,
            'Operational_Hours': 6000,
            'Temperature_C': 70,
            'Vibration_mms': 3,
            'Sound_dB': 65,
            'Oil_Level_pct': 60,
            'Coolant_Level_pct': 70,
            'Power_Consumption_kW': 20,
            'Last_Maintenance_Days_Ago': 45,
            'Maintenance_History_Count': 5,
            'Failure_History_Count': 1,
            'AI_Supervision': 'Yes',
            'Error_Codes_Last_30_Days': 2,
            'AI_Override_Events': 0
        },
        {
            # Test case 3: Low risk machine
            'Machine_Type': 'Conveyor_Belt',
            'Installation_Year': 2022,
            'Operational_Hours': 2000,
            'Temperature_C': 50,
            'Vibration_mms': 1,
            'Sound_dB': 55,
            'Oil_Level_pct': 85,
            'Coolant_Level_pct': 90,
            'Power_Consumption_kW': 15,
            'Last_Maintenance_Days_Ago': 20,
            'Maintenance_History_Count': 2,
            'Failure_History_Count': 0,
            'AI_Supervision': 'Yes',
            'Error_Codes_Last_30_Days': 0,
            'AI_Override_Events': 0
        }
    ]
    
    # Make predictions
    logger.info("Testing prediction engine with sample data...")
    
    for i, test_case in enumerate(test_cases, 1):
        logger.info(f"\n--- Test Case {i} ---")
        logger.info(f"Machine: {test_case['Machine_Type']}")
        
        try:
            result = predictor.predict(test_case)

            # Show formatted output as required
            logger.info(f"OUTPUT: {result['formatted_output']}")
            logger.info(f"Risk Level: {result['risk_level']}")
            logger.info(f"Probable cause: {result['failure_cause']}")
            logger.info("Recommendations:")
            for rec in result['recommendations']:
                logger.info(f"  - {rec}")
        
        except Exception as e:
            logger.error(f"Error in test case {i}: {e}")
    
    logger.info("Prediction engine testing completed!")


if __name__ == "__main__":
    main()