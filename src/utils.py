"""
Utility functions for IoT Predictive Maintenance System
Contains helper functions, data validation, and common operations
"""

import pandas as pd
import numpy as np
import os
import pickle
import json
import yaml
from datetime import datetime
import logging

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Device configuration mapping
DEVICE_CONFIG = {
    'Laser_Cutter': {
        'additional_field': 'Laser_Intensity',
        'field_type': 'numeric',
        'min_value': 0,
        'max_value': 100,
        'description': 'Dispositivo per taglio laser che richiede monitoraggio dell\'intensità laser'
    },
    'Hydraulic_Press': {
        'additional_field': 'Hydraulic_Pressure_bar',
        'field_type': 'numeric',
        'min_value': 0,
        'max_value': 500,
        'description': 'Pressa idraulica che richiede monitoraggio della pressione idraulica'
    },
    'Injection_Molder': {
        'additional_field': 'Hydraulic_Pressure_bar',
        'field_type': 'numeric',
        'min_value': 0,
        'max_value': 500,
        'description': 'Macchina per stampaggio a iniezione con controllo pressione idraulica'
    },
    'CNC_Lathe': {
        'additional_field': 'Coolant_Flow_L_min',
        'field_type': 'numeric',
        'min_value': 0,
        'max_value': 50,
        'description': 'Tornio CNC con sistema di raffreddamento monitorato'
    },
    'CNC_Mill': {
        'additional_field': 'Coolant_Flow_L_min',
        'field_type': 'numeric',
        'min_value': 0,
        'max_value': 50,
        'description': 'Fresatrice CNC con controllo portata refrigerante'
    },
    'Industrial_Chiller': {
        'additional_field': 'Coolant_Flow_L_min',
        'field_type': 'numeric',
        'min_value': 0,
        'max_value': 100,
        'description': 'Sistema di raffreddamento industriale'
    },
    'Boiler': {
        'additional_field': 'Heat_Index',
        'field_type': 'numeric',
        'min_value': 0,
        'max_value': 100,
        'description': 'Caldaia industriale con monitoraggio indice di calore'
    },
    'Furnace': {
        'additional_field': 'Heat_Index',
        'field_type': 'numeric',
        'min_value': 0,
        'max_value': 100,
        'description': 'Forno industriale ad alta temperatura'
    },
    'Heat_Exchanger': {
        'additional_field': 'Heat_Index',
        'field_type': 'numeric',
        'min_value': 0,
        'max_value': 100,
        'description': 'Scambiatore di calore con controllo termico'
    }
}

# Common devices (without additional fields)
COMMON_DEVICES = [
    '3D_Printer', 'AGV', 'Automated_Screwdriver', 'CMM', 'Carton_Former',
    'Compressor', 'Conveyor_Belt', 'Crane', 'Dryer', 'Forklift_Electric',
    'Grinder', 'Labeler', 'Mixer', 'Palletizer', 'Pick_and_Place',
    'Press_Brake', 'Pump', 'Robot_Arm', 'Shrink_Wrapper', 'Shuttle_System',
    'Vacuum_Packer', 'Valve_Controller', 'Vision_System', 'XRay_Inspector'
]

# Field validation ranges
FIELD_RANGES = {
    'Installation_Year': {'min': 2000, 'max': 2025},
    'Operational_Hours': {'min': 0, 'max': 100000},
    'Temperature_C': {'min': -50, 'max': 200},
    'Vibration_mms': {'min': 0, 'max': 50},
    'Sound_dB': {'min': 0, 'max': 150},
    'Oil_Level_pct': {'min': 0, 'max': 100},
    'Coolant_Level_pct': {'min': 0, 'max': 100},
    'Power_Consumption_kW': {'min': 0, 'max': 1000},
    'Last_Maintenance_Days_Ago': {'min': 0, 'max': 365},
    'Maintenance_History_Count': {'min': 0, 'max': 100},
    'Failure_History_Count': {'min': 0, 'max': 50},
    'Error_Codes_Last_30_Days': {'min': 0, 'max': 100},
    'AI_Override_Events': {'min': 0, 'max': 50}
}

def ensure_directory_exists(directory_path):
    """Create directory if it doesn't exist"""
    if not os.path.exists(directory_path):
        os.makedirs(directory_path)
        logger.info(f"Created directory: {directory_path}")

def save_pickle(obj, filepath):
    """Save object to pickle file"""
    ensure_directory_exists(os.path.dirname(filepath))
    with open(filepath, 'wb') as f:
        pickle.dump(obj, f)
    logger.info(f"Object saved to {filepath}")

def load_pickle(filepath):
    """Load object from pickle file"""
    try:
        with open(filepath, 'rb') as f:
            obj = pickle.load(f)
        logger.info(f"Object loaded from {filepath}")
        return obj
    except FileNotFoundError:
        logger.error(f"File not found: {filepath}")
        raise
    except Exception as e:
        logger.error(f"Error loading pickle file: {str(e)}")
        raise

def save_json(data, filepath):
    """Save data to JSON file"""
    ensure_directory_exists(os.path.dirname(filepath))
    with open(filepath, 'w') as f:
        json.dump(data, f, indent=2)
    logger.info(f"JSON data saved to {filepath}")

def load_json(filepath):
    """Load data from JSON file"""
    try:
        with open(filepath, 'r') as f:
            data = json.load(f)
        logger.info(f"JSON data loaded from {filepath}")
        return data
    except FileNotFoundError:
        logger.error(f"File not found: {filepath}")
        raise
    except Exception as e:
        logger.error(f"Error loading JSON file: {str(e)}")
        raise

def load_config(config_path='config.yaml'):
    """Load configuration from YAML file"""
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        logger.info(f"Configuration loaded from {config_path}")
        return config
    except FileNotFoundError:
        logger.warning(f"Config file not found: {config_path}")
        return {}
    except Exception as e:
        logger.error(f"Error loading config: {str(e)}")
        return {}

def validate_input_data(data, machine_type):
    """Validate input data for predictions"""
    errors = []
    
    # Check if machine type is valid
    all_devices = COMMON_DEVICES + list(DEVICE_CONFIG.keys())
    if machine_type not in all_devices:
        errors.append(f"Invalid machine type: {machine_type}")
    
    # Validate common fields
    for field, ranges in FIELD_RANGES.items():
        if field in data:
            value = data[field]
            if not isinstance(value, (int, float)):
                try:
                    value = float(value)
                except (TypeError, ValueError):
                    errors.append(f"Invalid value for {field}: must be numeric")
                    continue
            
            if value < ranges['min'] or value > ranges['max']:
                errors.append(f"Value for {field} out of range: {ranges['min']}-{ranges['max']}")
    
    # Validate AI_Supervision (binary field)
    if 'AI_Supervision' in data:
        if data['AI_Supervision'] not in [0, 1, '0', '1']:
            errors.append("AI_Supervision must be 0 or 1")
    
    # Validate additional fields for special devices
    if machine_type in DEVICE_CONFIG:
        additional_field = DEVICE_CONFIG[machine_type]['additional_field']
        if additional_field not in data or data[additional_field] is None:
            errors.append(f"Missing required field for {machine_type}: {additional_field}")
        else:
            config = DEVICE_CONFIG[machine_type]
            try:
                value = float(data[additional_field])
                if value < config['min_value'] or value > config['max_value']:
                    errors.append(f"{additional_field} out of range: {config['min_value']}-{config['max_value']}")
            except (TypeError, ValueError):
                errors.append(f"Invalid value for {additional_field}: must be numeric")
    
    return errors

def prepare_input_for_prediction(input_data, machine_type):
    """Prepare input data for model prediction"""
    # Create a copy to avoid modifying original data
    data = input_data.copy()
    
    # Ensure all common fields are present with default values if missing
    common_fields = [
        'Installation_Year', 'Operational_Hours', 'Temperature_C', 'Vibration_mms',
        'Sound_dB', 'Oil_Level_pct', 'Coolant_Level_pct', 'Power_Consumption_kW',
        'Last_Maintenance_Days_Ago', 'Maintenance_History_Count', 'Failure_History_Count',
        'AI_Supervision', 'Error_Codes_Last_30_Days', 'AI_Override_Events'
    ]
    
    for field in common_fields:
        if field not in data:
            # Set reasonable defaults
            if field == 'AI_Supervision':
                data[field] = 1  # Assume AI supervision is active by default
            elif 'pct' in field:
                data[field] = 50.0  # Default percentage values
            else:
                data[field] = 0.0
    
    # Handle additional fields
    additional_fields = ['Laser_Intensity', 'Hydraulic_Pressure_bar', 'Coolant_Flow_L_min', 'Heat_Index']
    
    if machine_type in DEVICE_CONFIG:
        required_field = DEVICE_CONFIG[machine_type]['additional_field']
        # Set the required additional field
        if required_field not in data:
            data[required_field] = 0.0
        
        # Set other additional fields to 0
        for field in additional_fields:
            if field != required_field and field not in data:
                data[field] = 0.0
    else:
        # For common devices, set all additional fields to 0
        for field in additional_fields:
            if field not in data:
                data[field] = 0.0
    
    return data

def generate_risk_assessment(prediction_proba, machine_type, input_data):
    """Generate risk assessment and recommendations"""
    risk_score = prediction_proba * 100
    
    # Determine risk level
    if risk_score >= 70:
        risk_level = "ALTO"
        risk_class = "risk-high"
    elif risk_score >= 40:
        risk_level = "MEDIO"
        risk_class = "risk-medium"
    else:
        risk_level = "BASSO"
        risk_class = "risk-low"
    
    # Identify risk factors
    risk_factors = []
    
    if input_data.get('Last_Maintenance_Days_Ago', 0) > 30:
        risk_factors.append("Manutenzione non recente")
    
    if input_data.get('Failure_History_Count', 0) > 2:
        risk_factors.append("Storico guasti elevato")
    
    if input_data.get('Vibration_mms', 0) > 10:
        risk_factors.append("Vibrazioni eccessive")
    
    if input_data.get('Oil_Level_pct', 100) < 20:
        risk_factors.append("Livello olio basso")
    
    if input_data.get('Error_Codes_Last_30_Days', 0) > 5:
        risk_factors.append("Errori frequenti")
    
    if input_data.get('Temperature_C', 25) > 80:
        risk_factors.append("Temperatura elevata")
    
    if not risk_factors:
        risk_factors = ["Parametri nella norma"]
    
    # Generate recommendations
    recommendations = []
    
    for factor in risk_factors:
        if factor == "Manutenzione non recente":
            recommendations.append("Programmare manutenzione ordinaria entro 7 giorni")
        elif factor == "Storico guasti elevato":
            recommendations.append("Analisi approfondita delle cause ricorrenti di guasto")
        elif factor == "Vibrazioni eccessive":
            recommendations.append("Controllo bilanciamento e allineamento componenti")
        elif factor == "Livello olio basso":
            recommendations.append("Rabbocco immediato dell'olio lubrificante")
        elif factor == "Errori frequenti":
            recommendations.append("Revisione software e sensori")
        elif factor == "Temperatura elevata":
            recommendations.append("Verifica sistema di raffreddamento")
        else:
            recommendations.append("Continuare monitoraggio regolare")
    
    # Calculate estimated remaining days
    remaining_days = max(1, int(90 - risk_score))
    failure_within_7_days = risk_score > 70
    
    return {
        'risk_score': risk_score,
        'risk_level': risk_level,
        'risk_class': risk_class,
        'risk_factors': risk_factors,
        'recommendations': recommendations,
        'remaining_days': remaining_days,
        'failure_within_7_days': failure_within_7_days
    }

def format_prediction_output(risk_assessment):
    """Format prediction output for display"""
    risk_score = risk_assessment['risk_score']
    main_risk_factor = risk_assessment['risk_factors'][0]
    
    if risk_assessment['failure_within_7_days']:
        output = f"{risk_score:.1f}% di probabilità di guasto per {main_risk_factor.lower()}. "
        output += "⚠️ Guasto entro 7 giorni. "
        output += "Azione consigliata: Intervento immediato richiesto"
    else:
        output = f"{risk_score:.1f}% di probabilità di guasto per {main_risk_factor.lower()}. "
        output += f"Azione consigliata: Pianificare manutenzione preventiva. "
        output += f"Giorni di vita rimanenti stimati: ~{risk_assessment['remaining_days']} giorni"
    
    return output

def log_prediction(input_data, prediction_result, machine_type):
    """Log prediction for monitoring and analysis"""
    log_entry = {
        'timestamp': datetime.now().isoformat(),
        'machine_type': machine_type,
        'input_data': input_data,
        'prediction_result': prediction_result
    }
    
    # Ensure logs directory exists
    ensure_directory_exists('logs')
    
    # Save to daily log file
    log_filename = f"logs/predictions_{datetime.now().strftime('%Y-%m-%d')}.json"
    
    try:
        # Load existing logs
        if os.path.exists(log_filename):
            with open(log_filename, 'r') as f:
                logs = json.load(f)
        else:
            logs = []
        
        # Add new log entry
        logs.append(log_entry)
        
        # Save updated logs
        with open(log_filename, 'w') as f:
            json.dump(logs, f, indent=2)
        
        logger.info(f"Prediction logged to {log_filename}")
    except Exception as e:
        logger.error(f"Error logging prediction: {str(e)}")

def get_device_info(machine_type):
    """Get device information and configuration"""
    if machine_type in DEVICE_CONFIG:
        return DEVICE_CONFIG[machine_type]
    elif machine_type in COMMON_DEVICES:
        return {
            'additional_field': None,
            'description': f'Dispositivo IoT industriale: {machine_type}'
        }
    else:
        return None

def calculate_model_confidence(prediction_proba):
    """Calculate model confidence based on prediction probability"""
    # Confidence is higher when probability is closer to 0 or 1
    confidence = abs(2 * prediction_proba - 1)
    
    if confidence >= 0.8:
        return "Alta"
    elif confidence >= 0.6:
        return "Media"
    else:
        return "Bassa"

def format_number(value, decimal_places=2):
    """Format number for display"""
    if isinstance(value, (int, float)):
        return f"{value:.{decimal_places}f}"
    return str(value)

def convert_to_display_format(data):
    """Convert data to user-friendly display format"""
    display_data = {}
    
    field_mappings = {
        'Machine_Type': 'Tipo di Dispositivo',
        'Installation_Year': 'Anno di Installazione',
        'Operational_Hours': 'Ore Operative',
        'Temperature_C': 'Temperatura (°C)',
        'Vibration_mms': 'Vibrazione (mm/s)',
        'Sound_dB': 'Suono (dB)',
        'Oil_Level_pct': 'Livello Olio (%)',
        'Coolant_Level_pct': 'Livello Refrigerante (%)',
        'Power_Consumption_kW': 'Consumo Energetico (kW)',
        'Last_Maintenance_Days_Ago': 'Giorni dall\'ultima Manutenzione',
        'Maintenance_History_Count': 'Numero Manutenzioni',
        'Failure_History_Count': 'Numero Guasti',
        'AI_Supervision': 'Supervisione AI',
        'Error_Codes_Last_30_Days': 'Errori ultimi 30 giorni',
        'AI_Override_Events': 'Allarmi AI Ignorati',
        'Laser_Intensity': 'Intensità Laser',
        'Hydraulic_Pressure_bar': 'Pressione Idraulica (bar)',
        'Coolant_Flow_L_min': 'Flusso Refrigerante (L/min)',
        'Heat_Index': 'Indice di Calore'
    }
    
    for key, value in data.items():
        display_key = field_mappings.get(key, key)
        if isinstance(value, (int, float)):
            display_data[display_key] = format_number(value)
        else:
            display_data[display_key] = str(value)
    
    return display_data

def validate_machine_type(machine_type):
    """Validate if machine type is supported"""
    all_devices = COMMON_DEVICES + list(DEVICE_CONFIG.keys())
    return machine_type in all_devices

def get_required_fields(machine_type):
    """Get required fields for a specific machine type"""
    required_fields = [
        'Installation_Year', 'Operational_Hours', 'Temperature_C', 'Vibration_mms',
        'Sound_dB', 'Oil_Level_pct', 'Coolant_Level_pct', 'Power_Consumption_kW',
        'Last_Maintenance_Days_Ago', 'Maintenance_History_Count', 'Failure_History_Count',
        'AI_Supervision', 'Error_Codes_Last_30_Days', 'AI_Override_Events'
    ]
    
    if machine_type in DEVICE_CONFIG:
        additional_field = DEVICE_CONFIG[machine_type]['additional_field']
        required_fields.append(additional_field)
    
    return required_fields

def create_sample_data(machine_type):
    """Create sample data for testing"""
    sample_data = {
        'Installation_Year': 2020,
        'Operational_Hours': 5000,
        'Temperature_C': 45,
        'Vibration_mms': 2.5,
        'Sound_dB': 65,
        'Oil_Level_pct': 75,
        'Coolant_Level_pct': 80,
        'Power_Consumption_kW': 15,
        'Last_Maintenance_Days_Ago': 25,
        'Maintenance_History_Count': 3,
        'Failure_History_Count': 1,
        'AI_Supervision': 1,
        'Error_Codes_Last_30_Days': 2,
        'AI_Override_Events': 0
    }
    
    if machine_type in DEVICE_CONFIG:
        additional_field = DEVICE_CONFIG[machine_type]['additional_field']
        if additional_field == 'Laser_Intensity':
            sample_data['Laser_Intensity'] = 75
        elif additional_field == 'Hydraulic_Pressure_bar':
            sample_data['Hydraulic_Pressure_bar'] = 250
        elif additional_field == 'Coolant_Flow_L_min':
            sample_data['Coolant_Flow_L_min'] = 25
        elif additional_field == 'Heat_Index':
            sample_data['Heat_Index'] = 60
    
    return sample_data