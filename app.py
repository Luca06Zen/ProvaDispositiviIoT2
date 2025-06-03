"""
Flask Web Application for IoT Predictive Maintenance System
Provides web interface for device failure prediction
"""

import os
import sys
from pathlib import Path
from flask import Flask, render_template, request, jsonify, session, redirect, url_for
from werkzeug.security import generate_password_hash, check_password_hash
import json
from datetime import datetime
import logging
import pickle
import pandas as pd
import numpy as np

# Aggiungere il percorso src per gli import
current_dir = Path(__file__).parent
src_dir = current_dir / 'src'
sys.path.insert(0, str(src_dir))

# Configurazione dispositivi
COMMON_DEVICES = [
    '3D_Printer', 'AGV', 'Automated_Screwdriver', 'CMM', 'Carton_Former', 
    'Compressor', 'Conveyor_Belt', 'Crane', 'Dryer', 'Forklift_Electric', 
    'Grinder', 'Labeler', 'Mixer', 'Palletizer', 'Pick_and_Place', 
    'Press_Brake', 'Pump', 'Robot_Arm', 'Shrink_Wrapper', 'Shuttle_System',
    'Vacuum_Packer', 'Valve_Controller', 'Vision_System', 'XRay_Inspector'
]

DEVICE_CONFIG = {
    'Laser_Cutter': ['Laser_Intensity'],
    'Hydraulic_Press': ['Hydraulic_Pressure_bar'],
    'Injection_Molder': ['Hydraulic_Pressure_bar'],
    'CNC_Lathe': ['Coolant_Flow_L_min'],
    'CNC_Mill': ['Coolant_Flow_L_min'],
    'Industrial_Chiller': ['Coolant_Flow_L_min'],
    'Boiler': ['Heat_Index'],
    'Furnace': ['Heat_Index'],
    'Heat_Exchanger': ['Heat_Index']
}

# Traduzione nomi campi per display
FIELD_TRANSLATIONS = {
    'Installation_Year': 'Anno di installazione',
    'Operational_Hours': 'Ore operative',
    'Temperature_C': 'Temperatura (°C)',
    'Vibration_mms': 'Vibrazione (mm/s)',
    'Sound_dB': 'Suono (dB)',
    'Oil_Level_pct': 'Livello olio (%)',
    'Coolant_Level_pct': 'Livello refrigerante (%)',
    'Power_Consumption_kW': 'Consumo energetico (kW)',
    'Last_Maintenance_Days_Ago': 'Giorni dall\'ultima manutenzione',
    'Maintenance_History_Count': 'Numero manutenzioni',
    'Failure_History_Count': 'Numero guasti',
    'AI_Supervision': 'Supervisione AI',
    'Error_Codes_Last_30_Days': 'Errori ultimi 30 giorni',
    'AI_Override_Events': 'Allarmi AI ignorati',
    'Laser_Intensity': 'Intensità laser',
    'Hydraulic_Pressure_bar': 'Pressione idraulica (bar)',
    'Coolant_Flow_L_min': 'Flusso refrigerante (L/min)',
    'Heat_Index': 'Indice di calore'
}

try:
    from prediction_engine import PredictionEngine
    from utils import load_config
    prediction_engine_imported = True
except ImportError as e:
    print(f"Import error: {e}")
    prediction_engine_imported = False
    
    # Fallback - creare una classe dummy
    class PredictionEngine:
        def __init__(self, model_dir='data/models'):  # CORREZIONE: percorso corretto
            self.model_dir = model_dir
            self.model = None
            self.scaler = None
            
        def load_models(self):
            try:
                # Usare il percorso corretto per i modelli
                models_dir = Path(self.model_dir)
                if not models_dir.exists():
                    print(f"Models directory not found: {models_dir}")
                    return False
                
                model_path = models_dir / 'best_classification_model.pkl'
                scaler_path = models_dir / 'scaler.pkl'
                
                if model_path.exists():
                    with open(model_path, 'rb') as f:
                        self.model = pickle.load(f)
                    print(f"Model loaded from: {model_path}")
                else:
                    print(f"Model file not found: {model_path}")
                    return False
                
                if scaler_path.exists():
                    with open(scaler_path, 'rb') as f:
                        self.scaler = pickle.load(f)
                    print(f"Scaler loaded from: {scaler_path}")
                else:
                    print(f"Scaler file not found: {scaler_path}")
                    
                return True
            except Exception as e:
                print(f"Error loading model: {e}")
                return False
        
        def predict(self, data):
            # Fallback prediction logic
            if self.model is None:
                # Simulazione per testing
                failure_prob = np.random.random()
                return {
                    'failure_probability': failure_prob * 100,
                    'failure_within_7_days': failure_prob > 0.7,
                    'remaining_useful_life_days': int(np.random.uniform(1, 365)),
                    'formatted_output': f"{int(failure_prob * 100)}% di probabilità guasto. Azione consigliata: {'sostituzione' if failure_prob > 0.7 else 'monitoraggio'}. Giorni di vita rimanenti: {int(np.random.uniform(1, 365))}. Guasto entro 7 giorni: {'sì' if failure_prob > 0.7 else 'no'}"
                }
            
            # Logica reale se il modello è caricato
            try:
                # Preparare i dati per la predizione
                input_df = pd.DataFrame([data])
                
                # Normalizzare se necessario
                if self.scaler:
                    input_scaled = self.scaler.transform(input_df)
                else:
                    input_scaled = input_df
                
                # Predizione
                failure_prob = self.model.predict_proba(input_scaled)[0][1]
                failure_within_7_days = failure_prob > 0.7
                
                # Stima giorni rimanenti (logica semplificata)
                remaining_days = int(max(1, 365 * (1 - failure_prob)))
                
                return {
                    'failure_probability': failure_prob * 100,
                    'failure_within_7_days': failure_within_7_days,
                    'remaining_useful_life_days': remaining_days,
                    'formatted_output': f"{int(failure_prob * 100)}% di probabilità guasto. Azione consigliata: {'sostituzione' if failure_prob > 0.7 else 'monitoraggio'}. Giorni di vita rimanenti: {remaining_days}. Guasto entro 7 giorni: {'sì' if failure_within_7_days else 'no'}"
                }
            except Exception as e:
                print(f"Prediction error: {e}")
                return {
                    'failure_probability': 50.0,
                    'failure_within_7_days': False,
                    'remaining_useful_life_days': 100,
                    'formatted_output': "50% di probabilità guasto. Azione consigliata: monitoraggio. Giorni di vita rimanenti: 100. Guasto entro 7 giorni: no"
                }

app = Flask(__name__, 
           template_folder='web/templates',
           static_folder='web/static')
app.secret_key = 'your-secret-key-change-in-production'

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize prediction engine
prediction_engine = None

def init_prediction_engine():
    """Initialize prediction engine"""
    global prediction_engine
    try:
        if prediction_engine_imported:
            # Usa la classe reale con il percorso corretto
            prediction_engine = PredictionEngine(model_dir='data/models')
        else:
            # Usa la classe fallback con il percorso corretto
            prediction_engine = PredictionEngine(model_dir='data/models')
            
        success = prediction_engine.load_models()
        if success:
            logger.info("Prediction engine initialized successfully")
            return True
        else:
            logger.warning("Prediction engine initialized but model loading failed")
            return False
    except Exception as e:
        logger.error(f"Failed to initialize prediction engine: {str(e)}")
        return False

def validate_input_data(data, machine_type):
    """Validate input data"""
    errors = []
    
    # Validazioni base
    required_fields = [
        'Installation_Year', 'Operational_Hours', 'Temperature_C', 
        'Vibration_mms', 'Sound_dB', 'Oil_Level_pct', 'Coolant_Level_pct',
        'Power_Consumption_kW', 'Last_Maintenance_Days_Ago'
    ]
    
    for field in required_fields:
        if field not in data or data[field] is None:
            errors.append(f"Campo {FIELD_TRANSLATIONS.get(field, field)} è obbligatorio")
    
    # Validazioni specifiche per tipo di dispositivo
    if machine_type in DEVICE_CONFIG:
        for extra_field in DEVICE_CONFIG[machine_type]:
            if extra_field not in data or data[extra_field] is None:
                errors.append(f"Campo {FIELD_TRANSLATIONS.get(extra_field, extra_field)} è obbligatorio per {machine_type}")
    
    return errors

def prepare_input_for_prediction(data, machine_type):
    """Prepare input data for prediction"""
    # Creare un dizionario con tutti i possibili campi
    prepared = {
        'Machine_Type': machine_type,
        'Installation_Year': data.get('Installation_Year', 2020),
        'Operational_Hours': data.get('Operational_Hours', 0),
        'Temperature_C': data.get('Temperature_C', 25),
        'Vibration_mms': data.get('Vibration_mms', 0),
        'Sound_dB': data.get('Sound_dB', 50),
        'Oil_Level_pct': data.get('Oil_Level_pct', 100),
        'Coolant_Level_pct': data.get('Coolant_Level_pct', 100),
        'Power_Consumption_kW': data.get('Power_Consumption_kW', 0),
        'Last_Maintenance_Days_Ago': data.get('Last_Maintenance_Days_Ago', 0),
        'Maintenance_History_Count': data.get('Maintenance_History_Count', 0),
        'Failure_History_Count': data.get('Failure_History_Count', 0),
        'AI_Supervision': data.get('AI_Supervision', 'No'),
        'Error_Codes_Last_30_Days': data.get('Error_Codes_Last_30_Days', 0),
        'AI_Override_Events': data.get('AI_Override_Events', 0),
        'Laser_Intensity': data.get('Laser_Intensity', 0),
        'Hydraulic_Pressure_bar': data.get('Hydraulic_Pressure_bar', 0),
        'Coolant_Flow_L_min': data.get('Coolant_Flow_L_min', 0),
        'Heat_Index': data.get('Heat_Index', 0)
    }
    
    return prepared

def generate_risk_assessment(failure_probability, machine_type, data):
    """Generate risk assessment"""
    if failure_probability < 20:
        risk_level = "Basso"
        action = "Nessuna azione richiesta"
    elif failure_probability < 50:
        risk_level = "Medio"
        action = "Monitoraggio aumentato"
    elif failure_probability < 80:
        risk_level = "Alto"
        action = "Manutenzione preventiva raccomandata"
    else:
        risk_level = "Critico"
        action = "Sostituzione immediata"
    
    return {
        'risk_level': risk_level,
        'recommended_action': action,
        'probability': failure_probability
    }

def calculate_maintenance_schedule(risk_assessment, machine_type):
    """Calculate maintenance schedule"""
    if risk_assessment['risk_level'] == "Critico":
        return "Immediata"
    elif risk_assessment['risk_level'] == "Alto":
        return "Entro 7 giorni"
    elif risk_assessment['risk_level'] == "Medio":
        return "Entro 30 giorni"
    else:
        return "Programmata normale"

def format_prediction_output(risk_assessment):
    """Format prediction output"""
    prob_percent = int(risk_assessment['probability'])
    return f"{prob_percent}% di probabilità guasto. Azione consigliata: {risk_assessment['recommended_action']}"

def convert_to_display_format(data):
    """Convert data to display format"""
    display_data = {}
    for key, value in data.items():
        if key in FIELD_TRANSLATIONS:
            display_data[FIELD_TRANSLATIONS[key]] = value
        else:
            display_data[key] = value
    return display_data

def get_device_info(machine_type):
    """Get device information"""
    if machine_type in COMMON_DEVICES:
        return {
            'type': 'common',
            'additional_fields': []
        }
    elif machine_type in DEVICE_CONFIG:
        return {
            'type': 'special',
            'additional_fields': DEVICE_CONFIG[machine_type]
        }
    return None

def get_system_health_status():
    """Get system health status"""
    return {
        'status': 'OK',
        'prediction_engine': prediction_engine is not None,
        'timestamp': datetime.now().isoformat()
    }

def log_prediction(input_data, prediction, machine_type):
    """Log prediction"""
    try:
        log_entry = {
            'timestamp': datetime.now().isoformat(),
            'machine_type': machine_type,
            'input_data': input_data,
            'prediction': prediction
        }
        
        os.makedirs('logs', exist_ok=True)
        today = datetime.now().strftime('%Y-%m-%d')
        log_file = f"logs/predictions_{today}.json"
        
        # Leggere log esistenti
        logs = []
        if os.path.exists(log_file):
            try:
                with open(log_file, 'r') as f:
                    logs = json.load(f)
            except:
                logs = []
        
        # Aggiungere nuovo log
        logs.append(log_entry)
        
        # Salvare
        with open(log_file, 'w') as f:
            json.dump(logs, f, indent=2)
            
    except Exception as e:
        logger.error(f"Error logging prediction: {str(e)}")

# Initialize prediction engine before first request
@app.before_request
def before_request():
    """Initialize application before handling requests"""
    if not hasattr(app, '_prediction_engine_initialized'):
        init_prediction_engine()
        app._prediction_engine_initialized = True

@app.route('/')
def index():
    """Main page"""
    system_status = get_system_health_status()
    all_devices = COMMON_DEVICES + list(DEVICE_CONFIG.keys())
    
    return render_template('index.html', 
                         devices=sorted(all_devices),
                         device_config=DEVICE_CONFIG,
                         system_status=system_status)

@app.route('/predict', methods=['POST'])
def predict():
    """Handle prediction request"""
    try:
        if not prediction_engine:
            return jsonify({
                'success': False,
                'error': 'Prediction engine not initialized'
            }), 500
        
        # Get form data
        data = request.get_json() if request.is_json else request.form.to_dict()
        machine_type = data.get('machine_type')
        
        if not machine_type:
            return jsonify({
                'success': False,
                'error': 'Machine type is required'
            }), 400
        
        # Convert numeric fields
        numeric_fields = [
            'Installation_Year', 'Operational_Hours', 'Temperature_C', 
            'Vibration_mms', 'Sound_dB', 'Oil_Level_pct', 'Coolant_Level_pct',
            'Power_Consumption_kW', 'Last_Maintenance_Days_Ago', 
            'Maintenance_History_Count', 'Failure_History_Count',
            'Error_Codes_Last_30_Days', 'AI_Override_Events',
            'Laser_Intensity', 'Hydraulic_Pressure_bar', 
            'Coolant_Flow_L_min', 'Heat_Index'
        ]
        
        input_data = {}
        for field in numeric_fields:
            if field in data and data[field] != '':
                try:
                    input_data[field] = float(data[field])
                except (ValueError, TypeError):
                    return jsonify({
                        'success': False,
                        'error': f'Invalid value for {field}'
                    }), 400
        
        # Handle AI_Supervision
        if 'AI_Supervision' in data:
            input_data['AI_Supervision'] = 'Yes' if data['AI_Supervision'] in ['1', 'on', True] else 'No'
        
        # Validate input data
        validation_errors = validate_input_data(input_data, machine_type)
        if validation_errors:
            return jsonify({
                'success': False,
                'error': 'Validation errors',
                'details': validation_errors
            }), 400
        
        # Prepare input for prediction
        prepared_data = prepare_input_for_prediction(input_data, machine_type)
        
        # Make prediction
        result = prediction_engine.predict(prepared_data)
        
        # Generate risk assessment
        risk_assessment = generate_risk_assessment(
            result['failure_probability'], 
            machine_type, 
            prepared_data
        )
        
        # Calculate maintenance schedule
        maintenance_schedule = calculate_maintenance_schedule(risk_assessment, machine_type)
        
        # Calcola giorni rimanenti e guasto entro 7 giorni
        remaining_days = result.get('remaining_useful_life_days', 100)
        failure_within_7_days = remaining_days <= 7 or result.get('failure_within_7_days', False)
        
        # Prepare response
        response_data = {
            'success': True,
            'prediction': {
                'failure_probability': result['failure_probability'],
                'failure_within_7_days': failure_within_7_days,
                'remaining_useful_life': remaining_days,
                'risk_assessment': risk_assessment,
                'maintenance_schedule': maintenance_schedule,
                'formatted_output': result.get('formatted_output', f"{int(result['failure_probability'])}% di probabilità guasto"),
                'input_data_display': convert_to_display_format(prepared_data),
                'machine_type': machine_type,
                'timestamp': datetime.now().isoformat()
            }
        }
        
        # Log prediction
        log_prediction(prepared_data, response_data['prediction'], machine_type)
        
        return jsonify(response_data)
        
    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        return jsonify({
            'success': False,
            'error': f'Prediction failed: {str(e)}'
        }), 500

@app.route('/device_info/<machine_type>')
def device_info(machine_type):
    """Get device information"""
    info = get_device_info(machine_type)
    if info:
        return jsonify({
            'success': True,
            'device_info': info
        })
    else:
        return jsonify({
            'success': False,
            'error': 'Device type not found'
        }), 404

@app.route('/system_status')
def system_status():
    """Get system health status"""
    status = get_system_health_status()
    return jsonify(status)

@app.route('/results')
def results():
    """Results page"""
    return render_template('results.html')

@app.route('/api/predictions/history')
def prediction_history():
    """Get prediction history"""
    try:
        today = datetime.now().strftime('%Y-%m-%d')
        log_file = f"logs/predictions_{today}.json"
        
        if os.path.exists(log_file):
            with open(log_file, 'r') as f:
                history = json.load(f)
            return jsonify({
                'success': True,
                'history': history[-10:]  # Return last 10 predictions
            })
        else:
            return jsonify({
                'success': True,
                'history': []
            })
    except Exception as e:
        logger.error(f"Error retrieving prediction history: {str(e)}")
        return jsonify({
            'success': False,
            'error': 'Failed to retrieve prediction history'
        }), 500

@app.route('/api/devices')
def get_devices():
    """Get list of all devices"""
    all_devices = COMMON_DEVICES + list(DEVICE_CONFIG.keys())
    return jsonify({
        'success': True,
        'devices': sorted(all_devices),
        'device_config': DEVICE_CONFIG
    })

@app.route('/health')
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'OK',
        'timestamp': datetime.now().isoformat(),
        'prediction_engine_ready': prediction_engine is not None
    })

@app.errorhandler(404)
def not_found(error):
    """Handle 404 errors"""
    return render_template('index.html', error="Page not found"), 404

@app.errorhandler(500)
def internal_error(error):
    """Handle 500 errors"""
    logger.error(f"Internal server error: {str(error)}")
    return render_template('index.html', error="Internal server error"), 500

if __name__ == '__main__':
    # Create required directories
    os.makedirs('logs', exist_ok=True)
    
    # Initialize prediction engine on startup
    if not init_prediction_engine():
        logger.warning("Prediction engine initialization failed - some features may not work")
    
    # Run the application
    app.run(debug=True, host='0.0.0.0', port=5000)