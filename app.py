"""
Flask Web Application for IoT Predictive Maintenance System
Provides web interface for device failure prediction
"""

from flask import Flask, render_template, request, jsonify, session, redirect, url_for
from werkzeug.security import generate_password_hash, check_password_hash
import os
import sys
import json
from datetime import datetime
import logging

# Add src directory to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

try:
    from prediction_engine import PredictionEngine
    from utils_complete import (
        validate_input_data, 
        get_device_info, 
        DEVICE_CONFIG, 
        COMMON_DEVICES,
        log_prediction,
        get_system_health_status,
        prepare_input_for_prediction,
        generate_risk_assessment,
        format_prediction_output,
        calculate_maintenance_schedule,
        convert_to_display_format
    )
except ImportError as e:
    print(f"Import error: {e}")
    # Fallback imports or dummy functions can be added here

app = Flask(__name__)
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
        prediction_engine = PredictionEngine()
        prediction_engine.load_model()
        logger.info("Prediction engine initialized successfully")
        return True
    except Exception as e:
        logger.error(f"Failed to initialize prediction engine: {str(e)}")
        return False

@app.before_first_request
def before_first_request():
    """Initialize application before handling first request"""
    init_prediction_engine()

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
            input_data['AI_Supervision'] = 1 if data['AI_Supervision'] in ['1', 'on', True] else 0
        
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
        result = prediction_engine.predict_failure(prepared_data, machine_type)
        
        # Generate risk assessment
        risk_assessment = generate_risk_assessment(
            result['failure_probability'], 
            machine_type, 
            prepared_data
        )
        
        # Calculate maintenance schedule
        maintenance_schedule = calculate_maintenance_schedule(risk_assessment, machine_type)
        
        # Format output
        formatted_output = format_prediction_output(risk_assessment)
        
        # Prepare response
        response_data = {
            'success': True,
            'prediction': {
                'failure_probability': result['failure_probability'],
                'failure_within_7_days': result['failure_within_7_days'],
                'risk_assessment': risk_assessment,
                'maintenance_schedule': maintenance_schedule,
                'formatted_output': formatted_output,
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
    os.makedirs('data/models', exist_ok=True)
    
    # Initialize prediction engine on startup
    if not init_prediction_engine():
        logger.warning("Prediction engine initialization failed - some features may not work")
    
    # Run the application
    app.run(debug=True, host='0.0.0.0', port=5000)