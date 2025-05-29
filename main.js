/**
 * Main JavaScript file for IoT Predictive Maintenance Web Interface
 */

document.addEventListener('DOMContentLoaded', function() {
    // Initialize the application
    initializeApp();
});

function initializeApp() {
    // Check system status
    checkSystemStatus();
    
    // Set up event listeners
    setupEventListeners();
    
    // Initialize machine type selection
    initializeMachineTypeSelection();
    
    // Load device configurations
    loadDeviceConfigurations();
}

function setupEventListeners() {
    // Machine type selection change
    const machineTypeSelect = document.getElementById('machine_type');
    if (machineTypeSelect) {
        machineTypeSelect.addEventListener('change', handleMachineTypeChange);
    }
    
    // Prediction form submission
    const predictionForm = document.getElementById('prediction-form');
    if (predictionForm) {
        predictionForm.addEventListener('submit', handlePredictionSubmit);
    }
    
    // Reset form button
    const resetBtn = document.getElementById('reset-btn');
    if (resetBtn) {
        resetBtn.addEventListener('click', resetForm);
    }
    
    // Load sample data buttons
    const sampleButtons = document.querySelectorAll('.sample-data-btn');
    sampleButtons.forEach(btn => {
        btn.addEventListener('click', loadSampleData);
    });
}

function handleMachineTypeChange(event) {
    const selectedType = event.target.value;
    updateAdditionalFields(selectedType);
    
    if (selectedType) {
        loadDeviceInfo(selectedType);
    }
}

function updateAdditionalFields(machineType) {
    // Hide all additional fields first
    const additionalFields = document.querySelectorAll('.additional-field');
    additionalFields.forEach(field => {
        field.style.display = 'none';
        const input = field.querySelector('input');
        if (input) {
            input.required = false;
        }
    });
    
    // Show relevant additional field based on machine type
    const deviceConfig = {
        'Laser_Cutter': 'laser-intensity-field',
        'Hydraulic_Press': 'hydraulic-pressure-field',
        'Injection_Molder': 'hydraulic-pressure-field',
        'CNC_Lathe': 'coolant-flow-field',
        'CNC_Mill': 'coolant-flow-field',
        'Industrial_Chiller': 'coolant-flow-field',
        'Boiler': 'heat-index-field',
        'Furnace': 'heat-index-field',
        'Heat_Exchanger': 'heat-index-field'
    };
    
    const fieldId = deviceConfig[machineType];
    if (fieldId) {
        const field = document.getElementById(fieldId);
        if (field) {
            field.style.display = 'block';
            const input = field.querySelector('input');
            if (input) {
                input.required = true;
            }
        }
    }
}

function loadDeviceInfo(machineType) {
    fetch(`/device_info/${machineType}`)
        .then(response => response.json())
        .then(data => {
            if (data.success) {
                displayDeviceInfo(data.device_info);
            }
        })
        .catch(error => {
            console.error('Error loading device info:', error);
        });
}

function displayDeviceInfo(deviceInfo) {
    const infoContainer = document.getElementById('device-info');
    if (infoContainer && deviceInfo.description) {
        infoContainer.innerHTML = `
            <div class="alert alert-info">
                <i class="fas fa-info-circle"></i>
                <strong>Informazioni dispositivo:</strong> ${deviceInfo.description}
            </div>
        `;
    }
}

function handlePredictionSubmit(event) {
    event.preventDefault();
    
    // Show loading state
    showLoadingState();
    
    // Collect form data
    const formData = new FormData(event.target);
    const data = {};
    
    for (let [key, value] of formData.entries()) {
        if (value !== '') {
            data[key] = value;
        }
    }
    
    // Make prediction request
    fetch('/predict', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
        },
        body: JSON.stringify(data)
    })
    .then(response => response.json())
    .then(data => {
        hideLoadingState();
        if (data.success) {
            displayPredictionResults(data.prediction);
        } else {
            displayError(data.error, data.details);
        }
    })
    .catch(error => {
        hideLoadingState();
        console.error('Prediction error:', error);
        displayError('Errore di connessione. Riprova più tardi.');
    });
}

function displayPredictionResults(prediction) {
    const resultsContainer = document.getElementById('results-container');
    const resultsContent = document.getElementById('results-content');
    
    if (!resultsContainer || !resultsContent) return;
    
    // Show results container
    resultsContainer.style.display = 'block';
    
    // Create results HTML
    const riskLevel = prediction.risk_assessment.risk_level;
    const riskClass = prediction.risk_assessment.risk_class;
    const riskScore = prediction.failure_probability * 100;
    
    resultsContent.innerHTML = `
        <div class="prediction-summary">
            <div class="risk-indicator ${riskClass}">
                <div class="risk-score">${riskScore.toFixed(1)}%</div>
                <div class="risk-label">Probabilità di Guasto</div>
            </div>
            <div class="risk-level-badge ${riskClass.replace('risk-', 'badge-')}">
                Rischio ${riskLevel}
            </div>
        </div>
        
        <div class="prediction-details">
            <div class="alert ${getAlertClass(riskLevel)}">
                <i class="${getRiskIcon(riskLevel)}"></i>
                <div class="alert-content">
                    <strong>${prediction.formatted_output}</strong>
                </div>
            </div>
        </div>
        
        <div class="risk-factors">
            <h4><i class="fas fa-exclamation-triangle"></i> Fattori di Rischio</h4>
            <ul>
                ${prediction.risk_assessment.risk_factors.map(factor => `<li>${factor}</li>`).join('')}
            </ul>
        </div>
        
        <div class="recommendations">
            <h4><i class="fas fa-tools"></i> Raccomandazioni</h4>
            <ul>
                ${prediction.risk_assessment.recommendations.map(rec => `<li>${rec}</li>`).join('')}
            </ul>
        </div>
        
        <div class="maintenance-schedule">
            <h4><i class="fas fa-calendar-alt"></i> Programma Manutenzione</h4>
            <div class="maintenance-info">
                <div class="maintenance-item">
                    <strong>Priorità:</strong> ${prediction.maintenance_schedule.priority}
                </div>
                <div class="maintenance-item">
                    <strong>Tipo:</strong> ${prediction.maintenance_schedule.maintenance_type}
                </div>
                <div class="maintenance-item">
                    <strong>Entro:</strong> ${prediction.maintenance_schedule.days_until_maintenance} giorni
                </div>
                <div class="maintenance-item">
                    <strong>Costo stimato:</strong> €${prediction.maintenance_schedule.estimated_cost}
                </div>
                <div class="maintenance-item">
                    <strong>Durata stimata:</strong> ${prediction.maintenance_schedule.estimated_duration} ore
                </div>
            </div>
        </div>
        
        <div class="prediction-chart">
            ${createRiskChart(riskScore)}
        </div>
    `;
    
    // Scroll to results
    resultsContainer.scrollIntoView({ behavior: 'smooth' });
}

function createRiskChart(riskScore) {
    return `
        <div class="chart-container">
            <h4><i class="fas fa-chart-bar"></i> Analisi del Rischio</h4>
            <div class="risk-bar">
                <div class="risk-bar-fill" style="width: ${riskScore}%"></div>
                <div class="risk-bar-text">${riskScore.toFixed(1)}%</div>
            </div>
            <div class="risk-scale">
                <span class="scale-low">0% - Basso</span>
                <span class="scale-medium">40% - Medio</span>
                <span class="scale-high">70% - Alto</span>
            </div>
        </div>
    `;
}

function getAlertClass(riskLevel) {
    switch (riskLevel) {
        case 'ALTO': return 'alert-danger';
        case 'MEDIO': return 'alert-warning';
        case 'BASSO': return 'alert-success';
        default: return 'alert-info';
    }
}

function getRiskIcon(riskLevel) {
    switch (riskLevel) {
        case 'ALTO': return 'fas fa-exclamation-triangle';
        case 'MEDIO': return 'fas fa-exclamation-circle';
        case 'BASSO': return 'fas fa-check-circle';
        default: return 'fas fa-info-circle';
    }
}

function displayError(error, details = []) {
    const resultsContainer = document.getElementById('results-container');
    const resultsContent = document.getElementById('results-content');
    
    if (!resultsContainer || !resultsContent) return;
    
    resultsContainer.style.display = 'block';
    
    let errorHtml = `
        <div class="alert alert-danger">
            <i class="fas fa-exclamation-triangle"></i>
            <strong>Errore:</strong> ${error}
        </div>
    `;
    
    if (details && details.length > 0) {
        errorHtml += `
            <div class="error-details">
                <h5>Dettagli:</h5>
                <ul>
                    ${details.map(detail => `<li>${detail}</li>`).join('')}
                </ul>
            </div>
        `;
    }
    
    resultsContent.innerHTML = errorHtml;
    resultsContainer.scrollIntoView({ behavior: 'smooth' });
}

function showLoadingState() {
    const submitBtn = document.querySelector('button[type="submit"]');
    const resultsContainer = document.getElementById('results-container');
    
    if (submitBtn) {
        submitBtn.disabled = true;
        submitBtn.innerHTML = '<i class="fas fa-spinner fa-spin"></i> Analizzando...';
    }
    
    if (resultsContainer) {
        resultsContainer.style.display = 'block';
        document.getElementById('results-content').innerHTML = `
            <div class="loading-state">
                <div class="spinner">
                    <i class="fas fa-cog fa-spin"></i>
                </div>
                <p>Analisi in corso... Attendere prego.</p>
            </div>
        `;
    }
}

function hideLoadingState() {
    const submitBtn = document.querySelector('button[type="submit"]');
    
    if (submitBtn) {
        submitBtn.disabled = false;
        submitBtn.innerHTML = '<i class="fas fa-search"></i> Analizza Dispositivo';
    }
}

function resetForm() {
    const form = document.getElementById('prediction-form');
    const resultsContainer = document.getElementById('results-container');
    
    if (form) {
        form.reset();
        
        // Hide additional fields
        const additionalFields = document.querySelectorAll('.additional-field');
        additionalFields.forEach(field => {
            field.style.display = 'none';
        });
    }
    
    if (resultsContainer) {
        resultsContainer.style.display = 'none';
    }
    
    // Clear device info
    const infoContainer = document.getElementById('device-info');
    if (infoContainer) {
        infoContainer.innerHTML = '';
    }
}

function loadSampleData(event) {
    const riskType = event.target.dataset.risk;
    const machineType = document.getElementById('machine_type').value;
    
    if (!machineType) {
        alert('Seleziona prima un tipo di dispositivo');
        return;
    }
    
    // Sample data based on risk type
    const sampleData = getSampleData(machineType, riskType);
    
    // Fill form with sample data
    Object.keys(sampleData).forEach(key => {
        const input = document.querySelector(`[name="${key}"]`);
        if (input) {
            input.value = sampleData[key];
        }
    });
}

function getSampleData(machineType, riskType) {
    const baseData = {
        'Installation_Year': 2020,
        'Operational_Hours': 15000,
        'Temperature_C': 45,
        'Vibration_mms': 5,
        'Sound_dB': 65,
        'Oil_Level_pct': 75,
        'Coolant_Level_pct': 80,
        'Power_Consumption_kW': 25,
        'Last_Maintenance_Days_Ago': 15,
        'Maintenance_History_Count': 3,
        'Failure_History_Count': 1,
        'AI_Supervision': 1,
        'Error_Codes_Last_30_Days': 2,
        'AI_Override_Events': 0
    };
    
    // Adjust based on risk level
    if (riskType === 'high') {
        Object.assign(baseData, {
            'Last_Maintenance_Days_Ago': 60,
            'Failure_History_Count': 5,
            'Temperature_C': 85,
            'Vibration_mms': 15,
            'Oil_Level_pct': 15,
            'Error_Codes_Last_30_Days': 12
        });
    } else if (riskType === 'low') {
        Object.assign(baseData, {
            'Last_Maintenance_Days_Ago': 5,
            'Failure_History_Count': 0,
            'Temperature_C': 35,
            'Vibration_mms': 2,
            'Oil_Level_pct': 90,
            'Error_Codes_Last_30_Days': 0
        });
    }
    
    // Add machine-specific fields
    const deviceConfig = {
        'Laser_Cutter': { 'Laser_Intensity': riskType === 'high' ? 90 : riskType === 'low' ? 30 : 60 },
        'Hydraulic_Press': { 'Hydraulic_Pressure_bar': riskType === 'high' ? 450 : riskType === 'low' ? 150 : 300 },
        'Injection_Molder': { 'Hydraulic_Pressure_bar': riskType === 'high' ? 450 : riskType === 'low' ? 150 : 300 },
        'CNC_Lathe': { 'Coolant_Flow_L_min': riskType === 'high' ? 45 : riskType === 'low' ? 15 : 30 },
        'CNC_Mill': { 'Coolant_Flow_L_min': riskType === 'high' ? 45 : riskType === 'low' ? 15 : 30 },
        'Industrial_Chiller': { 'Coolant_Flow_L_min': riskType === 'high' ? 90 : riskType === 'low' ? 30 : 60 },
        'Boiler': { 'Heat_Index': riskType === 'high' ? 90 : riskType === 'low' ? 30 : 60 },
        'Furnace': { 'Heat_Index': riskType === 'high' ? 90 : riskType === 'low' ? 30 : 60 },
        'Heat_Exchanger': { 'Heat_Index': riskType === 'high' ? 90 : riskType === 'low' ? 30 : 60 }
    };
    
    if (deviceConfig[machineType]) {
        Object.assign(baseData, deviceConfig[machineType]);
    }
    
    return baseData;
}

function checkSystemStatus() {
    fetch('/system_status')
        .then(response => response.json())
        .then(data => {
            updateSystemStatus(data);
        })
        .catch(error => {
            console.error('Error checking system status:', error);
        });
}

function updateSystemStatus(status) {
    const statusContainer = document.getElementById('system-status');
    if (statusContainer) {
        let statusClass = 'status-ok';
        let statusIcon = 'fas fa-check-circle';
        
        if (status.status === 'WARNING') {
            statusClass = 'status-warning';
            statusIcon = 'fas fa-exclamation-triangle';
        } else if (status.status === 'ERROR') {
            statusClass = 'status-error';
            statusIcon = 'fas fa-times-circle';
        }
        
        statusContainer.innerHTML = `
            <div class="system-status ${statusClass}">
                <i class="${statusIcon}"></i>
                <span>${status.message}</span>
            </div>
        `;
    }
}

function initializeMachineTypeSelection() {
    // Add any initialization logic for machine type selection
    const machineTypeSelect = document.getElementById('machine_type');
    if (machineTypeSelect && machineTypeSelect.options.length > 1) {
        // Optionally select a default machine type
        // machineTypeSelect.selectedIndex = 1;
        // handleMachineTypeChange({ target: machineTypeSelect });
    }
}

function loadDeviceConfigurations() {
    fetch('/api/devices')
        .then(response => response.json())
        .then(data => {
            if (data.success) {
                // Store device configurations for later use
                window.deviceConfigurations = data.device_config;
            }
        })
        .catch(error => {
            console.error('Error loading device configurations:', error);
        });
}