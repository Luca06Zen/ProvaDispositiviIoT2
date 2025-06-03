// Configurazione dispositivi e campi aggiuntivi
const deviceConfig = {
    'Laser_Cutter': {
        additionalField: 'laser',
        description: 'Dispositivo per taglio laser che richiede monitoraggio dell\'intensità laser'
    },
    'Hydraulic_Press': {
        additionalField: 'hydraulic',
        description: 'Pressa idraulica che richiede monitoraggio della pressione idraulica'
    },
    'Injection_Molder': {
        additionalField: 'hydraulic',
        description: 'Macchina per stampaggio a iniezione con controllo pressione idraulica'
    },
    'CNC_Lathe': {
        additionalField: 'coolant',
        description: 'Tornio CNC con sistema di raffreddamento monitorato'
    },
    'CNC_Mill': {
        additionalField: 'coolant',
        description: 'Fresatrice CNC con controllo portata refrigerante'
    },
    'Industrial_Chiller': {
        additionalField: 'coolant',
        description: 'Sistema di raffreddamento industriale'
    },
    'Boiler': {
        additionalField: 'heat',
        description: 'Caldaia industriale con monitoraggio indice di calore'
    },
    'Furnace': {
        additionalField: 'heat',
        description: 'Forno industriale ad alta temperatura'
    },
    'Heat_Exchanger': {
        additionalField: 'heat',
        description: 'Scambiatore di calore con controllo termico'
    }
};

// Variabile globale per il grafico
let riskChart = null;

// Inizializzazione al caricamento della pagina
document.addEventListener('DOMContentLoaded', function() {
    initializeEventListeners();
    resetForm();
    
    // Inizializza Chart.js se disponibile
    if (typeof Chart !== 'undefined') {
        Chart.defaults.responsive = true;
        Chart.defaults.maintainAspectRatio = false;
    }
});

function initializeEventListeners() {
    // Gestione selezione tipo dispositivo
    document.getElementById('machineType').addEventListener('change', handleMachineTypeChange);
    
    // Gestione supervisione AI
    document.getElementById('aiSupervision').addEventListener('change', handleAiSupervisionChange);
    
    // Gestione form submission
    document.getElementById('predictionForm').addEventListener('submit', handleFormSubmit);
}

function handleMachineTypeChange() {
    const selectedType = this.value;
    const additionalFields = document.getElementById('additionalFields');
    const machineInfo = document.getElementById('machineInfo');
    
    // Nascondi tutti i campi aggiuntivi
    hideAllAdditionalFields();
    
    // Reset valori campi aggiuntivi
    resetAdditionalFields();
    
    if (selectedType && deviceConfig[selectedType]) {
        const config = deviceConfig[selectedType];
        
        // Mostra sezione caratteristiche aggiuntive
        additionalFields.style.display = 'block';
        machineInfo.style.display = 'block';
        machineInfo.textContent = config.description;
        
        // Mostra il campo specifico
        showSpecificAdditionalField(config.additionalField);
    } else {
        // Nascondi sezione caratteristiche aggiuntive
        additionalFields.style.display = 'none';
        machineInfo.style.display = 'none';
        removeRequiredFromAdditionalFields();
    }
}

function handleAiSupervisionChange() {
    const aiSupervisionValue = this.value;
    const aiOverridesGroup = document.getElementById('aiOverridesGroup');
    const aiOverridesInput = document.getElementById('aiOverrides');
    
    if (aiSupervisionValue === '1') {
        // Mostra campo Override AI se supervisione è attiva
        aiOverridesGroup.style.display = 'block';
        aiOverridesInput.required = true;
        
        // Animazione di apertura
        aiOverridesGroup.style.opacity = '0';
        aiOverridesGroup.style.transform = 'translateY(-10px)';
        setTimeout(() => {
            aiOverridesGroup.style.opacity = '1';
            aiOverridesGroup.style.transform = 'translateY(0)';
        }, 100);
    } else {
        // Nascondi campo Override AI
        aiOverridesGroup.style.display = 'none';
        aiOverridesInput.required = false;
        aiOverridesInput.value = '0'; // Imposta valore di default
    }
}

function hideAllAdditionalFields() {
    const fieldGroups = [
        'laserIntensityGroup',
        'hydraulicPressureGroup', 
        'coolantFlowGroup',
        'heatIndexGroup'
    ];
    
    fieldGroups.forEach(groupId => {
        document.getElementById(groupId).style.display = 'none';
    });
}

function resetAdditionalFields() {
    const fields = [
        'laserIntensity',
        'hydraulicPressure',
        'coolantFlow', 
        'heatIndex'
    ];
    
    fields.forEach(fieldId => {
        const field = document.getElementById(fieldId);
        field.value = '';
        field.required = false;
    });
}

function showSpecificAdditionalField(fieldType) {
    let groupId, inputId;
    
    switch(fieldType) {
        case 'laser':
            groupId = 'laserIntensityGroup';
            inputId = 'laserIntensity';
            break;
        case 'hydraulic':
            groupId = 'hydraulicPressureGroup';
            inputId = 'hydraulicPressure';
            break;
        case 'coolant':
            groupId = 'coolantFlowGroup';
            inputId = 'coolantFlow';
            break;
        case 'heat':
            groupId = 'heatIndexGroup';
            inputId = 'heatIndex';
            break;
        default:
            return;
    }
    
    // Mostra il campo e lo rende obbligatorio
    const group = document.getElementById(groupId);
    const input = document.getElementById(inputId);
    
    group.style.display = 'block';
    input.required = true;
    
    // Animazione di apertura
    group.style.opacity = '0';
    group.style.transform = 'translateY(-10px)';
    setTimeout(() => {
        group.style.opacity = '1';
        group.style.transform = 'translateY(0)';
    }, 100);
}

function removeRequiredFromAdditionalFields() {
    const inputs = [
        'laserIntensity',
        'hydraulicPressure',
        'coolantFlow',
        'heatIndex'
    ];
    
    inputs.forEach(inputId => {
        document.getElementById(inputId).required = false;
    });
}

function resetForm() {
    // Reset del form
    document.getElementById('predictionForm').reset();
    
    // Nascondi tutti i campi aggiuntivi
    hideAllAdditionalFields();
    document.getElementById('additionalFields').style.display = 'none';
    document.getElementById('machineInfo').style.display = 'none';
    document.getElementById('aiOverridesGroup').style.display = 'none';
    
    // Reset pannello risultati
    showWelcomeMessage();
    
    // Distruggi grafico esistente
    destroyExistingChart();
}

function handleFormSubmit(event) {
    event.preventDefault();
    
    try {
        // Raccogli e valida dati dal form
        const formData = collectFormData();
        validateFormData(formData);
        
        // Mostra loading
        showLoading();
        
        // Simula chiamata API (sostituire con chiamata reale)
        setTimeout(() => {
            simulatePrediction(formData);
        }, 2000);
        
    } catch (error) {
        showError('Errore nella validazione dei dati: ' + error.message);
    }
}

function collectFormData() {
    const form = document.getElementById('predictionForm');
    const formData = new FormData(form);
    const data = {};
    
    // Converti FormData in oggetto normale
    for (let [key, value] of formData.entries()) {
        data[key] = value;
    }
    
    // Aggiungi valori di default per campi non compilati
    if (!data.aiOverrides) data.aiOverrides = '0';
    if (!data.laserIntensity) data.laserIntensity = null;
    if (!data.hydraulicPressure) data.hydraulicPressure = null;
    if (!data.coolantFlow) data.coolantFlow = null;
    if (!data.heatIndex) data.heatIndex = null;
    
    return data;
}

function showLoading() {
    document.getElementById('welcomeMessage').style.display = 'none';
    document.getElementById('resultsContainer').style.display = 'none';
    document.getElementById('errorContainer').style.display = 'none';
    document.getElementById('loadingMessage').style.display = 'block';
    
    // Distruggi grafico esistente durante il loading
    destroyExistingChart();
}

function showWelcomeMessage() {
    document.getElementById('loadingMessage').style.display = 'none';
    document.getElementById('resultsContainer').style.display = 'none';
    document.getElementById('errorContainer').style.display = 'none';
    document.getElementById('welcomeMessage').style.display = 'block';
    
    // Distruggi grafico esistente
    destroyExistingChart();
}

function showResults() {
    document.getElementById('loadingMessage').style.display = 'none';
    document.getElementById('welcomeMessage').style.display = 'none';
    document.getElementById('errorContainer').style.display = 'none';
    document.getElementById('resultsContainer').style.display = 'block';
}

function showError(errorMessage) {
    document.getElementById('loadingMessage').style.display = 'none';
    document.getElementById('welcomeMessage').style.display = 'none';
    document.getElementById('resultsContainer').style.display = 'none';
    document.getElementById('errorContainer').style.display = 'block';
    document.getElementById('errorText').textContent = errorMessage;
    
    // Distruggi grafico esistente in caso di errore
    destroyExistingChart();
}

function simulatePrediction(data) {
    try {
        // Simula una previsione basata sui dati (sostituire con chiamata API reale)
        const prediction = generateMockPrediction(data);
        displayResults(prediction);
        showResults();
    } catch (error) {
        showError('Errore durante l\'elaborazione dei dati: ' + error.message);
    }
}

function generateMockPrediction(data) {
    // Genera una previsione simulata basata sui parametri
    const riskFactors = calculateRiskFactors(data);
    const failureProbability = Math.min(Math.max(riskFactors * 100, 0), 100);
    const remainingDays = Math.max(365 - Math.floor(riskFactors * 300), 1);
    const failureWithin7Days = remainingDays <= 7;
    
    let action = 'nessuna';
    if (failureProbability > 80) {
        action = 'sostituzione immediata';
    } else if (failureProbability > 50) {
        action = 'manutenzione urgente';
    } else if (failureProbability > 20) {
        action = 'monitoraggio intensivo';
    }
    
    return {
        failureProbability: Math.round(failureProbability),
        remainingDays: remainingDays,
        failureWithin7Days: failureWithin7Days,
        recommendedAction: action,
        riskLevel: getRiskLevel(failureProbability),
        deviceType: data.machineType
    };
}

function calculateRiskFactors(data) {
    let risk = 0;
    
    // Fattori di rischio basati sui parametri
    if (data.lastMaintenance > 90) risk += 0.3;
    if (data.failureCount > 3) risk += 0.2;
    if (data.errorCodes > 10) risk += 0.25;
    if (data.oilLevel < 30) risk += 0.15;
    if (data.coolantLevel < 40) risk += 0.1;
    if (data.temperature > 80) risk += 0.1;
    if (data.vibration > 5) risk += 0.1;
    if (data.sound > 85) risk += 0.05;
    
    // Fattori di età
    const age = 2025 - parseInt(data.installationYear);
    if (age > 10) risk += 0.2;
    else if (age > 5) risk += 0.1;
    
    return Math.min(risk, 1);
}

function getRiskLevel(probability) {
    if (probability >= 70) return 'high';
    if (probability >= 30) return 'medium';
    return 'low';
}

function displayResults(prediction) {
    // Testo della previsione
    const predictionText = `
        <strong>${prediction.failureProbability}% di probabilità guasto</strong><br>
        <strong>Azione consigliata:</strong> ${prediction.recommendedAction}<br>
        <strong>Giorni di vita rimanenti:</strong> ${prediction.remainingDays}<br>
        <strong>Guasto entro 7 giorni:</strong> ${prediction.failureWithin7Days ? 'sì' : 'no'}
    `;
    
    document.getElementById('predictionText').innerHTML = predictionText;
    
    // Indicatore di rischio
    const riskElement = document.getElementById('riskLevel');
    const riskClass = `risk-${prediction.riskLevel}`;
    const riskText = prediction.riskLevel === 'high' ? 'RISCHIO ALTO' : 
                    prediction.riskLevel === 'medium' ? 'RISCHIO MEDIO' : 'RISCHIO BASSO';
    
    riskElement.innerHTML = `<span class="${riskClass}">${riskText}</span>`;
    
    // Raccomandazioni dettagliate
    const recommendations = generateRecommendations(prediction);
    document.getElementById('recommendationsText').innerHTML = recommendations;
    
    // Aspetta che il contenitore sia visibile prima di creare il grafico
    setTimeout(() => {
        createRiskChart(prediction);
    }, 100);
}

function generateRecommendations(prediction) {
    let recommendations = '';
    
    switch(prediction.riskLevel) {
        case 'high':
            recommendations = `
                <p><strong>⚠️ AZIONE IMMEDIATA RICHIESTA:</strong></p>
                <ul>
                    <li>🔧 Programmare manutenzione entro 24-48 ore</li>
                    <li>📊 Monitoraggio continuo dei parametri critici</li>
                    <li>👥 Allertare il team tecnico specializzato</li>
                    <li>📋 Preparare ricambi e attrezzature necessarie</li>
                    <li>⏸️ Considerare la sospensione temporanea dell'attività</li>
                </ul>
            `;
            break;
        case 'medium':
            recommendations = `
                <p><strong>⚡ MONITORAGGIO INTENSIVO:</strong></p>
                <ul>
                    <li>🔍 Aumentare frequenza controlli giornalieri</li>
                    <li>📅 Pianificare manutenzione preventiva entro 1-2 settimane</li>
                    <li>📈 Verificare trend dei parametri operativi</li>
                    <li>🛠️ Controllare livelli olio e liquidi</li>
                    <li>📞 Contattare fornitore per consulenza tecnica</li>
                </ul>
            `;
            break;
        case 'low':
            recommendations = `
                <p><strong>✅ SITUAZIONE SOTTO CONTROLLO:</strong></p>
                <ul>
                    <li>📅 Mantenere programma di manutenzione ordinaria</li>
                    <li>📊 Continuare monitoraggio parametri standard</li>
                    <li>📋 Aggiornare log di manutenzione</li>
                    <li>🎯 Ottimizzare efficienza operativa</li>
                    <li>📈 Analizzare dati per miglioramenti futuri</li>
                </ul>
            `;
            break;
    }
    
    return recommendations;
}

function destroyExistingChart() {
    if (riskChart && typeof riskChart.destroy === 'function') {
        try {
            riskChart.destroy();
            riskChart = null;
        } catch (error) {
            console.warn('Errore nella distruzione del grafico:', error);
            riskChart = null;
        }
    }
}

function createRiskChart(prediction) {
    // Verifica che Chart.js sia caricato
    if (typeof Chart === 'undefined') {
        console.error('Chart.js non è caricato');
        return;
    }
    
    const chartContainer = document.getElementById('chartContainer');
    const canvas = document.getElementById('riskChart');
    
    // Verifica che gli elementi esistano e siano visibili
    if (!canvas || !chartContainer) {
        console.error('Elementi canvas o container non trovati');
        return;
    }
    
    // Verifica che il container sia visibile
    if (chartContainer.offsetWidth === 0 || chartContainer.offsetHeight === 0) {
        console.warn('Container del grafico non visibile, riprovo tra 200ms');
        setTimeout(() => createRiskChart(prediction), 200);
        return;
    }
    
    // Distruggi grafico esistente
    destroyExistingChart();
    
    try {
        // Ottieni il contesto 2D
        const ctx = canvas.getContext('2d');
        if (!ctx) {
            console.error('Impossibile ottenere il context 2D del canvas');
            return;
        }
        
        // Forza le dimensioni del canvas
        canvas.width = chartContainer.offsetWidth || 400;
        canvas.height = 300;
        
        const data = {
            labels: ['Probabilità Guasto', 'Affidabilità'],
            datasets: [{
                data: [prediction.failureProbability, 100 - prediction.failureProbability],
                backgroundColor: [
                    prediction.riskLevel === 'high' ? '#e74c3c' :
                    prediction.riskLevel === 'medium' ? '#f39c12' : '#27ae60',
                    '#ecf0f1'
                ],
                borderWidth: 2,
                borderColor: '#2c3e50',
                hoverBorderWidth: 3
            }]
        };
        
        const options = {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: {
                    position: 'bottom',
                    labels: {
                        font: {
                            size: 12
                        },
                        padding: 20,
                        usePointStyle: true
                    }
                },
                title: {
                    display: true,
                    text: `Analisi Rischio - ${getDeviceDisplayName(prediction.deviceType)}`,
                    font: {
                        size: 14,
                        weight: 'bold'
                    },
                    padding: {
                        top: 10,
                        bottom: 20
                    }
                },
                tooltip: {
                    callbacks: {
                        label: function(context) {
                            return context.label + ': ' + context.parsed + '%';
                        }
                    }
                }
            },
            animation: {
                animateRotate: true,
                animateScale: true,
                duration: 1000
            }
        };
        
        // Crea il nuovo grafico
        riskChart = new Chart(ctx, {
            type: 'doughnut',
            data: data,
            options: options
        });
        
        console.log('Grafico creato con successo');
        
    } catch (error) {
        console.error('Errore nella creazione del grafico:', error);
        
        // Fallback: mostra un messaggio testuale se il grafico non può essere creato
        const fallbackHtml = `
            <div style="text-align: center; padding: 2rem; background: #f8f9fa; border-radius: 10px;">
                <h4>Analisi Rischio - ${getDeviceDisplayName(prediction.deviceType)}</h4>
                <div style="font-size: 1.2rem; margin: 1rem 0;">
                    <strong>Probabilità Guasto: ${prediction.failureProbability}%</strong>
                </div>
                <div style="font-size: 1.2rem;">
                    <strong>Affidabilità: ${100 - prediction.failureProbability}%</strong>
                </div>
            </div>
        `;
        
        chartContainer.innerHTML = fallbackHtml;
    }

    setTimeout(() => {
        const chartContainer = document.querySelector('.chart-container');
        const canvas = document.getElementById('riskChart');
        if (chartContainer && canvas) {
            chartContainer.classList.add('chart-visible');
            canvas.style.visibility = 'visible';
            canvas.style.opacity = '1';
            window.riskChart.resize();
        }
    }, 100);
}

function getDeviceDisplayName(deviceType) {
    const displayNames = {
        '3D_Printer': 'Stampante 3D',
        'AGV': 'Veicolo Guidato Automaticamente',
        'Automated_Screwdriver': 'Avvitatore Automatico',
        'CMM': 'Macchina di Misura Coordinata',
        'Carton_Former': 'Formatrice Cartoni',
        'Compressor': 'Compressore',
        'Conveyor_Belt': 'Nastro Trasportatore',
        'Crane': 'Gru',
        'Dryer': 'Essiccatore',
        'Forklift_Electric': 'Carrello Elevatore Elettrico',
        'Grinder': 'Smerigliatrice',
        'Labeler': 'Etichettatrice',
        'Mixer': 'Miscelatore',
        'Palletizer': 'Palletizzatore',
        'Pick_and_Place': 'Sistema Pick and Place',
        'Press_Brake': 'Pressa Piegatrice',
        'Pump': 'Pompa',
        'Robot_Arm': 'Braccio Robotico',
        'Shrink_Wrapper': 'Termoconfezionatrice',
        'Shuttle_System': 'Sistema Shuttle',
        'Vacuum_Packer': 'Confezionatrice Sottovuoto',
        'Valve_Controller': 'Controller Valvole',
        'Vision_System': 'Sistema di Visione',
        'XRay_Inspector': 'Ispettore a Raggi X',
        'Laser_Cutter': 'Tagliatrice Laser',
        'Hydraulic_Press': 'Pressa Idraulica',
        'Injection_Molder': 'Stampaggio a Iniezione',
        'CNC_Lathe': 'Tornio CNC',
        'CNC_Mill': 'Fresatrice CNC',
        'Industrial_Chiller': 'Refrigeratore Industriale',
        'Boiler': 'Caldaia',
        'Furnace': 'Forno',
        'Heat_Exchanger': 'Scambiatore di Calore'
    };
    
    return displayNames[deviceType] || deviceType;
}

// Funzioni di utilità per la gestione dei dati
function validateFormData(data) {
    const requiredFields = [
        'machineType', 'installationYear', 'operationalHours', 
        'temperature', 'vibration', 'sound', 'oilLevel', 
        'coolantLevel', 'powerConsumption', 'lastMaintenance',
        'maintenanceCount', 'failureCount', 'errorCodes', 'aiSupervision'
    ];
    
    for (let field of requiredFields) {
        if (!data[field] || data[field] === '') {
            throw new Error(`Campo obbligatorio mancante: ${field}`);
        }
    }
    
    // Validazione AI Override se supervisione è attiva
    if (data.aiSupervision === '1' && (!data.aiOverrides || data.aiOverrides === '')) {
        throw new Error('Campo "Allarmi AI Ignorati" obbligatorio quando la supervisione AI è attiva');
    }
    
    // Validazione campi speciali per dispositivi con caratteristiche aggiuntive
    if (deviceConfig[data.machineType]) {
        const fieldType = deviceConfig[data.machineType].additionalField;
        let requiredField = null;
        
        switch(fieldType) {
            case 'laser':
                requiredField = 'laserIntensity';
                break;
            case 'hydraulic':
                requiredField = 'hydraulicPressure';
                break;
            case 'coolant':
                requiredField = 'coolantFlow';
                break;
            case 'heat':
                requiredField = 'heatIndex';
                break;
        }
        
        if (requiredField && (!data[requiredField] || data[requiredField] === '')) {
            throw new Error(`Campo obbligatorio mancante per questo tipo di dispositivo: ${requiredField}`);
        }
    }
    
    return true;
}

// Gestione eventi globali
window.addEventListener('beforeunload', function(e) {
    const form = document.getElementById('predictionForm');
    if (form && formHasData(form)) {
        e.preventDefault();
        e.returnValue = '';
    }
});

function formHasData(form) {
    const inputs = form.querySelectorAll('input, select');
    for (let input of inputs) {
        if (input.value && input.value !== '') {
            return true;
        }
    }
    return false;
}

// Gestione ridimensionamento finestra
window.addEventListener('resize', function() {
    if (riskChart) {
        setTimeout(() => {
            try {
                riskChart.resize();
            } catch (error) {
                console.warn('Errore nel ridimensionamento del grafico:', error);
            }
        }, 100);
    }
});

// Debug function per verificare lo stato del grafico
window.debugChart = function() {
    console.log('Chart.js disponibile:', typeof Chart !== 'undefined');
    console.log('Grafico corrente:', riskChart);
    console.log('Canvas element:', document.getElementById('riskChart'));
    console.log('Container element:', document.getElementById('chartContainer'));
};