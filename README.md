# Struttura del Progetto - Industrial Machine Failure Prediction

## Struttura delle cartelle:
```
industrial_failure_prediction/
├── data/
│   ├── raw/
│   │   └── industrial_iot_dataset.csv
│   ├── processed/
│   │   ├── cleaned_data.csv
│   │   ├── train_data.csv
│   │   └── test_data.csv
│   └── models/
│       ├── best_classification_model.pkl
│       ├── best_regression_model.pkl
│       ├── label_encoder.pkl
│       ├── model_info.json
│       └── scaler.pkl
├── src/
│   ├── __pycache__/
│   │   ├── prediction_engine.cpython-311.pyc
│   │   └── utils.cpython-311.pyc
│   ├── data_preprocessing.py
│   ├── model_training.py
│   ├── model_evaluation.py
│   ├── prediction_engine.py
│   └── utils.py
├── web/
│   ├── static/
│   │   ├── css/
│   │   │   └── style.css
│   │   ├── js/
│   │   │   └── main.js
│   │   └── images/
│   │       ├── logo.png
│   │       └── favicon.ico
│   └── templates/
│       ├── index.html
│       └── results.html
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   ├── 02_data_preprocessing.ipynb
│   ├── 03_model_development.ipynb
│   └── 04_model_evaluation.ipynb
├── tests/
│   ├── test_model.py
│   ├── test_preprocessing.py
│   └── test_predictions.py
├── app.py
├── requirements.txt
├── setup.py
├── README.md
└── config.yaml
```

## Descrizione dei file principali:

### Directory `src/`:
- **data_preprocessing.py**: Pulizia e preparazione dei dati, gestione valori nulli, normalizzazione
- **model_training.py**: Training di diversi algoritmi (Random Forest, XGBoost, SVM)
- **model_evaluation.py**: Valutazione prestazioni con metriche appropriate
- **prediction_engine.py**: Motore di predizione per nuovi input
- **utils.py**: Funzioni di utilità comuni

### Directory `web/`:
- **app.py**: Applicazione Flask per l'interfaccia web
- **templates/index.html**: Pagina principale con form di input
- **static/css/style.css**: Styling dell'interfaccia
- **static/js/main.js**: Logica JavaScript lato client

### Directory `notebooks/`:
- **01_data_exploration.ipynb**: Analisi esplorativa del dataset
- **02_data_preprocessing.ipynb**: Preprocessing dettagliato con visualizzazioni
- **03_model_development.ipynb**: Sviluppo e confronto modelli
- **04_model_evaluation.ipynb**: Valutazione finale e interpretazione

## Dataset Scelto:
**Machine Predictive Maintenance Classification** - Un dataset sintetico che simula dati reali di macchine industriali con:
- 10.000 campioni di dati di sensori
- Multiple tipologie di macchine (L, M, H quality variants)
- Features: temperatura, velocità di rotazione, torque, usura utensili
- Target: failure prediction e tipo di guasto

## Algoritmi implementati:
1. **Random Forest** (principale) - Robusto e interpretabile
2. **XGBoost** - Alte prestazioni su dati tabulari
3. **Support Vector Machine** - Buono per classificazione binaria
4. **Logistic Regression** - Baseline interpretabile

## Metriche di valutazione:
- Accuracy, Precision, Recall, F1-Score
- ROC-AUC per classificazione binaria
- Confusion Matrix
- Feature Importance Analysis

## Installazioni richieste:
```bash
pip install -r requirements.txt
```

Le librerie principali includono:
- pandas, numpy (manipolazione dati)
- scikit-learn (machine learning)
- xgboost (gradient boosting)
- flask (web framework)
- plotly, seaborn (visualizzazioni)
- joblib (serializzazione modelli)
```
