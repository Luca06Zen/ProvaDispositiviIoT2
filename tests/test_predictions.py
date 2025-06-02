#!/usr/bin/env python3
"""
Test del sistema di predizione con input realistici per diversi tipi di macchine IoT industriali.
Questo file testa il modello con almeno tre scenari realistici come richiesto dal progetto.
"""

import sys
import os
import pandas as pd
import numpy as np
from pathlib import Path

# Aggiungi il percorso src al PYTHONPATH
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root / "src"))

try:
    from prediction_engine import PredictionEngine
    from utils import load_config
except ImportError as e:
    print(f"Errore nell'importazione: {e}")
    print("Assicurati che i file prediction_engine.py e utils.py siano nella cartella src/")
    sys.exit(1)

# Verifica che la struttura del progetto sia corretta
def verify_project_structure():
    """Verifica che la struttura del progetto corrisponda a quella richiesta"""
    required_paths = [
        project_root / "data" / "models",
        project_root / "data" / "processed",
        project_root / "data" / "raw",
        project_root / "src",
        project_root / "tests",
        project_root / "web"
    ]
    
    missing_paths = []
    for path in required_paths:
        if not path.exists():
            missing_paths.append(str(path))
    
    if missing_paths:
        print(f"❌ Percorsi mancanti: {missing_paths}")
        return False
    return True


class TestPredictions:
    """Classe per testare le predizioni del modello con scenari realistici"""
    
    def __init__(self):
        self.prediction_engine = None
        self.test_cases = []
        self.results = []
    
    def setup_prediction_engine(self):
        """Inizializza il motore di predizione"""
        try:
            # Percorso corretto per i modelli
            models_path = project_root / "data" / "models"
            if not models_path.exists():
                print(f"❌ Cartella modelli non trovata: {models_path}")
                return False
                
            self.prediction_engine = PredictionEngine(models_path=str(models_path))
            print("✅ Motore di predizione caricato con successo")
        except Exception as e:
            print(f"❌ Errore nel caricamento del motore di predizione: {e}")
            return False
        return True
    
    def create_test_cases(self):
        """Crea i casi di test con input realistici per diversi tipi di macchine"""
        
        # Caso 1: Laser_Cutter in buone condizioni (bassa probabilità di guasto)
        test_case_1 = {
            'name': 'Laser_Cutter in buone condizioni',
            'description': 'Macchina relativamente nuova, ben mantenuta, con parametri ottimali',
            'data': {
                'Machine_Type': 'Laser_Cutter',
                'Installation_Year': 2023,  # Macchina relativamente nuova
                'Operational_Hours': 800,   # Poche ore operative
                'Temperature_C': 40.0,      # Temperatura normale
                'Vibration_mms': 1.5,       # Vibrazione bassa
                'Sound_dB': 68.5,           # Rumore normale
                'Oil_Level_pct': 95.0,      # Livello olio buono
                'Coolant_Level_pct': 90.0,  # Coolant level alto
                'Power_Consumption_kW': 12.5,    # Consumo normale
                'Last_Maintenance_Days_Ago': 10, # Manutenzione recente
                'Maintenance_History_Count': 1,  # Poche manutenzioni
                'Failure_History_Count': 0,      # Nessun guasto
                'AI_Supervision': 1,             # Con supervisione AI
                'Error_Codes_Last_30_Days': 0,   # Nessun errore
                'AI_Override_Events': 0,         # Nessun override
                'Laser_Intensity': 75.0          # Intensità laser normale
            }
        }
        
        # Caso 2: Hydraulic_Press in condizioni critiche (alta probabilità di guasto)
        test_case_2 = {
            'name': 'Hydraulic_Press in condizioni critiche',
            'description': 'Macchina vecchia, molto utilizzata, con segnali di allarme',
            'data': {
                'Machine_Type': 'Hydraulic_Press',
                'Installation_Year': 2010,  # Macchina vecchia
                'Operational_Hours': 15000, # Ore operative estreme
                'Temperature_C': 95.0,      # Temperatura critica
                'Vibration_mms': 9.5,       # Vibrazione estrema
                'Sound_dB': 95.0,           # Rumore critico
                'Oil_Level_pct': 25.0,      # Livello olio basso
                'Coolant_Level_pct': 30.0,  # Coolant level basso
                'Power_Consumption_kW': 45.8,     # Consumo elevato
                'Last_Maintenance_Days_Ago': 180, # Manutenzione critica
                'Maintenance_History_Count': 35,  # Manutenzioni eccessive
                'Failure_History_Count': 12,      # Molti guasti
                'AI_Supervision': 0,              # Senza supervisione AI
                'Error_Codes_Last_30_Days': 25,   # Errori critici
                'AI_Override_Events': 0,          # Senza supervisione AI
                'Hydraulic_Pressure_bar': 180.5   # Pressione idraulica alta
            }
        }
        
        # Caso 3: CNC_Mill in condizioni intermedie (probabilità media di guasto)
        test_case_3 = {
            'name': 'CNC_Mill in condizioni intermedie',
            'description': 'Macchina di media età con alcuni segnali di usura ma ancora funzionante',
            'data': {
                'Machine_Type': 'CNC_Mill',
                'Installation_Year': 2019,  # Macchina di media età
                'Operational_Hours': 4200,  # Ore medie
                'Temperature_C': 68.8,      # Temperatura leggermente alta
                'Vibration_mms': 4.2,       # Vibrazione media
                'Sound_dB': 76.2,           # Rumore medio-alto
                'Oil_Level_pct': 55.0,      # Livello olio medio-basso
                'Coolant_Level_pct': 65.0,  # Coolant level medio
                'Power_Consumption_kW': 28.3, # Consumo medio-alto
                'Last_Maintenance_Days_Ago': 60,  # Manutenzione non recente
                'Maintenance_History_Count': 12,  # Manutenzioni medie
                'Failure_History_Count': 3,       # Alcuni guasti
                'AI_Supervision': 1,              # Con supervisione AI
                'Error_Codes_Last_30_Days': 5,    # Alcuni errori
                'AI_Override_Events': 2,          # Pochi override
                'Coolant_Flow_L_min': 8.5         # Portata refrigerante normale
            }
        }
        
        # Caso 4: Heat_Exchanger appena installato (probabilità molto bassa di guasto)
        test_case_4 = {
            'name': 'Heat_Exchanger nuovo',
            'description': 'Macchina appena installata, condizioni ottimali',
            'data': {
                'Machine_Type': 'Heat_Exchanger',
                'Installation_Year': 2024,  # Macchina nuovissima
                'Operational_Hours': 150,   # Pochissime ore
                'Temperature_C': 35.2,      # Temperatura bassa
                'Vibration_mms': 1.2,       # Vibrazione minima
                'Sound_dB': 62.1,           # Rumore basso
                'Oil_Level_pct': 95.0,      # Livello olio ottimo
                'Coolant_Level_pct': 98.0,  # Coolant level ottimo
                'Power_Consumption_kW': 8.2, # Consumo basso
                'Last_Maintenance_Days_Ago': 5,   # Manutenzione appena fatta
                'Maintenance_History_Count': 1,   # Prima manutenzione
                'Failure_History_Count': 0,       # Nessun guasto
                'AI_Supervision': 1,              # Con supervisione AI
                'Error_Codes_Last_30_Days': 0,    # Nessun errore
                'AI_Override_Events': 0,          # Nessun override
                'Heat_Index': 42.3                # Indice di calore normale
            }
        }
        
        self.test_cases = [test_case_1, test_case_2, test_case_3, test_case_4]
        print(f"✅ Creati {len(self.test_cases)} casi di test")
    
    def run_single_prediction(self, test_case):
        """Esegue una singola predizione e formatta l'output"""
        try:
            # Prepara i dati per la predizione
            input_data = test_case['data']

            # Campi base comuni a tutte le macchine
            required_fields = ['Machine_Type', 'Installation_Year', 'Operational_Hours', 
                            'Temperature_C', 'Vibration_mms', 'Sound_dB', 'Oil_Level_pct',
                            'Coolant_Level_pct', 'Power_Consumption_kW', 'Last_Maintenance_Days_Ago',
                            'Maintenance_History_Count', 'Failure_History_Count', 'AI_Supervision',
                            'Error_Codes_Last_30_Days', 'AI_Override_Events']
            
            # Aggiungi campi specifici in base al tipo di macchina
            machine_type = test_case['data']['Machine_Type']
            if machine_type == 'Laser_Cutter':
                required_fields.append('Laser_Intensity')
            elif machine_type in ['Hydraulic_Press', 'Injection_Molder']:
                required_fields.append('Hydraulic_Pressure_bar')
            elif machine_type in ['CNC_Lathe', 'CNC_Mill', 'Industrial_Chiller']:
                required_fields.append('Coolant_Flow_L_min')
            elif machine_type in ['Boiler', 'Furnace', 'Heat_Exchanger']:
                required_fields.append('Heat_Index')

            missing_fields = [field for field in required_fields if field not in test_case['data']]
            if missing_fields:
                raise ValueError(f"Campi mancanti: {missing_fields}")
            
            # Esegue la predizione (prediction_engine.predict restituisce un dizionario)
            prediction_result = self.prediction_engine.predict(input_data)
            
            # Estrae i valori dal dizionario
            failure_probability = prediction_result['failure_probability']  # Già in percentuale
            remaining_days = prediction_result['remaining_useful_life_days']
            failure_within_7_days = prediction_result['failure_within_7_days']
            
            # Determina l'azione consigliata basata sulla probabilità
            if failure_probability < 15:
                recommended_action = "nessuna"
            elif failure_probability < 35:
                recommended_action = "controllo"
            elif failure_probability < 50:
                recommended_action = "ispezione"
            elif failure_probability < 70:
                recommended_action = "manutenzione"
            elif failure_probability < 85:
                recommended_action = "riparazione"
            else:
                recommended_action = "sostituzione"
            
            # Formatta l'output esattamente come richiesto nel PDF
            output = (f"{failure_probability:.1f}% di probabilità guasto. "
                f"Azione consigliata: {recommended_action}. "
                f"Giorni di vita rimanenti: {remaining_days}. "
                f"Guasto entro 7 giorni: {'sì' if failure_within_7_days else 'no'}")
            
            return {
                'success': True,
                'output': output,
                'details': {
                    'failure_probability': failure_probability,
                    'recommended_action': recommended_action,
                    'remaining_days': remaining_days,
                    'failure_within_7_days': failure_within_7_days
                }
            }
            
        except Exception as e:
            return {
                'success': False,
                'error': str(e),
                'output': f"Errore nella predizione: {e}"
            }
    
    def run_all_tests(self):
        """Esegue tutti i test di predizione"""
        print("\n" + "="*80)
        print("ESECUZIONE TEST DI PREDIZIONE")
        print("="*80)
        
        for i, test_case in enumerate(self.test_cases, 1):
            print(f"\n--- TEST {i}: {test_case['name']} ---")
            print(f"Descrizione: {test_case['description']}")
            print(f"Tipo macchina: {test_case['data']['Machine_Type']}")
            
            # Mostra alcuni parametri chiave
            key_params = ['Installation_Year', 'Operational_Hours', 'Temperature_C', 
                         'Vibration_mms', 'Oil_Level_pct', 'Last_Maintenance_Days_Ago']
            
            print("Parametri chiave:")
            for param in key_params:
                if param in test_case['data']:
                    print(f"  - {param}: {test_case['data'][param]}")
            
            # Esegue la predizione
            result = self.run_single_prediction(test_case)
            self.results.append({
                'test_case': test_case['name'],
                'result': result
            })
            
            print(f"\n🔮 RISULTATO PREDIZIONE:")
            print(f"   {result['output']}")
            
            if result['success']:
                print("✅ Test completato con successo")
            else:
                print(f"❌ Test fallito: {result.get('error', 'Errore sconosciuto')}")
    
    def print_summary(self):
        """Stampa un riassunto dei risultati"""
        print("\n" + "="*80)
        print("RIASSUNTO RISULTATI TEST")
        print("="*80)
        
        successful_tests = sum(1 for r in self.results if r['result']['success'])
        total_tests = len(self.results)
        
        print(f"Test eseguiti: {total_tests}")
        print(f"Test riusciti: {successful_tests}")
        print(f"Test falliti: {total_tests - successful_tests}")
        
        if successful_tests > 0:
            print(f"\n📊 RIASSUNTO PREDIZIONI:")
            for result in self.results:
                if result['result']['success']:
                    details = result['result']['details']
                    print(f"• {result['test_case']}: {details['failure_probability']:.0f}% rischio, "
                          f"azione: {details['recommended_action']}")
        
        print(f"\n{'='*80}")
        print("Test completati. Verificare che gli output siano realistici e coerenti.")
    
    def save_results_to_file(self, filename="test_predictions_results.txt"):
        """Salva i risultati in un file"""
        try:
            results_path = project_root / filename
            with open(results_path, 'w', encoding='utf-8') as f:
                f.write("RISULTATI TEST DI PREDIZIONE\n")
                f.write("="*50 + "\n\n")
                
                for i, result in enumerate(self.results, 1):
                    f.write(f"TEST {i}: {result['test_case']}\n")
                    f.write("-" * 30 + "\n")
                    f.write(f"Risultato: {result['result']['output']}\n")
                    if result['result']['success']:
                        f.write("Status: ✅ Successo\n")
                    else:
                        f.write(f"Status: ❌ Errore - {result['result'].get('error', 'N/A')}\n")
                    f.write("\n")
                
                f.write(f"\nTest eseguiti il: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                
            print(f"📄 Risultati salvati in: {results_path}")
        except Exception as e:
            print(f"❌ Errore nel salvataggio: {e}")


def main():
    """Funzione principale per eseguire tutti i test"""
    print("🚀 Avvio test di predizione per il sistema IoT industriale")
    print("="*80)

    # Verifica la struttura del progetto
    if not verify_project_structure():
        print("❌ Struttura del progetto non corretta. Verifica i percorsi.")
        return

    # Inizializza il sistema di test
    tester = TestPredictions()
    
    # Setup del motore di predizione
    if not tester.setup_prediction_engine():
        print("❌ Impossibile inizializzare il motore di predizione. Verifica i file del modello.")
        return
    
    # Crea i casi di test
    tester.create_test_cases()
    
    # Esegue tutti i test
    tester.run_all_tests()
    
    # Stampa il riassunto
    tester.print_summary()
    
    # Salva i risultati
    tester.save_results_to_file()
    
    print("\n🎯 Test di predizione completati!")


if __name__ == "__main__":
    main()
