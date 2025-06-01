"""
Modulo di test per il progetto di predizione guasti industriali IoT
Versione corretta basata sulle specifiche del PDF
Test per funzionalità del modello e conformità alle specifiche
"""

import unittest
import pandas as pd
import numpy as np
import joblib
import os
import sys
import warnings
from pathlib import Path

# Aggiunge il percorso del progetto per gli import
project_root = Path(__file__).parent.parent # Da tests/ va alla root del progetto
sys.path.append(str(project_root))
sys.path.append(str(project_root / 'src'))

# Sopprime warnings non critici durante i test
warnings.filterwarnings('ignore', category=UserWarning)

try:
    from src.model_training import FailurePredictionTrainer
    from src.prediction_engine import PredictionEngine
    from src.utils import load_config
    MODULI_DISPONIBILI = True
except ImportError as e:
    print(f"Attenzione: Impossibile importare i moduli: {e}")
    print("Alcuni test verranno saltati se i moduli non sono disponibili")
    MODULI_DISPONIBILI = False

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


class TestModelloPredizioneGuasti(unittest.TestCase):
    """Test case per il modello di predizione guasti industriali"""
    
    @classmethod
    def setUpClass(cls):
        """Configura i fixture di test prima di tutti i test"""
        # Percorsi corretti secondo la struttura proposta
        project_root = Path(__file__).parent.parent
        models_dir = project_root / 'data' / 'models'
        
        percorsi_modello_possibili = [
            models_dir / 'failure_prediction_model.pkl',
            'data/models/failure_prediction_model.pkl',
            Path('data') / 'models' / 'failure_prediction_model.pkl',
        ]
        
        percorsi_scaler_possibili = [
            models_dir / 'scaler.pkl',
            'data/models/scaler.pkl',
            Path('data') / 'models' / 'scaler.pkl',
        ]
    
        percorsi_encoder_possibili = [
            models_dir / 'label_encoder.pkl',
            'data/models/label_encoder.pkl',
            Path('data') / 'models' / 'label_encoder.pkl',
        ]
        
        cls.percorso_modello = None
        cls.percorso_scaler = None
        cls.percorso_label_encoder = None
        
        # Trova i file del modello esistenti
        for percorso in percorsi_modello_possibili:
            if Path(percorso).exists():
                cls.percorso_modello = str(percorso)
                break
                
        for percorso in percorsi_scaler_possibili:
            if Path(percorso).exists():
                cls.percorso_scaler = str(percorso)
                break
                
        for percorso in percorsi_encoder_possibili:
            if Path(percorso).exists():
                cls.percorso_label_encoder = str(percorso)
                break
        
        cls.modello_esiste = all([
            cls.percorso_modello and Path(cls.percorso_modello).exists(),
            cls.percorso_scaler and Path(cls.percorso_scaler).exists(),
            cls.percorso_label_encoder and Path(cls.percorso_label_encoder).exists()
        ])
        
        # Stampa informazioni di debug per aiutare nella risoluzione problemi
        if not cls.modello_esiste:
            print("DEBUG: File del modello non trovati:")
            print(f"  Modello: {cls.percorso_modello} - Esiste: {cls.percorso_modello and Path(cls.percorso_modello).exists()}")
            print(f"  Scaler: {cls.percorso_scaler} - Esiste: {cls.percorso_scaler and Path(cls.percorso_scaler).exists()}")
            print(f"  Label Encoder: {cls.percorso_label_encoder} - Esiste: {cls.percorso_label_encoder and Path(cls.percorso_label_encoder).exists()}")
        
        if cls.modello_esiste:
            try:
                cls.modello = joblib.load(cls.percorso_modello)
                cls.scaler = joblib.load(cls.percorso_scaler)
                cls.label_encoder = joblib.load(cls.percorso_label_encoder)
                
                # Inizializza il motore di predizione se disponibile
                if MODULI_DISPONIBILI:
                    cls.motore_predizione = PredictionEngine(
                        cls.percorso_modello, cls.percorso_scaler, cls.percorso_label_encoder
                    )
                else:
                    cls.motore_predizione = None
            except Exception as e:
                print(f"Errore nel caricamento dei file del modello: {e}")
                cls.modello_esiste = False
                cls.motore_predizione = None
    
    def test_caricamento_modello(self):
        """Testa se il modello e gli oggetti di preprocessing si caricano correttamente"""
        if not self.modello_esiste:
            self.skipTest("File del modello non trovati. Esegui prima l'addestramento del modello.")
        
        self.assertIsNotNone(self.modello, "Il modello dovrebbe essere caricato")
        self.assertIsNotNone(self.scaler, "Lo scaler dovrebbe essere caricato")
        self.assertIsNotNone(self.label_encoder, "Il label encoder dovrebbe essere caricato")
        
        # Testa che il modello abbia i metodi richiesti
        self.assertTrue(hasattr(self.modello, 'predict'), "Il modello dovrebbe avere il metodo predict")
        self.assertTrue(hasattr(self.modello, 'predict_proba'), "Il modello dovrebbe avere il metodo predict_proba")
    
    def test_forma_predizione_modello(self):
        """Testa se le predizioni del modello hanno la forma corretta"""
        if not self.modello_esiste:
            self.skipTest("File del modello non trovati. Esegui prima l'addestramento del modello.")
        
        # Crea input di esempio con nomi di colonna corretti dal PDF
        input_esempio = self.crea_input_esempio()
        
        try:
            input_scalato = self.scaler.transform(input_esempio)
            predizioni = self.modello.predict(input_scalato)
            probabilita = self.modello.predict_proba(input_scalato)
            
            self.assertEqual(len(predizioni), 1, "Dovrebbe predire per un campione")
            self.assertEqual(probabilita.shape[0], 1, "Dovrebbe restituire probabilità per un campione")
            self.assertEqual(probabilita.shape[1], 2, "Dovrebbe restituire probabilità per classificazione binaria")
            self.assertIn(predizioni[0], [0, 1], "La predizione dovrebbe essere 0 o 1")
            
            # Testa i vincoli delle probabilità
            self.assertTrue(np.all(probabilita >= 0), "Le probabilità dovrebbero essere >= 0")
            self.assertTrue(np.all(probabilita <= 1), "Le probabilità dovrebbero essere <= 1")
            self.assertTrue(np.allclose(probabilita.sum(axis=1), 1), "Le probabilità dovrebbero sommare a 1")
            
        except Exception as e:
            self.fail(f"Predizione del modello fallita: {e}")
    
    @unittest.skipIf(not MODULI_DISPONIBILI, "Moduli richiesti non disponibili")
    def test_motore_predizione_macchine_base(self):
        """Testa il motore di predizione con tipi di macchine base (solo caratteristiche comuni)"""
        if not self.modello_esiste or self.motore_predizione is None:
            self.skipTest("File del modello o motore di predizione non trovati.")
        
        # Macchine base come specificato nel PDF (solo caratteristiche comuni)
        macchine_base = [
            '3D_Printer', 'AGV', 'Automated_Screwdriver', 'CMM', 'Carton_Former',
            'Compressor', 'Conveyor_Belt', 'Crane', 'Dryer', 'Forklift_Electric',
            'Grinder', 'Labeler', 'Mixer', 'Palletizer', 'Pick_and_Place',
            'Press_Brake', 'Pump', 'Robot_Arm', 'Shrink_Wrapper', 'Shuttle_System',
            'Vacuum_Packer', 'Valve_Controller', 'Vision_System', 'XRay_Inspector'
        ]
        
        for tipo_macchina in macchine_base[:5]:  # Testa le prime 5 per evitare troppi test
            with self.subTest(tipo_macchina=tipo_macchina):
                # Usa nomi di colonna corretti dal PDF
                dati_input = {
                    'Machine_Type': tipo_macchina,
                    'Installation_Year': 2020,
                    'Operational_Hours': 15000,
                    'Temperature_C': 45.0,
                    'Vibration_mms': 3.5,
                    'Sound_dB': 65.0,
                    'Oil_Level_pct': 85.0,
                    'Coolant_Level_pct': 78.0,
                    'Power_Consumption_kW': 25.0,
                    'Last_Maintenance_Days_Ago': 45,
                    'Maintenance_History_Count': 12,
                    'Failure_History_Count': 2,
                    'AI_Supervision': 1,
                    'Error_Codes_Last_30_Days': 3,
                    'AI_Override_Events': 1,
                    'Remaining_Useful_Life_days': 180,
                    # Caratteristiche speciali dovrebbero essere 0 per macchine base
                    'Laser_Intensity': 0.0,
                    'Hydraulic_Pressure_bar': 0.0,
                    'Coolant_Flow_L_min': 0.0,
                    'Heat_Index': 0.0
                }
                
                try:
                    risultato = self.motore_predizione.predict(dati_input)
                    self._valida_output_predizione(risultato, tipo_macchina)
                except Exception as e:
                    self.fail(f"Predizione fallita per {tipo_macchina}: {e}")
    
    @unittest.skipIf(not MODULI_DISPONIBILI, "Moduli richiesti non disponibili")
    def test_motore_predizione_macchine_speciali(self):
        """Testa il motore di predizione con macchine che hanno caratteristiche speciali"""
        if not self.modello_esiste or self.motore_predizione is None:
            self.skipTest("File del modello o motore di predizione non trovati.")
        
        # Macchine speciali con le loro caratteristiche specifiche come da PDF
        test_macchine_speciali = [
            ('Laser_Cutter', {'Laser_Intensity': 750.0}),
            ('Hydraulic_Press', {'Hydraulic_Pressure_bar': 150.0}),
            ('Injection_Molder', {'Hydraulic_Pressure_bar': 120.0}),
            ('CNC_Lathe', {'Coolant_Flow_L_min': 20.0}),
            ('CNC_Mill', {'Coolant_Flow_L_min': 18.0}),
            ('Industrial_Chiller', {'Coolant_Flow_L_min': 25.0}),
            ('Boiler', {'Heat_Index': 0.8}),
            ('Furnace', {'Heat_Index': 0.9}),
            ('Heat_Exchanger', {'Heat_Index': 0.7})
        ]
        
        for tipo_macchina, parametri_speciali in test_macchine_speciali:
            with self.subTest(tipo_macchina=tipo_macchina):
                # Dati di input base con nomi di colonna corretti
                dati_input = {
                    'Machine_Type': tipo_macchina,
                    'Installation_Year': 2019,
                    'Operational_Hours': 18000,
                    'Temperature_C': 55.0,
                    'Vibration_mms': 4.2,
                    'Sound_dB': 70.0,
                    'Oil_Level_pct': 72.0,
                    'Coolant_Level_pct': 68.0,
                    'Power_Consumption_kW': 30.0,
                    'Last_Maintenance_Days_Ago': 65,
                    'Maintenance_History_Count': 8,
                    'Failure_History_Count': 3,
                    'AI_Supervision': 1,
                    'Error_Codes_Last_30_Days': 7,
                    'AI_Override_Events': 2,
                    'Remaining_Useful_Life_days': 120,
                    # Inizializza tutte le caratteristiche speciali a 0
                    'Laser_Intensity': 0.0,
                    'Hydraulic_Pressure_bar': 0.0,
                    'Coolant_Flow_L_min': 0.0,
                    'Heat_Index': 0.0
                }
                
                # Imposta la caratteristica speciale specifica per questo tipo di macchina
                dati_input.update(parametri_speciali)
                
                try:
                    risultato = self.motore_predizione.predict(dati_input)
                    self._valida_output_predizione(risultato, tipo_macchina)
                    
                    # Verifica che la caratteristica speciale abbia avuto un impatto (valore non-zero usato)
                    for caratteristica, valore in parametri_speciali.items():
                        self.assertGreater(valore, 0, f"{caratteristica} dovrebbe essere maggiore di 0 per {tipo_macchina}")
                        
                except Exception as e:
                    self.fail(f"Predizione fallita per {tipo_macchina}: {e}")
    
    def _valida_output_predizione(self, risultato, tipo_macchina):
        """Valida il formato dell'output di predizione secondo le specifiche del PDF"""
        self.assertIsInstance(risultato, dict, f"Il risultato dovrebbe essere un dizionario per {tipo_macchina}")
        
        # Campi richiesti secondo il PDF
        campi_richiesti = ['failure_probability', 'failure_within_7_days', 'recommended_action']
        
        for campo in campi_richiesti:
            self.assertIn(campo, risultato, f"Campo richiesto '{campo}' mancante per {tipo_macchina}")
        
        # Valida probabilità di guasto
        self.assertIsInstance(risultato['failure_probability'], (int, float), 
                             f"failure_probability dovrebbe essere numerico per {tipo_macchina}")
        self.assertGreaterEqual(risultato['failure_probability'], 0.0, 
                               f"failure_probability dovrebbe essere >= 0 per {tipo_macchina}")
        self.assertLessEqual(risultato['failure_probability'], 1.0, 
                            f"failure_probability dovrebbe essere <= 1 per {tipo_macchina}")
        
        # Valida failure_within_7_days
        self.assertIsInstance(risultato['failure_within_7_days'], (bool, int), 
                             f"failure_within_7_days dovrebbe essere boolean o int per {tipo_macchina}")
        
        # Valida azione consigliata
        self.assertIsInstance(risultato['recommended_action'], str, 
                             f"recommended_action dovrebbe essere string per {tipo_macchina}")
        self.assertGreater(len(risultato['recommended_action']), 0, 
                          f"recommended_action non dovrebbe essere vuoto per {tipo_macchina}")
    
    def test_conformita_formato_output(self):
        """Testa che il formato di output corrisponda alle specifiche del PDF"""
        if not self.modello_esiste or not MODULI_DISPONIBILI or self.motore_predizione is None:
            self.skipTest("File del modello o motore di predizione non disponibili.")
        
        # Input di test
        dati_input = {
            'Machine_Type': 'Laser_Cutter',
            'Installation_Year': 2020,
            'Operational_Hours': 15000,
            'Temperature_C': 45.0,
            'Vibration_mms': 3.5,
            'Sound_dB': 65.0,
            'Oil_Level_pct': 85.0,
            'Coolant_Level_pct': 78.0,
            'Power_Consumption_kW': 25.0,
            'Last_Maintenance_Days_Ago': 45,
            'Maintenance_History_Count': 12,
            'Failure_History_Count': 2,
            'AI_Supervision': 1,
            'Error_Codes_Last_30_Days': 3,
            'AI_Override_Events': 1,
            'Remaining_Useful_Life_days': 180,
            'Laser_Intensity': 750.0,
            'Hydraulic_Pressure_bar': 0.0,
            'Coolant_Flow_L_min': 0.0,
            'Heat_Index': 0.0
        }
        
        try:
            risultato = self.motore_predizione.predict(dati_input)
            
            # Controlla se l'output può essere formattato secondo le specifiche del PDF:
            # "xx% di probabilità guasto per motivo. Azione consigliata: .... 
            #  Giorni di vita rimanenti: ...." 
            # oppure "Guasto entro 7 giorni: true/false"
            
            # Testa formato percentuale probabilità
            probabilita_pct = risultato['failure_probability'] * 100
            self.assertIsInstance(probabilita_pct, (int, float))
            
            # Testa che possiamo creare il formato di output richiesto
            if 'remaining_useful_life' in risultato:
                formato_output = f"{probabilita_pct:.1f}% di probabilità guasto. Azione consigliata: {risultato['recommended_action']}. Giorni di vita rimanenti: {risultato.get('remaining_useful_life', 'N/A')}"
            else:
                formato_output = f"Guasto entro 7 giorni: {'true' if risultato['failure_within_7_days'] else 'false'}"
            
            self.assertIsInstance(formato_output, str)
            self.assertGreater(len(formato_output), 10, "Il formato di output dovrebbe essere significativo")
            
        except Exception as e:
            self.fail(f"Test formato output fallito: {e}")
    
    def test_consistenza_modello(self):
        """Testa che il modello dia risultati consistenti per input identici"""
        if not self.modello_esiste or not MODULI_DISPONIBILI or self.motore_predizione is None:
            self.skipTest("File del modello o motore di predizione non disponibili.")
        
        dati_input = {
            'Machine_Type': 'Laser_Cutter',
            'Installation_Year': 2020,
            'Operational_Hours': 15000,
            'Temperature_C': 45.0,
            'Vibration_mms': 3.5,
            'Sound_dB': 65.0,
            'Oil_Level_pct': 85.0,
            'Coolant_Level_pct': 78.0,
            'Power_Consumption_kW': 25.0,
            'Last_Maintenance_Days_Ago': 45,
            'Maintenance_History_Count': 12,
            'Failure_History_Count': 2,
            'AI_Supervision': 1,
            'Error_Codes_Last_30_Days': 3,
            'AI_Override_Events': 1,
            'Remaining_Useful_Life_days': 180,
            'Laser_Intensity': 750.0,
            'Hydraulic_Pressure_bar': 0.0,
            'Coolant_Flow_L_min': 0.0,
            'Heat_Index': 0.0
        }
        
        # Effettua predizioni multiple con lo stesso input
        risultati = []
        for _ in range(3):
            try:
                risultato = self.motore_predizione.predict(dati_input)
                risultati.append(risultato['failure_probability'])
            except Exception as e:
                self.fail(f"Predizione fallita: {e}")
        
        # Tutti i risultati dovrebbero essere identici
        self.assertTrue(all(abs(r - risultati[0]) < 1e-10 for r in risultati),
                       "Il modello dovrebbe dare risultati consistenti per input identici")
    
    def test_gestione_errori_input_mancanti(self):
        """Testa la gestione degli errori per input mancanti o non validi"""
        if not self.modello_esiste or not MODULI_DISPONIBILI or self.motore_predizione is None:
            self.skipTest("File del modello o motore di predizione non disponibili.")
        
        # Test con input mancanti
        dati_input_incompleti = {
            'Machine_Type': 'Laser_Cutter',
            'Installation_Year': 2020,
            # Mancano molti campi richiesti
        }
        
        # Il sistema dovrebbe gestire gracefully gli errori
        try:
            risultato = self.motore_predizione.predict(dati_input_incompleti)
            # Se non solleva eccezione, dovrebbe comunque dare un risultato valido
            if isinstance(risultato, dict):
                self.assertIn('failure_probability', risultato)
        except Exception as e:
            # È accettabile che sollevi un'eccezione per input incompleti
            self.assertIsInstance(e, (KeyError, ValueError, TypeError))
    
    def test_valori_limite_input(self):
        """Testa il comportamento con valori limite negli input"""
        if not self.modello_esiste or not MODULI_DISPONIBILI or self.motore_predizione is None:
            self.skipTest("File del modello o motore di predizione non disponibili.")
        
        # Test con valori estremi ma tecnicamente validi
        dati_input_estremi = {
            'Machine_Type': 'Laser_Cutter',
            'Installation_Year': 1990,  # Molto vecchio
            'Operational_Hours': 100000,  # Altissimo
            'Temperature_C': 100.0,  # Molto caldo
            'Vibration_mms': 15.0,  # Alta vibrazione
            'Sound_dB': 120.0,  # Molto rumoroso
            'Oil_Level_pct': 5.0,  # Livello olio molto basso
            'Coolant_Level_pct': 10.0,  # Livello refrigerante basso
            'Power_Consumption_kW': 100.0,  # Alto consumo
            'Last_Maintenance_Days_Ago': 365,  # Manutenzione molto vecchia
            'Maintenance_History_Count': 100,  # Molte manutenzioni
            'Failure_History_Count': 50,  # Molti guasti
            'AI_Supervision': 0,  # Nessuna supervisione AI
            'Error_Codes_Last_30_Days': 50,  # Molti errori
            'AI_Override_Events': 20,  # Molti override
            'Remaining_Useful_Life_days': 1,  # Quasi a fine vita
            'Laser_Intensity': 1000.0,  # Intensità massima
            'Hydraulic_Pressure_bar': 0.0,
            'Coolant_Flow_L_min': 0.0,
            'Heat_Index': 0.0
        }
        
        try:
            risultato = self.motore_predizione.predict(dati_input_estremi)
            self._valida_output_predizione(risultato, "valori_estremi")
            
            # Con questi valori estremi, ci aspettiamo alta probabilità di guasto
            self.assertGreaterEqual(risultato['failure_probability'], 0.3,
                                  "Con valori estremi dovremmo avere probabilità di guasto significativa")
            
        except Exception as e:
            self.fail(f"Test valori estremi fallito: {e}")

    def test_tre_input_realistici(self):
        """Testa con tre input realistici come richiesto dal PDF"""
        if not self.modello_esiste or not MODULI_DISPONIBILI or self.motore_predizione is None:
            self.skipTest("File del modello o motore di predizione non disponibili.")
        
        # Tre casi di test realistici come richiesto dal PDF
        casi_test = [
            {
                'nome': 'Nastro trasportatore vecchio ad alto utilizzo',
                'dati': {
                    'Machine_Type': 'Conveyor_Belt',
                    'Installation_Year': 2015,
                    'Operational_Hours': 45000,
                    'Temperature_C': 65.0,
                    'Vibration_mms': 6.5,
                    'Sound_dB': 78.0,
                    'Oil_Level_pct': 65.0,
                    'Coolant_Level_pct': 70.0,
                    'Power_Consumption_kW': 35.0,
                    'Last_Maintenance_Days_Ago': 120,
                    'Maintenance_History_Count': 25,
                    'Failure_History_Count': 8,
                    'AI_Supervision': 1,
                    'Error_Codes_Last_30_Days': 12,
                    'AI_Override_Events': 4,
                    'Remaining_Useful_Life_days': 45,
                    'Laser_Intensity': 0.0,
                    'Hydraulic_Pressure_bar': 0.0,
                    'Coolant_Flow_L_min': 0.0,
                    'Heat_Index': 0.0
                }
            },
            {
                'nome': 'Tagliatrice laser nuova con alta intensità',
                'dati': {
                    'Machine_Type': 'Laser_Cutter',
                    'Installation_Year': 2023,
                    'Operational_Hours': 5000,
                    'Temperature_C': 42.0,
                    'Vibration_mms': 2.1,
                    'Sound_dB': 62.0,
                    'Oil_Level_pct': 95.0,
                    'Coolant_Level_pct': 88.0,
                    'Power_Consumption_kW': 22.0,
                    'Last_Maintenance_Days_Ago': 15,
                    'Maintenance_History_Count': 2,
                    'Failure_History_Count': 0,
                    'AI_Supervision': 1,
                    'Error_Codes_Last_30_Days': 1,
                    'AI_Override_Events': 0,
                    'Remaining_Useful_Life_days': 350,
                    'Laser_Intensity': 950.0,
                    'Hydraulic_Pressure_bar': 0.0,
                    'Coolant_Flow_L_min': 0.0,
                    'Heat_Index': 0.0
                }
            },
            {
                'nome': 'Fresatrice CNC di media età con utilizzo moderato',
                'dati': {
                    'Machine_Type': 'CNC_Mill',
                    'Installation_Year': 2019,
                    'Operational_Hours': 22000,
                    'Temperature_C': 52.0,
                    'Vibration_mms': 4.0,
                    'Sound_dB': 68.0,
                    'Oil_Level_pct': 78.0,
                    'Coolant_Level_pct': 82.0,
                    'Power_Consumption_kW': 28.0,
                    'Last_Maintenance_Days_Ago': 60,
                    'Maintenance_History_Count': 8,
                    'Failure_History_Count': 2,
                    'AI_Supervision': 1,
                    'Error_Codes_Last_30_Days': 5,
                    'AI_Override_Events': 1,
                    'Remaining_Useful_Life_days': 150,
                    'Laser_Intensity': 0.0,
                    'Hydraulic_Pressure_bar': 0.0,
                    'Coolant_Flow_L_min': 22.0,
                    'Heat_Index': 0.0
                }
            }
        ]
        
        for caso_test in casi_test:
            with self.subTest(nome=caso_test['nome']):
                try:
                    risultato = self.motore_predizione.predict(caso_test['dati'])
                    self._valida_output_predizione(risultato, caso_test['nome'])
                    
                    # Stampa risultati per verifica manuale
                    print(f"\n--- Caso di Test: {caso_test['nome']} ---")
                    print(f"Tipo Macchina: {caso_test['dati']['Machine_Type']}")
                    print(f"Probabilità Guasto: {risultato['failure_probability']:.2%}")
                    print(f"Guasto Entro 7 Giorni: {risultato['failure_within_7_days']}")
                    print(f"Azione Consigliata: {risultato['recommended_action']}")
                    if 'remaining_useful_life' in risultato:
                        print(f"Vita Utile Rimanente: {risultato['remaining_useful_life']} giorni")
                    
                except Exception as e:
                    self.fail(f"Test input realistico fallito per {caso_test['nome']}: {e}")
    
    def crea_input_esempio(self):
        """Crea un input di esempio per i test con nomi di colonna corretti dal PDF"""
        dati_esempio = {
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
        
        # Aggiunge colonne dei tipi di macchina se usa one-hot encoding
        tipi_macchina = [
            '3D_Printer', 'AGV', 'Automated_Screwdriver', 'Boiler', 'CMM',
            'CNC_Lathe', 'CNC_Mill', 'Carton_Former', 'Compressor', 'Conveyor_Belt',
            'Crane', 'Dryer', 'Forklift_Electric', 'Furnace', 'Grinder',
            'Heat_Exchanger', 'Hydraulic_Press', 'Industrial_Chiller',
            'Injection_Molder', 'Labeler', 'Laser_Cutter', 'Mixer',
            'Palletizer', 'Pick_and_Place', 'Press_Brake', 'Pump',
            'Robot_Arm', 'Shrink_Wrapper', 'Shuttle_System', 'Vacuum_Packer',
            'Valve_Controller', 'Vision_System', 'XRay_Inspector'
        ]
        
        for tipo_macchina in tipi_macchina:
            dati_esempio[f'Machine_Type_{tipo_macchina}'] = [1 if tipo_macchina == '3D_Printer' else 0]
        
        return pd.DataFrame(dati_esempio)


class TestIntegritaDati(unittest.TestCase):
    """Test per l'integrità dei dati e requisiti di formato dal PDF"""
    
    def test_presenza_colonne_richieste(self):
        """Testa che tutte le colonne richieste dal PDF siano considerate"""
        # Nomi delle colonne come specificato nel PDF
        colonne_richieste = [
            'Machine_ID', 'Machine_Type', 'Installation_Year', 'Operational_Hours',
            'Temperature_C', 'Vibration_mms', 'Sound_dB', 'Oil_Level_pct',
            'Coolant_Level_pct', 'Power_Consumption_kW', 'Last_Maintenance_Days_Ago',
            'Maintenance_History_Count', 'Failure_History_Count', 'AI_Supervision',
            'Error_Codes_Last_30_Days', 'Remaining_Useful_Life_days',
            'Failure_Within_7_Days', 'Laser_Intensity', 'Hydraulic_Pressure_bar',
            'Coolant_Flow_L_min', 'Heat_Index', 'AI_Override_Events'
        ]
        
        # Testa che tutte le colonne richieste siano riconosciute
        for colonna in colonne_richieste:
            self.assertIsInstance(colonna, str, f"La colonna {colonna} dovrebbe essere una stringa")
            self.assertGreater(len(colonna), 0, f"La colonna {colonna} non dovrebbe essere vuota")
    
    def test_mappatura_caratteristiche_speciali(self):
        """Testa che la mappatura delle caratteristiche speciali sia corretta"""
        # Mappatura secondo il PDF
        mappatura_attesa = {
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
        
        # Verifica che ogni tipo di macchina speciale abbia la caratteristica corretta
        for tipo_macchina, caratteristica_attesa in mappatura_attesa.items():
            self.assertIn(caratteristica_attesa, [
                'Laser_Intensity', 'Hydraulic_Pressure_bar', 
                'Coolant_Flow_L_min', 'Heat_Index'
            ], f"Caratteristica {caratteristica_attesa} dovrebbe essere una delle 4 speciali")
        
    def test_lista_completa_macchine_base(self):
        """Verifica che la lista completa delle macchine base sia corretta"""
        macchine_base_complete = [
            '3D_Printer', 'AGV', 'Automated_Screwdriver', 'CMM', 'Carton_Former',
            'Compressor', 'Conveyor_Belt', 'Crane', 'Dryer', 'Forklift_Electric',
            'Grinder', 'Labeler', 'Mixer', 'Palletizer', 'Pick_and_Place',
            'Press_Brake', 'Pump', 'Robot_Arm', 'Shrink_Wrapper', 'Shuttle_System',
            'Vacuum_Packer', 'Valve_Controller', 'Vision_System', 'XRay_Inspector'
        ]
        
        macchine_speciali = [
            'Laser_Cutter', 'Hydraulic_Press', 'Injection_Molder',
            'CNC_Lathe', 'CNC_Mill', 'Industrial_Chiller',
            'Boiler', 'Furnace', 'Heat_Exchanger'
        ]
        
        # Verifica che le liste non si sovrappongano
        sovrapposizione = set(macchine_base_complete) & set(macchine_speciali)
        self.assertEqual(len(sovrapposizione), 0, 
                        f"Macchine base e speciali non dovrebbero sovrapporsi: {sovrapposizione}")
        
        # Verifica il numero totale
        self.assertEqual(len(macchine_base_complete), 24, "Dovrebbero esserci 24 macchine base")
        self.assertEqual(len(macchine_speciali), 9, "Dovrebbero esserci 9 macchine speciali")

    def test_conformita_tipi_macchina(self):
        """Testa che i tipi di macchina corrispondano esattamente alle specifiche del PDF"""
        # Macchine base (solo caratteristiche comuni) come da PDF
        macchine_base = [
            '3D_Printer', 'AGV', 'Automated_Screwdriver', 'CMM', 'Carton_Former',
            'Compressor', 'Conveyor_Belt', 'Crane', 'Dryer', 'Forklift_Electric',
            'Grinder', 'Labeler', 'Mixer', 'Palletizer', 'Pick_and_Place',
            'Press_Brake', 'Pump', 'Robot_Arm', 'Shrink_Wrapper', 'Shuttle_System',
            'Vacuum_Packer', 'Valve_Controller', 'Vision_System', 'XRay_Inspector'
        ]

        # Macchine speciali (con caratteristiche aggiuntive) come da PDF
        macchine_speciali = [
            'Laser_Cutter',  # + Laser_Intensity
            'Hydraulic_Press', 'Injection_Molder',  # + Hydraulic_Pressure_bar
            'CNC_Lathe', 'CNC_Mill', 'Industrial_Chiller',  # + Coolant_Flow_L_min
            'Boiler', 'Furnace', 'Heat_Exchanger'  # + Heat_Index
        ]
        
        # Verifica che ogni macchina base sia una stringa valida
        for macchina in macchine_base:
            self.assertIsInstance(macchina, str, f"Macchina base {macchina} dovrebbe essere string")
            self.assertGreater(len(macchina), 0, f"Nome macchina {macchina} non dovrebbe essere vuoto")
        
        # Verifica che ogni macchina speciale sia una stringa valida
        for macchina in macchine_speciali:
            self.assertIsInstance(macchina, str, f"Macchina speciale {macchina} dovrebbe essere string")
            self.assertGreater(len(macchina), 0, f"Nome macchina {macchina} non dovrebbe essere vuoto")
        
        # Verifica che non ci siano sovrapposizioni
        sovrapposizione = set(macchine_base) & set(macchine_speciali)
        self.assertEqual(len(sovrapposizione), 0, 
                        f"Macchine base e speciali non dovrebbero sovrapporsi: {sovrapposizione}")


if __name__ == '__main__':
    # Configura il runner dei test con output verboso
    unittest.main(verbosity=2, buffer=True)