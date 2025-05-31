"""
Model training module for Industrial Machine Failure Prediction
Implements multiple ML algorithms and hyperparameter tuning
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.svm import SVC, SVR
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import xgboost as xgb
import joblib
import os
import logging
from datetime import datetime

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ModelTrainer:
    """Class to handle model training and selection"""
    
    def __init__(self):
        self.models = {}
        self.best_models = {}
        self.training_history = []
        
    def define_models(self):
        """Define the models to be trained"""
        self.models = {
            'classification': {
                'RandomForest': {
                    'model': RandomForestClassifier(random_state=42),
                    'params': {
                        'n_estimators': [100, 200, 300],
                        'max_depth': [10, 15, 20, None],
                        'min_samples_split': [5, 10, 15],
                        'min_samples_leaf': [2, 5, 10]
                    }
                },
                'XGBoost': {
                    'model': xgb.XGBClassifier(random_state=42),
                    'params': {
                        'n_estimators': [100, 200, 300],
                        'max_depth': [3, 6, 9],
                        'learning_rate': [0.01, 0.1, 0.2],
                        'subsample': [0.8, 0.9, 1.0]
                    }
                },
                'SVM': {
                    'model': SVC(random_state=42, probability=True),
                    'params': {
                        'C': [0.1, 1, 10],
                        'kernel': ['rbf', 'linear'],
                        'gamma': ['scale', 'auto']
                    }
                },
                'LogisticRegression': {
                    'model': LogisticRegression(random_state=42, max_iter=1000),
                    'params': {
                        'C': [0.1, 1, 10, 100],
                        'penalty': ['l1', 'l2'],
                        'solver': ['liblinear', 'saga']
                    }
                }
            },
            'regression': {
                'RandomForest': {
                    'model': RandomForestRegressor(random_state=42),
                    'params': {
                        'n_estimators': [100, 200, 300],
                        'max_depth': [10, 15, 20, None],
                        'min_samples_split': [5, 10, 15],
                        'min_samples_leaf': [2, 5, 10]
                    }
                },
                'XGBoost': {
                    'model': xgb.XGBRegressor(random_state=42),
                    'params': {
                        'n_estimators': [100, 200, 300],
                        'max_depth': [3, 6, 9],
                        'learning_rate': [0.01, 0.1, 0.2],
                        'subsample': [0.8, 0.9, 1.0]
                    }
                },
                'SVR': {
                    'model': SVR(),
                    'params': {
                        'C': [0.1, 1, 10],
                        'kernel': ['rbf', 'linear'],
                        'gamma': ['scale', 'auto']
                    }
                },
                'LinearRegression': {
                    'model': LinearRegression(),
                    'params': {}  # No hyperparameters to tune
                }
            }
        }
    
    def train_classification_models(self, X_train, y_train, X_test, y_test):
        """Train and evaluate classification models"""
        logger.info("Training classification models...")
        
        classification_results = {}
        
        for model_name, model_config in self.models['classification'].items():
            logger.info(f"Training {model_name}...")
            
            if model_config['params']:
                # Perform grid search for hyperparameter tuning
                grid_search = GridSearchCV(
                    model_config['model'],
                    model_config['params'],
                    cv=5,
                    scoring='roc_auc',
                    n_jobs=-1,
                    verbose=0
                )
                grid_search.fit(X_train, y_train)
                best_model = grid_search.best_estimator_
                
                logger.info(f"Best parameters for {model_name}: {grid_search.best_params_}")
            else:
                # No hyperparameters to tune
                best_model = model_config['model']
                best_model.fit(X_train, y_train)
            
            # Make predictions
            y_pred = best_model.predict(X_test)
            y_pred_proba = best_model.predict_proba(X_test)[:, 1] if hasattr(best_model, 'predict_proba') else None
            
            # Calculate metrics
            from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
            
            accuracy = accuracy_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred)
            recall = recall_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred)
            roc_auc = roc_auc_score(y_test, y_pred_proba) if y_pred_proba is not None else None
            
            classification_results[model_name] = {
                'model': best_model,
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1_score': f1,
                'roc_auc': roc_auc,
                'predictions': y_pred,
                'probabilities': y_pred_proba
            }
            
            logger.info(f"{model_name} - Accuracy: {accuracy:.4f}, F1: {f1:.4f}, ROC-AUC: {roc_auc:.4f if roc_auc else 'N/A'}")
        
        return classification_results
    
    def train_regression_models(self, X_train, y_train, X_test, y_test):
        """Train and evaluate regression models"""
        logger.info("Training regression models...")
        
        regression_results = {}
        
        for model_name, model_config in self.models['regression'].items():
            logger.info(f"Training {model_name} for regression...")
            
            if model_config['params']:
                # Perform grid search for hyperparameter tuning
                grid_search = GridSearchCV(
                    model_config['model'],
                    model_config['params'],
                    cv=5,
                    scoring='neg_mean_squared_error',
                    n_jobs=-1,
                    verbose=0
                )
                grid_search.fit(X_train, y_train)
                best_model = grid_search.best_estimator_
                
                logger.info(f"Best parameters for {model_name}: {grid_search.best_params_}")
            else:
                # No hyperparameters to tune
                best_model = model_config['model']
                best_model.fit(X_train, y_train)
            
            # Make predictions
            y_pred = best_model.predict(X_test)
            
            # Calculate metrics
            mse = mean_squared_error(y_test, y_pred)
            rmse = np.sqrt(mse)
            mae = mean_absolute_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)
            
            regression_results[model_name] = {
                'model': best_model,
                'mse': mse,
                'rmse': rmse,
                'mae': mae,
                'r2_score': r2,
                'predictions': y_pred
            }
            
            logger.info(f"{model_name} - RMSE: {rmse:.4f}, MAE: {mae:.4f}, R2: {r2:.4f}")
        
        return regression_results
    
    def select_best_models(self, classification_results, regression_results):
        """Select the best performing models"""
        # Select best classification model based on F1 score
        best_classification = max(
            classification_results.items(),
            key=lambda x: x[1]['f1_score']
        )
        
        # Select best regression model based on R2 score
        best_regression = max(
            regression_results.items(),
            key=lambda x: x[1]['r2_score']
        )
        
        self.best_models = {
            'classification': {
                'name': best_classification[0],
                'model': best_classification[1]['model'],
                'metrics': {k: v for k, v in best_classification[1].items() if k != 'model'}
            },
            'regression': {
                'name': best_regression[0],
                'model': best_regression[1]['model'],
                'metrics': {k: v for k, v in best_regression[1].items() if k != 'model'}
            }
        }
        
        logger.info(f"Best classification model: {best_classification[0]}")
        logger.info(f"Best regression model: {best_regression[0]}")
        
        return self.best_models
    
    def get_feature_importance(self, model, feature_names):
        """Extract feature importance from the model"""
        if hasattr(model, 'feature_importances_'):
            importance = model.feature_importances_
            feature_importance = pd.DataFrame({
                'feature': feature_names,
                'importance': importance
            }).sort_values('importance', ascending=False)
            
            return feature_importance
        else:
            logger.warning("Model does not have feature_importances_ attribute")
            return None
    
    def save_models(self, save_dir):
        """Save the trained models"""
        os.makedirs(save_dir, exist_ok=True)
        
        # Save best models
        joblib.dump(
            self.best_models['classification']['model'],
            os.path.join(save_dir, 'best_classification_model.pkl')
        )
        
        joblib.dump(
            self.best_models['regression']['model'],
            os.path.join(save_dir, 'best_regression_model.pkl')
        )
        
        # Save model metadata
        model_info = {
            'classification': {
                'name': self.best_models['classification']['name'],
                'metrics': self.best_models['classification']['metrics']
            },
            'regression': {
                'name': self.best_models['regression']['name'],
                'metrics': self.best_models['regression']['metrics']
            },
            'training_date': datetime.now().isoformat()
        }
        
        import json
        with open(os.path.join(save_dir, 'model_info.json'), 'w') as f:
            json.dump(model_info, f, indent=2, default=str)
        
        logger.info(f"Models saved to {save_dir}")
    
    def load_models(self, save_dir):
        """Load the trained models"""
        self.best_models = {
            'classification': {
                'model': joblib.load(os.path.join(save_dir, 'best_classification_model.pkl'))
            },
            'regression': {
                'model': joblib.load(os.path.join(save_dir, 'best_regression_model.pkl'))
            }
        }
        
        # Load model metadata
        import json
        with open(os.path.join(save_dir, 'model_info.json'), 'r') as f:
            model_info = json.load(f)
        
        self.best_models['classification'].update(model_info['classification'])
        self.best_models['regression'].update(model_info['regression'])
        
        logger.info(f"Models loaded from {save_dir}")


def main():
    """Main training pipeline"""
    # Load preprocessed data
    train_data = pd.read_csv('data/processed/train_data.csv')
    test_data = pd.read_csv('data/processed/test_data.csv')
    
    # Separate features and targets
    feature_cols = [col for col in train_data.columns if col not in [
        'Failure_Within_7_Days', 'Remaining_Useful_Life_days'
    ]]
    
    X_train = train_data[feature_cols]
    y_class_train = train_data['Failure_Within_7_Days']
    y_reg_train = train_data['Remaining_Useful_Life_days']
    
    X_test = test_data[feature_cols]
    y_class_test = test_data['Failure_Within_7_Days']
    y_reg_test = test_data['Remaining_Useful_Life_days']
    
    # Initialize trainer
    trainer = ModelTrainer()
    trainer.define_models()
    
    # Train models
    classification_results = trainer.train_classification_models(
        X_train, y_class_train, X_test, y_class_test
    )
    
    regression_results = trainer.train_regression_models(
        X_train, y_reg_train, X_test, y_reg_test
    )
    
    # Select best models
    best_models = trainer.select_best_models(classification_results, regression_results)
    
    # Get feature importance
    for model_type in ['classification', 'regression']:
        model = best_models[model_type]['model']
        feature_importance = trainer.get_feature_importance(model, feature_cols)
        
        if feature_importance is not None:
            logger.info(f"\nTop 10 important features for {model_type}:")
            logger.info(feature_importance.head(10).to_string(index=False))
    
    # Save models
    trainer.save_models('data/models')
    
    logger.info("Model training completed successfully!")


if __name__ == "__main__":
    main()