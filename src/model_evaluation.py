"""
Model Evaluation Module for IoT Predictive Maintenance System
Evaluates trained models using various metrics and generates performance reports
"""

import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix, roc_auc_score, roc_curve
)
from sklearn.model_selection import cross_val_score
import warnings
warnings.filterwarnings('ignore')

class ModelEvaluator:
    def __init__(self, model_path='data/models/failure_prediction_model.pkl'):
        """Initialize the model evaluator"""
        self.model_path = model_path
        self.model = None
        self.test_data = None
        self.test_labels = None
        
    def load_model(self):
        """Load the trained model"""
        try:
            with open(self.model_path, 'rb') as f:
                self.model = pickle.load(f)
            print("Model loaded successfully")
        except FileNotFoundError:
            raise FileNotFoundError(f"Model file not found at {self.model_path}")
    
    def load_test_data(self, test_data_path='data/processed/test_data.csv'):
        """Load test data"""
        try:
            data = pd.read_csv(test_data_path)
            # Separate features and target
            if 'Failure_Within_7_Days' in data.columns:
                self.test_labels = data['Failure_Within_7_Days']
                self.test_data = data.drop(['Failure_Within_7_Days'], axis=1)
            else:
                raise ValueError("Target column 'Failure_Within_7_Days' not found")
            print(f"Test data loaded: {self.test_data.shape}")
        except FileNotFoundError:
            raise FileNotFoundError(f"Test data file not found at {test_data_path}")
    
    def evaluate_classification(self):
        """Evaluate classification performance"""
        if self.model is None or self.test_data is None:
            raise ValueError("Model and test data must be loaded first")
        
        # Make predictions
        y_pred = self.model.predict(self.test_data)
        y_pred_proba = self.model.predict_proba(self.test_data)[:, 1] if hasattr(self.model, 'predict_proba') else None
        
        # Calculate metrics
        metrics = {
            'accuracy': accuracy_score(self.test_labels, y_pred),
            'precision': precision_score(self.test_labels, y_pred),
            'recall': recall_score(self.test_labels, y_pred),
            'f1_score': f1_score(self.test_labels, y_pred),
        }
        
        if y_pred_proba is not None:
            metrics['roc_auc'] = roc_auc_score(self.test_labels, y_pred_proba)
        
        return metrics, y_pred, y_pred_proba
    
    def generate_classification_report(self):
        """Generate detailed classification report"""
        if self.model is None or self.test_data is None:
            raise ValueError("Model and test data must be loaded first")
        
        y_pred = self.model.predict(self.test_data)
        
        report = classification_report(
            self.test_labels, 
            y_pred, 
            target_names=['No Failure', 'Failure Within 7 Days'],
            output_dict=True
        )
        
        return report
    
    def plot_confusion_matrix(self, save_path=None):
        """Plot confusion matrix"""
        if self.model is None or self.test_data is None:
            raise ValueError("Model and test data must be loaded first")
        
        y_pred = self.model.predict(self.test_data)
        cm = confusion_matrix(self.test_labels, y_pred)
        
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=['No Failure', 'Failure Within 7 Days'],
                   yticklabels=['No Failure', 'Failure Within 7 Days'])
        plt.title('Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
        
        return cm
    
    def plot_roc_curve(self, save_path=None):
        """Plot ROC curve"""
        if self.model is None or self.test_data is None:
            raise ValueError("Model and test data must be loaded first")
        
        if not hasattr(self.model, 'predict_proba'):
            print("Model doesn't support probability predictions. Skipping ROC curve.")
            return None
        
        y_pred_proba = self.model.predict_proba(self.test_data)[:, 1]
        fpr, tpr, _ = roc_curve(self.test_labels, y_pred_proba)
        auc_score = roc_auc_score(self.test_labels, y_pred_proba)
        
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, color='darkorange', lw=2, 
                label=f'ROC curve (AUC = {auc_score:.2f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic (ROC) Curve')
        plt.legend(loc="lower right")
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
        
        return auc_score
    
    def cross_validation_score(self, cv=5):
        """Perform cross-validation"""
        if self.model is None:
            raise ValueError("Model must be loaded first")
        
        # Load full training data for cross-validation
        try:
            train_data = pd.read_csv('data/processed/train_data.csv')
            y_train = train_data['Failure_Within_7_Days']
            X_train = train_data.drop(['Failure_Within_7_Days'], axis=1)
            
            cv_scores = cross_val_score(self.model, X_train, y_train, cv=cv, scoring='f1')
            
            return {
                'cv_scores': cv_scores,
                'mean_cv_score': cv_scores.mean(),
                'std_cv_score': cv_scores.std()
            }
        except FileNotFoundError:
            print("Training data not found for cross-validation")
            return None
    
    def feature_importance_analysis(self, feature_names=None):
        """Analyze feature importance if supported by the model"""
        if self.model is None:
            raise ValueError("Model must be loaded first")
        
        if hasattr(self.model, 'feature_importances_'):
            importances = self.model.feature_importances_
            
            if feature_names is None:
                feature_names = [f'Feature_{i}' for i in range(len(importances))]
            
            # Create feature importance dataframe
            feature_imp = pd.DataFrame({
                'feature': feature_names,
                'importance': importances
            }).sort_values('importance', ascending=False)
            
            # Plot feature importance
            plt.figure(figsize=(12, 8))
            sns.barplot(data=feature_imp.head(15), x='importance', y='feature')
            plt.title('Top 15 Feature Importances')
            plt.xlabel('Importance')
            plt.tight_layout()
            plt.show()
            
            return feature_imp
        else:
            print("Model doesn't support feature importance analysis")
            return None
    
    def generate_evaluation_report(self, save_path='evaluation_report.txt'):
        """Generate comprehensive evaluation report"""
        if self.model is None or self.test_data is None:
            raise ValueError("Model and test data must be loaded first")
        
        # Get evaluation metrics
        metrics, y_pred, y_pred_proba = self.evaluate_classification()
        classification_rep = self.generate_classification_report()
        
        # Generate report
        report_lines = []
        report_lines.append("=" * 60)
        report_lines.append("IoT PREDICTIVE MAINTENANCE - MODEL EVALUATION REPORT")
        report_lines.append("=" * 60)
        report_lines.append(f"Test Set Size: {len(self.test_data)} samples")
        report_lines.append(f"Model Type: {type(self.model).__name__}")
        report_lines.append("")
        
        report_lines.append("CLASSIFICATION METRICS:")
        report_lines.append("-" * 30)
        for metric, value in metrics.items():
            report_lines.append(f"{metric.capitalize()}: {value:.4f}")
        report_lines.append("")
        
        report_lines.append("DETAILED CLASSIFICATION REPORT:")
        report_lines.append("-" * 40)
        report_lines.append(f"{'Class':<20} {'Precision':<10} {'Recall':<10} {'F1-Score':<10}")
        report_lines.append("-" * 50)
        
        for class_name, class_metrics in classification_rep.items():
            if class_name in ['0', '1']:
                class_label = 'No Failure' if class_name == '0' else 'Failure (7 days)'
                report_lines.append(
                    f"{class_label:<20} {class_metrics['precision']:<10.4f} "
                    f"{class_metrics['recall']:<10.4f} {class_metrics['f1-score']:<10.4f}"
                )
        
        # Cross-validation results
        cv_results = self.cross_validation_score()
        if cv_results:
            report_lines.append("")
            report_lines.append("CROSS-VALIDATION RESULTS:")
            report_lines.append("-" * 30)
            report_lines.append(f"Mean CV Score (F1): {cv_results['mean_cv_score']:.4f}")
            report_lines.append(f"Std CV Score: {cv_results['std_cv_score']:.4f}")
        
        report_lines.append("")
        report_lines.append("=" * 60)
        
        # Save report
        with open(save_path, 'w') as f:
            f.write('\n'.join(report_lines))
        
        print(f"Evaluation report saved to {save_path}")
        return '\n'.join(report_lines)

def main():
    """Main function to run model evaluation"""
    evaluator = ModelEvaluator()
    
    try:
        # Load model and test data
        evaluator.load_model()
        evaluator.load_test_data()
        
        # Generate evaluation report
        report = evaluator.generate_evaluation_report()
        print(report)
        
        # Generate visualizations
        print("\nGenerating confusion matrix...")
        evaluator.plot_confusion_matrix()
        
        print("\nGenerating ROC curve...")
        evaluator.plot_roc_curve()
        
        print("\nAnalyzing feature importance...")
        evaluator.feature_importance_analysis()
        
    except Exception as e:
        print(f"Error during evaluation: {str(e)}")

if __name__ == "__main__":
    main()
