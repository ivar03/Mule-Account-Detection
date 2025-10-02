import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import os

class PatternClassificationModel:
    """
    MODEL 4: XGBoost Multi-class Classifier
    Classifies wallets into laundering patterns: 
    legitimate, layering, smurfing, mixing, dormant-active
    """
    
    def __init__(self, random_state=42):
        """
        Initialize XGBoost Classifier
        
        Parameters:
        -----------
        random_state : int
            Random seed for reproducibility
        """
        self.model = xgb.XGBClassifier(
            n_estimators=200,
            max_depth=8,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=random_state,
            eval_metric='mlogloss',
            use_label_encoder=False,
            n_jobs=-1
        )
        self.label_encoder = LabelEncoder()
        self.feature_columns = None
        self.class_names = None
        self.is_fitted = False
        
    def prepare_data(self, df):
        """
        Prepare data for classification
        
        Parameters:
        -----------
        df : pd.DataFrame
            Data with selected features from model3
            
        Returns:
        --------
        tuple : (X, y, feature_names)
        """
        # Exclude identifier and target columns
        exclude_cols = ['wallet_address', 'is_laundering', 'laundering_type']
        
        feature_cols = [col for col in df.columns if col not in exclude_cols]
        
        # Check if target variable exists
        if 'laundering_type' not in df.columns:
            raise ValueError("Target variable 'laundering_type' not found in data")
        
        X = df[feature_cols].fillna(0)
        y = df['laundering_type']
        
        self.feature_columns = feature_cols
        
        return X, y, feature_cols
    
    def fit(self, df, test_size=0.2):
        """
        Train XGBoost classifier
        
        Parameters:
        -----------
        df : pd.DataFrame
            Data with selected features from model3
        test_size : float
            Proportion of data for testing
            
        Returns:
        --------
        self
        """
        X, y, feature_names = self.prepare_data(df)
        
        # Encode labels
        y_encoded = self.label_encoder.fit_transform(y)
        self.class_names = self.label_encoder.classes_
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y_encoded, test_size=test_size, random_state=42, stratify=y_encoded
        )
        
        # Train model
        self.model.fit(
            X_train, y_train,
            eval_set=[(X_test, y_test)],
            verbose=False
        )
        
        # Evaluate on test set
        y_pred = self.model.predict(X_test)
        
        self.is_fitted = True
        
        print(f"✓ Model 4 training complete:")
        print(f"  Classes: {', '.join(self.class_names)}")
        print(f"  Train samples: {len(X_train)}, Test samples: {len(X_test)}")
        print(f"\nClassification Report:")
        print(classification_report(y_test, y_pred, target_names=self.class_names))
        
        return self
    
    def predict(self, df):
        """
        Classify wallets into laundering patterns
        
        Parameters:
        -----------
        df : pd.DataFrame
            Data with selected features from model3
            
        Returns:
        --------
        pd.DataFrame : Original data with added columns:
            - model4_predicted_class: Predicted laundering pattern
            - model4_confidence: Confidence score for prediction
            - model4_prob_{class}: Probability for each class
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")
        
        X = df[self.feature_columns].fillna(0)
        
        # Predict classes
        y_pred_encoded = self.model.predict(X)
        y_pred = self.label_encoder.inverse_transform(y_pred_encoded)
        
        # Predict probabilities
        y_proba = self.model.predict_proba(X)
        
        # Add predictions to dataframe
        result_df = df.copy()
        result_df['model4_predicted_class'] = y_pred
        result_df['model4_confidence'] = y_proba.max(axis=1)
        
        # Add probability for each class
        for idx, class_name in enumerate(self.class_names):
            result_df[f'model4_prob_{class_name}'] = y_proba[:, idx]
        
        # Calculate class distribution
        class_counts = pd.Series(y_pred).value_counts()
        
        print(f"✓ Model 4 classification complete:")
        print(f"  Mean confidence: {result_df['model4_confidence'].mean():.4f}")
        print(f"\n  Class distribution:")
        for class_name, count in class_counts.items():
            pct = (count / len(result_df)) * 100
            print(f"    {class_name:<20} {count:>6} ({pct:>5.2f}%)")
        
        return result_df
    
    def get_feature_importance(self, top_n=20):
        """
        Get feature importance from trained model
        
        Parameters:
        -----------
        top_n : int
            Number of top features to return
            
        Returns:
        --------
        pd.DataFrame : Feature importance rankings
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted first")
        
        importance = self.model.feature_importances_
        
        feature_importance = pd.DataFrame({
            'feature': self.feature_columns,
            'importance': importance
        }).sort_values('importance', ascending=False)
        
        return feature_importance.head(top_n)
    
    def plot_confusion_matrix(self, df, figsize=(10, 8), save_path=None):
        """
        Plot confusion matrix
        
        Parameters:
        -----------
        df : pd.DataFrame
            Data with true labels
        figsize : tuple
            Figure size
        save_path : str, optional
            Path to save plot
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted first")
        
        if 'laundering_type' not in df.columns:
            print("Warning: True labels not available, cannot plot confusion matrix")
            return
        
        X = df[self.feature_columns].fillna(0)
        y_true = self.label_encoder.transform(df['laundering_type'])
        y_pred = self.model.predict(X)
        
        cm = confusion_matrix(y_true, y_pred)
        
        plt.figure(figsize=figsize)
        sns.heatmap(
            cm, 
            annot=True, 
            fmt='d', 
            cmap='Blues',
            xticklabels=self.class_names,
            yticklabels=self.class_names
        )
        plt.title('Confusion Matrix - XGBoost Classifier', fontsize=16)
        plt.ylabel('True Label', fontsize=12)
        plt.xlabel('Predicted Label', fontsize=12)
        plt.tight_layout()
        
        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✓ Plot saved to {save_path}")
        
        plt.show()
    
    def save_model(self, filepath='models/model4_xgboost_classifier.pkl'):
        """
        Save trained model to disk
        
        Parameters:
        -----------
        filepath : str
            Path to save model
        """
        if not self.is_fitted:
            raise ValueError("Cannot save unfitted model")
        
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        model_data = {
            'model': self.model,
            'label_encoder': self.label_encoder,
            'feature_columns': self.feature_columns,
            'class_names': self.class_names,
            'is_fitted': self.is_fitted
        }
        
        joblib.dump(model_data, filepath)
        print(f"✓ Model 4 saved to {filepath}")
    
    def load_model(self, filepath='models/model4_xgboost_classifier.pkl'):
        """
        Load trained model from disk
        
        Parameters:
        -----------
        filepath : str
            Path to load model from
        """
        model_data = joblib.load(filepath)
        
        self.model = model_data['model']
        self.label_encoder = model_data['label_encoder']
        self.feature_columns = model_data['feature_columns']
        self.class_names = model_data['class_names']
        self.is_fitted = model_data['is_fitted']
        
        print(f"✓ Model 4 loaded from {filepath}")
        
        return self


# Example usage
if __name__ == "__main__":
    # Load model3 output
    # df = pd.read_csv('data/model3_output.csv')
    
    # Initialize and train model
    model4 = PatternClassificationModel()
    # model4.fit(df)
    
    # Classify patterns
    # result_df = model4.predict(df)
    
    # Plot confusion matrix
    # model4.plot_confusion_matrix(df, save_path='outputs/confusion_matrix.png')
    
    # Save results
    # result_df.to_csv('data/model4_output.csv', index=False)
    # model4.save_model()
    
    pass