import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
import joblib
import os

class OutlierDetectionModel:
    """
    MODEL 1: Isolation Forest for Anomaly Detection
    Detects statistically unusual wallets based on transaction patterns
    """
    
    def __init__(self, contamination=0.1, random_state=42):
        """
        Initialize Isolation Forest model
        
        Parameters:
        -----------
        contamination : float
            Expected proportion of outliers in dataset (0.1 = 10%)
        random_state : int
            Random seed for reproducibility
        """
        self.model = IsolationForest(
            contamination=contamination,
            random_state=random_state,
            n_estimators=100,
            max_samples='auto',
            n_jobs=-1
        )
        self.scaler = StandardScaler()
        self.feature_columns = None
        self.is_fitted = False
        
    def select_features(self, df):
        """
        Select numerical features for anomaly detection
        
        Parameters:
        -----------
        df : pd.DataFrame
            Feature engineered dataset
            
        Returns:
        --------
        list : Selected feature column names
        """
        # Select numerical transaction-based features
        numerical_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        
        # Remove target/identifier columns if present
        exclude_cols = ['wallet_address', 'is_laundering', 'laundering_type', 
                       'timestamp', 'block_number']
        
        selected_features = [col for col in numerical_cols if col not in exclude_cols]
        
        return selected_features
    
    def fit(self, df):
        """
        Train the Isolation Forest model
        
        Parameters:
        -----------
        df : pd.DataFrame
            Feature engineered dataset from feature engineering module
            
        Returns:
        --------
        self
        """
        # Select features
        self.feature_columns = self.select_features(df)
        X = df[self.feature_columns].fillna(0)
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        # Fit Isolation Forest
        self.model.fit(X_scaled)
        self.is_fitted = True
        
        print(f"✓ Model 1 trained on {len(self.feature_columns)} features")
        print(f"  Features: {', '.join(self.feature_columns[:5])}...")
        
        return self
    
    def predict(self, df):
        """
        Detect anomalies in wallet data
        
        Parameters:
        -----------
        df : pd.DataFrame
            Feature engineered dataset
            
        Returns:
        --------
        pd.DataFrame : Original data with added columns:
            - anomaly_score: Anomaly score (higher = more anomalous)
            - is_anomaly: Binary flag (1 = anomaly, 0 = normal)
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")
        
        # Prepare features
        X = df[self.feature_columns].fillna(0)
        X_scaled = self.scaler.transform(X)
        
        # Predict anomalies (-1 = anomaly, 1 = normal)
        predictions = self.model.predict(X_scaled)
        
        # Get anomaly scores (more negative = more anomalous)
        anomaly_scores_raw = self.model.score_samples(X_scaled)
        
        # Normalize scores to 0-1 range (1 = most anomalous)
        anomaly_scores = self._normalize_scores(anomaly_scores_raw)
        
        # Add results to dataframe
        result_df = df.copy()
        result_df['model1_anomaly_score'] = anomaly_scores
        result_df['model1_is_anomaly'] = (predictions == -1).astype(int)
        
        anomaly_count = result_df['model1_is_anomaly'].sum()
        anomaly_pct = (anomaly_count / len(result_df)) * 100
        
        print(f"✓ Model 1 detection complete:")
        print(f"  Anomalies detected: {anomaly_count} ({anomaly_pct:.2f}%)")
        print(f"  Mean anomaly score: {anomaly_scores.mean():.4f}")
        
        return result_df
    
    def _normalize_scores(self, scores):
        """
        Normalize anomaly scores to 0-1 range
        
        Parameters:
        -----------
        scores : np.array
            Raw anomaly scores from Isolation Forest (more negative = more anomalous)
            
        Returns:
        --------
        np.array : Normalized scores (0-1, where 1 = most anomalous)
        """
        # Invert scores so higher values indicate anomalies
        inverted_scores = -scores
        
        # Normalize to 0-1
        min_score = inverted_scores.min()
        max_score = inverted_scores.max()
        
        if max_score == min_score:
            return np.ones_like(scores) * 0.5
        
        normalized = (inverted_scores - min_score) / (max_score - min_score)
        
        return normalized
    
    def save_model(self, filepath='models/model1_isolation_forest.pkl'):
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
            'scaler': self.scaler,
            'feature_columns': self.feature_columns,
            'is_fitted': self.is_fitted
        }
        
        joblib.dump(model_data, filepath)
        print(f"✓ Model 1 saved to {filepath}")
    
    def load_model(self, filepath='models/model1_isolation_forest.pkl'):
        """
        Load trained model from disk
        
        Parameters:
        -----------
        filepath : str
            Path to load model from
        """
        model_data = joblib.load(filepath)
        
        self.model = model_data['model']
        self.scaler = model_data['scaler']
        self.feature_columns = model_data['feature_columns']
        self.is_fitted = model_data['is_fitted']
        
        print(f"✓ Model 1 loaded from {filepath}")
        
        return self


# Example usage
if __name__ == "__main__":
    # Load feature-engineered data (output from feature engineering module)
    # df = pd.read_csv('data/features_engineered.csv')
    
    # Initialize and train model
    model1 = OutlierDetectionModel(contamination=0.1)
    # model1.fit(df)
    
    # Detect anomalies
    # result_df = model1.predict(df)
    
    # Save results for next model
    # result_df.to_csv('data/model1_output.csv', index=False)
    
    # Save model
    # model1.save_model()
    
    pass