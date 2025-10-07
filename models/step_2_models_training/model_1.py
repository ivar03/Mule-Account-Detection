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
    
    def __init__(self, contamination=0.15, random_state=42):
        """
        Initialize Isolation Forest model
        
        Parameters:
        -----------
        contamination : float
            Expected proportion of outliers in dataset (0.15 = 15%)
        random_state : int
            Random seed for reproducibility
        """
        self.model = IsolationForest(
            contamination=contamination,
            random_state=random_state,
            n_estimators=100,
            max_samples='auto',
            n_jobs=-1,
            verbose=1
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
        # Exclude identifier and target columns
        exclude_cols = [
            'address', 'wallet_id', 'type', 'pattern', 'is_illicit', 'created_at'
        ]
        
        # Select all numerical columns except excluded ones
        numerical_cols = df.select_dtypes(include=[np.number]).columns.tolist()
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
        print("\n" + "="*60)
        print("MODEL 1: ISOLATION FOREST - TRAINING")
        print("="*60)
        
        # Select features
        self.feature_columns = self.select_features(df)
        X = df[self.feature_columns].fillna(0)
        
        print(f"Training on {len(X)} wallets with {len(self.feature_columns)} features")
        print(f"Features: {self.feature_columns[:5]}... (showing first 5)")
        
        # Scale features
        print("\nScaling features...")
        X_scaled = self.scaler.fit_transform(X)
        
        # Fit Isolation Forest
        print("Training Isolation Forest...")
        self.model.fit(X_scaled)
        self.is_fitted = True
        
        print("\n" + "="*60)
        print("MODEL 1: TRAINING COMPLETE")
        print("="*60)
        
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
            - model1_anomaly_score: Anomaly score (higher = more anomalous)
            - model1_is_anomaly: Binary flag (1 = anomaly, 0 = normal)
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")
        
        print("\n" + "="*60)
        print("MODEL 1: ISOLATION FOREST - PREDICTION")
        print("="*60)
        
        # Prepare features
        X = df[self.feature_columns].fillna(0)
        X_scaled = self.scaler.transform(X)
        
        # Predict anomalies (-1 = anomaly, 1 = normal)
        print("Detecting anomalies...")
        predictions = self.model.predict(X_scaled)
        
        # Get anomaly scores (more negative = more anomalous)
        print("Computing anomaly scores...")
        anomaly_scores_raw = self.model.score_samples(X_scaled)
        
        # Normalize scores to 0-1 range (1 = most anomalous)
        anomaly_scores = self._normalize_scores(anomaly_scores_raw)
        
        # Add results to dataframe
        result_df = df.copy()
        result_df['model1_anomaly_score'] = anomaly_scores
        result_df['model1_is_anomaly'] = (predictions == -1).astype(int)
        
        # Statistics
        anomaly_count = result_df['model1_is_anomaly'].sum()
        anomaly_pct = (anomaly_count / len(result_df)) * 100
        
        # Check against actual labels if available
        if 'is_illicit' in result_df.columns:
            illicit_detected = result_df[result_df['is_illicit'] == True]['model1_is_anomaly'].sum()
            total_illicit = result_df['is_illicit'].sum()
            detection_rate = (illicit_detected / total_illicit * 100) if total_illicit > 0 else 0
            
            print(f"\nDetection Statistics:")
            print(f"  Total anomalies detected: {anomaly_count} ({anomaly_pct:.2f}%)")
            print(f"  Actual illicit wallets: {total_illicit}")
            print(f"  Illicit wallets detected: {illicit_detected} ({detection_rate:.2f}%)")
            print(f"  Mean anomaly score: {anomaly_scores.mean():.4f}")
            print(f"  Std anomaly score: {anomaly_scores.std():.4f}")
        else:
            print(f"\nDetection Statistics:")
            print(f"  Anomalies detected: {anomaly_count} ({anomaly_pct:.2f}%)")
            print(f"  Mean anomaly score: {anomaly_scores.mean():.4f}")
        
        print("\n" + "="*60)
        print("MODEL 1: PREDICTION COMPLETE")
        print("="*60)
        
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
    
    def get_top_anomalies(self, df_with_predictions, n=100):
        """
        Get top N most anomalous wallets
        
        Parameters:
        -----------
        df_with_predictions : pd.DataFrame
            DataFrame with model predictions
        n : int
            Number of top anomalies to return
            
        Returns:
        --------
        pd.DataFrame : Top N anomalies sorted by anomaly score
        """
        return df_with_predictions.nlargest(n, 'model1_anomaly_score')
    
    def save_model(self, filepath='models/saved_models/model1_isolation_forest.pkl'):
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
        print(f"\nModel 1 saved to {filepath}")
    
    def load_model(self, filepath='models/saved_models/model1_isolation_forest.pkl'):
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
        
        print(f"Model 1 loaded from {filepath}")
        
        return self


# ============================================================================
# USAGE EXAMPLE
# ============================================================================

if __name__ == "__main__":
    # Load feature-engineered data
    print("Loading feature-engineered dataset...")
    features_df = pd.read_csv('models/step_1_feature_engineering/refined_data/blockchain_wallet_features.csv')
    
    print(f"Loaded {len(features_df)} wallets with {len(features_df.columns)} columns")
    print(f"\nDataset info:")
    print(f"  Total wallets: {len(features_df)}")
    print(f"  Illicit wallets: {features_df['is_illicit'].sum()}")
    print(f"  Legitimate wallets: {(~features_df['is_illicit']).sum()}")
    
    # Initialize and train Model 1
    model1 = OutlierDetectionModel(contamination=0.15, random_state=42)
    model1.fit(features_df)
    
    # Predict anomalies
    result_df = model1.predict(features_df)
    
    # Show top 10 most anomalous wallets
    print("\nTop 10 most anomalous wallets:")
    top_anomalies = model1.get_top_anomalies(result_df, n=10)
    print(top_anomalies[['address', 'pattern', 'is_illicit', 'model1_anomaly_score', 'model1_is_anomaly']].to_string())
    
    # Save results
    output_path = 'data/step_2_models_training/model1_output.csv'
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    result_df.to_csv(output_path, index=False)
    print(f"\nResults saved to {output_path}")
    
    # Save model
    model1.save_model()
    
    print("\nModel 1 pipeline complete!")