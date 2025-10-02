import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, classification_report
import joblib
import os

class EnsembleStackingModel:
    """
    MODEL 7: Ensemble Stacking for Final Risk Scoring
    Combines outputs from Models 1-6 into a final risk score
    """
    
    def __init__(self, random_state=42):
        """
        Initialize Ensemble Stacking model
        
        Parameters:
        -----------
        random_state : int
            Random seed for reproducibility
        """
        # Meta-learner: Logistic Regression for final risk scoring
        self.meta_model_classifier = LogisticRegression(
            random_state=random_state,
            max_iter=1000,
            class_weight='balanced'
        )
        
        # Secondary meta-learner for continuous risk scores
        self.meta_model_regressor = RandomForestRegressor(
            n_estimators=100,
            max_depth=10,
            random_state=random_state,
            n_jobs=-1
        )
        
        self.scaler = StandardScaler()
        self.meta_features = None
        self.is_fitted = False
        self.risk_thresholds = {
            'low': 25,
            'medium': 50,
            'high': 75,
            'critical': 90
        }
        
    def extract_meta_features(self, df):
        """
        Extract features from all previous models
        
        Parameters:
        -----------
        df : pd.DataFrame
            Data with outputs from models 1-6
            
        Returns:
        --------
        pd.DataFrame : Meta-features for stacking
        """
        meta_features = []
        
        # Model 1 features (Isolation Forest)
        if 'model1_anomaly_score' in df.columns:
            meta_features.append('model1_anomaly_score')
        if 'model1_is_anomaly' in df.columns:
            meta_features.append('model1_is_anomaly')
        
        # Model 2 features (Community Detection)
        if 'model2_community_risk' in df.columns:
            meta_features.append('model2_community_risk')
        if 'model2_community_size' in df.columns:
            meta_features.append('model2_community_size')
        
        # Model 4 features (XGBoost Classification)
        if 'model4_confidence' in df.columns:
            meta_features.append('model4_confidence')
        
        # Add all model4 probability features
        prob_cols = [col for col in df.columns if col.startswith('model4_prob_')]
        meta_features.extend(prob_cols)
        
        # Model 5 features (HMM Temporal)
        if 'model5_hidden_state' in df.columns:
            meta_features.append('model5_hidden_state')
        if 'model5_state_probability' in df.columns:
            meta_features.append('model5_state_probability')
        if 'model5_transition_risk' in df.columns:
            meta_features.append('model5_transition_risk')
        
        # Model 6 features (Network Tracing)
        if 'model6_pagerank' in df.columns:
            meta_features.append('model6_pagerank')
        if 'model6_betweenness' in df.columns:
            meta_features.append('model6_betweenness')
        if 'model6_network_risk' in df.columns:
            meta_features.append('model6_network_risk')
        if 'model6_in_degree' in df.columns:
            meta_features.append('model6_in_degree')
        if 'model6_out_degree' in df.columns:
            meta_features.append('model6_out_degree')
        
        self.meta_features = meta_features
        
        return df[meta_features].fillna(0)
    
    def fit(self, df, test_size=0.2):
        """
        Train ensemble stacking model
        
        Parameters:
        -----------
        df : pd.DataFrame
            Data with outputs from models 1-6
        test_size : float
            Proportion of data for testing
            
        Returns:
        --------
        self
        """
        # Extract meta-features
        X_meta = self.extract_meta_features(df)
        
        # Check for target variable
        if 'is_laundering' not in df.columns:
            raise ValueError("Target variable 'is_laundering' not found")
        
        y = df['is_laundering']
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X_meta, y, test_size=test_size, random_state=42, stratify=y
        )
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Train classifier (for binary classification)
        self.meta_model_classifier.fit(X_train_scaled, y_train)
        
        # Train regressor (for continuous risk scores)
        self.meta_model_regressor.fit(X_train_scaled, y_train)
        
        # Evaluate models
        y_pred_proba = self.meta_model_classifier.predict_proba(X_test_scaled)[:, 1]
        y_pred = (y_pred_proba > 0.5).astype(int)
        
        auc_score = roc_auc_score(y_test, y_pred_proba)
        
        self.is_fitted = True
        
        print(f"✓ Model 7 ensemble training complete:")
        print(f"  Meta-features: {len(self.meta_features)}")
        print(f"  Train samples: {len(X_train)}, Test samples: {len(X_test)}")
        print(f"  ROC-AUC Score: {auc_score:.4f}")
        print(f"\nClassification Report:")
        print(classification_report(y_test, y_pred, target_names=['Legitimate', 'Laundering']))
        
        # Feature importance from regressor
        feature_importance = pd.DataFrame({
            'feature': self.meta_features,
            'importance': self.meta_model_regressor.feature_importances_
        }).sort_values('importance', ascending=False)
        
        print(f"\nTop 10 Meta-Feature Importance:")
        for idx, row in feature_importance.head(10).iterrows():
            print(f"  {row['feature']:<40} {row['importance']:.4f}")
        
        return self
    
    def predict(self, df):
        """
        Generate final risk scores and classifications
        
        Parameters:
        -----------
        df : pd.DataFrame
            Data with outputs from models 1-6
            
        Returns:
        --------
        pd.DataFrame : Original data with added columns:
            - model7_risk_score: Final risk score (0-100)
            - model7_risk_category: Risk category (low/medium/high/critical)
            - model7_laundering_probability: Probability of laundering
            - model7_is_flagged: Binary flag (1 = high risk, 0 = normal)
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")
        
        # Extract meta-features
        X_meta = self.extract_meta_features(df)
        X_scaled = self.scaler.transform(X_meta)
        
        # Get probability predictions from classifier
        laundering_proba = self.meta_model_classifier.predict_proba(X_scaled)[:, 1]
        
        # Get risk scores from regressor (0-1 range)
        risk_scores_normalized = self.meta_model_regressor.predict(X_scaled)
        risk_scores_normalized = np.clip(risk_scores_normalized, 0, 1)
        
        # Combine both predictions (weighted average)
        combined_risk = (laundering_proba * 0.6 + risk_scores_normalized * 0.4)
        
        # Scale to 0-100
        risk_scores_100 = combined_risk * 100
        
        # Assign risk categories
        risk_categories = pd.cut(
            risk_scores_100,
            bins=[0, self.risk_thresholds['low'], self.risk_thresholds['medium'],
                  self.risk_thresholds['high'], self.risk_thresholds['critical'], 100],
            labels=['low', 'low-medium', 'medium-high', 'high', 'critical'],
            include_lowest=True
        )
        
        # Add predictions to dataframe
        result_df = df.copy()
        result_df['model7_risk_score'] = risk_scores_100
        result_df['model7_risk_category'] = risk_categories
        result_df['model7_laundering_probability'] = laundering_proba
        result_df['model7_is_flagged'] = (risk_scores_100 >= self.risk_thresholds['high']).astype(int)
        
        # Calculate statistics
        category_counts = result_df['model7_risk_category'].value_counts()
        flagged_count = result_df['model7_is_flagged'].sum()
        
        print(f"✓ Model 7 final risk scoring complete:")
        print(f"  Mean risk score: {risk_scores_100.mean():.2f}")
        print(f"  Flagged wallets: {flagged_count} ({(flagged_count/len(result_df))*100:.2f}%)")
        print(f"\n  Risk category distribution:")
        for category, count in category_counts.items():
            pct = (count / len(result_df)) * 100
            print(f"    {category:<15} {count:>6} ({pct:>5.2f}%)")
        
        return result_df
    
    def get_top_risk_wallets(self, df, top_n=100):
        """
        Get top N highest risk wallets
        
        Parameters:
        -----------
        df : pd.DataFrame
            Data with model7 outputs
        top_n : int
            Number of top wallets to return
            
        Returns:
        --------
        pd.DataFrame : Top risk wallets with details
        """
        if 'model7_risk_score' not in df.columns:
            raise ValueError("Must run predict() first")
        
        top_wallets = df.nlargest(top_n, 'model7_risk_score')[[
            'wallet_address',
            'model7_risk_score',
            'model7_risk_category',
            'model7_laundering_probability',
            'model4_predicted_class',
            'model1_anomaly_score',
            'model6_network_risk'
        ]].copy()
        
        return top_wallets
    
    def get_risk_summary(self, df):
        """
        Generate summary statistics of risk assessment
        
        Parameters:
        -----------
        df : pd.DataFrame
            Data with model7 outputs
            
        Returns:
        --------
        dict : Summary statistics
        """
        if 'model7_risk_score' not in df.columns:
            raise ValueError("Must run predict() first")
        
        summary = {
            'total_wallets': len(df),
            'flagged_wallets': df['model7_is_flagged'].sum(),
            'flagged_percentage': (df['model7_is_flagged'].sum() / len(df)) * 100,
            'mean_risk_score': df['model7_risk_score'].mean(),
            'median_risk_score': df['model7_risk_score'].median(),
            'max_risk_score': df['model7_risk_score'].max(),
            'risk_category_counts': df['model7_risk_category'].value_counts().to_dict(),
            'critical_wallets': (df['model7_risk_score'] >= self.risk_thresholds['critical']).sum(),
            'high_risk_wallets': ((df['model7_risk_score'] >= self.risk_thresholds['high']) & 
                                 (df['model7_risk_score'] < self.risk_thresholds['critical'])).sum(),
        }
        
        return summary
    
    def save_model(self, filepath='models/model7_ensemble_stacking.pkl'):
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
            'meta_model_classifier': self.meta_model_classifier,
            'meta_model_regressor': self.meta_model_regressor,
            'scaler': self.scaler,
            'meta_features': self.meta_features,
            'risk_thresholds': self.risk_thresholds,
            'is_fitted': self.is_fitted
        }
        
        joblib.dump(model_data, filepath)
        print(f"✓ Model 7 saved to {filepath}")
    
    def load_model(self, filepath='models/model7_ensemble_stacking.pkl'):
        """
        Load trained model from disk
        
        Parameters:
        -----------
        filepath : str
            Path to load model from
        """
        model_data = joblib.load(filepath)
        
        self.meta_model_classifier = model_data['meta_model_classifier']
        self.meta_model_regressor = model_data['meta_model_regressor']
        self.scaler = model_data['scaler']
        self.meta_features = model_data['meta_features']
        self.risk_thresholds = model_data['risk_thresholds']
        self.is_fitted = model_data['is_fitted']
        
        print(f"✓ Model 7 loaded from {filepath}")
        
        return self


# Example usage
if __name__ == "__main__":
    # Load model6 output
    # df = pd.read_csv('data/model6_output.csv')
    
    # Initialize and train model
    model7 = EnsembleStackingModel()
    # model7.fit(df)
    
    # Generate final risk scores
    # result_df = model7.predict(df)
    
    # Get top risk wallets
    # top_risk = model7.get_top_risk_wallets(result_df, top_n=100)
    # print(top_risk)
    
    # Get summary
    # summary = model7.get_risk_summary(result_df)
    # print(summary)
    
    # Save results
    # result_df.to_csv('data/model7_output.csv', index=False)
    # model7.save_model()
    
    pass