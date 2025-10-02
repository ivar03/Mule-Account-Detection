import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import os

class FeatureSelectionModel:
    """
    MODEL 3: Random Forest Feature Importance
    Identifies most predictive features for laundering detection
    """
    
    def __init__(self, n_estimators=100, top_n_features=30, random_state=42):
        """
        Initialize Feature Selection model
        
        Parameters:
        -----------
        n_estimators : int
            Number of trees in random forest
        top_n_features : int
            Number of top features to select
        random_state : int
            Random seed for reproducibility
        """
        self.model = RandomForestClassifier(
            n_estimators=n_estimators,
            random_state=random_state,
            max_depth=10,
            min_samples_split=10,
            n_jobs=-1
        )
        self.scaler = StandardScaler()
        self.top_n_features = top_n_features
        self.feature_importance = None
        self.selected_features = None
        self.all_features = None
        self.is_fitted = False
        
    def prepare_features(self, df):
        """
        Prepare features for importance calculation
        
        Parameters:
        -----------
        df : pd.DataFrame
            Data with model1 and model2 outputs
            
        Returns:
        --------
        tuple : (X, y, feature_names)
        """
        # Identify feature columns (exclude identifiers and target)
        exclude_cols = ['wallet_address', 'is_laundering', 'laundering_type',
                       'timestamp', 'block_number', 'from_address', 'to_address',
                       'hash', 'date']
        
        feature_cols = [col for col in df.columns if col not in exclude_cols]
        
        # Check if target variable exists
        if 'is_laundering' not in df.columns:
            raise ValueError("Target variable 'is_laundering' not found in data")
        
        X = df[feature_cols].fillna(0)
        y = df['is_laundering']
        
        self.all_features = feature_cols
        
        return X, y, feature_cols
    
    def fit(self, df):
        """
        Calculate feature importance using Random Forest
        
        Parameters:
        -----------
        df : pd.DataFrame
            Data with model1 and model2 outputs
            
        Returns:
        --------
        self
        """
        X, y, feature_names = self.prepare_features(df)
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        # Train Random Forest
        self.model.fit(X_scaled, y)
        
        # Get feature importance
        importances = self.model.feature_importances_
        
        # Create feature importance dataframe
        self.feature_importance = pd.DataFrame({
            'feature': feature_names,
            'importance': importances
        }).sort_values('importance', ascending=False)
        
        # Select top features
        self.selected_features = self.feature_importance.head(self.top_n_features)['feature'].tolist()
        
        self.is_fitted = True
        
        print(f"✓ Model 3 feature selection complete:")
        print(f"  Total features: {len(feature_names)}")
        print(f"  Selected features: {len(self.selected_features)}")
        print(f"\n  Top 10 features:")
        for idx, row in self.feature_importance.head(10).iterrows():
            print(f"    {row['feature']:<40} {row['importance']:.4f}")
        
        return self
    
    def predict(self, df):
        """
        Apply feature selection to data
        
        Parameters:
        -----------
        df : pd.DataFrame
            Data with model1 and model2 outputs
            
        Returns:
        --------
        pd.DataFrame : Data with selected features + model outputs
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")
        
        # Keep identifier columns and selected features
        keep_cols = ['wallet_address', 'is_laundering', 'laundering_type', 
                    'model1_anomaly_score', 'model1_is_anomaly',
                    'model2_community_id', 'model2_community_risk', 
                    'model2_community_size'] + self.selected_features
        
        # Filter to existing columns
        keep_cols = [col for col in keep_cols if col in df.columns]
        
        result_df = df[keep_cols].copy()
        
        print(f"✓ Model 3 feature selection applied:")
        print(f"  Output features: {len(result_df.columns)}")
        print(f"  Selected predictive features: {len(self.selected_features)}")
        
        return result_df
    
    def get_feature_importance(self, top_n=None):
        """
        Get feature importance rankings
        
        Parameters:
        -----------
        top_n : int, optional
            Number of top features to return (default: all)
            
        Returns:
        --------
        pd.DataFrame : Feature importance rankings
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted first")
        
        if top_n is None:
            return self.feature_importance
        
        return self.feature_importance.head(top_n)
    
    def plot_feature_importance(self, top_n=20, figsize=(12, 8), save_path=None):
        """
        Plot feature importance
        
        Parameters:
        -----------
        top_n : int
            Number of top features to plot
        figsize : tuple
            Figure size
        save_path : str, optional
            Path to save plot
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted first")
        
        plt.figure(figsize=figsize)
        
        top_features = self.feature_importance.head(top_n)
        
        sns.barplot(
            data=top_features,
            x='importance',
            y='feature',
            palette='viridis'
        )
        
        plt.title(f'Top {top_n} Feature Importance - Random Forest', fontsize=16)
        plt.xlabel('Importance Score', fontsize=12)
        plt.ylabel('Feature', fontsize=12)
        plt.tight_layout()
        
        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✓ Plot saved to {save_path}")
        
        plt.show()
    
    def get_feature_correlations(self, df, threshold=0.8):
        """
        Identify highly correlated features
        
        Parameters:
        -----------
        df : pd.DataFrame
            Data with features
        threshold : float
            Correlation threshold
            
        Returns:
        --------
        pd.DataFrame : Highly correlated feature pairs
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted first")
        
        # Calculate correlation matrix for selected features
        feature_data = df[self.selected_features].fillna(0)
        corr_matrix = feature_data.corr().abs()
        
        # Get upper triangle
        upper = corr_matrix.where(
            np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
        )
        
        # Find highly correlated pairs
        high_corr = []
        for column in upper.columns:
            corr_features = upper[column][upper[column] > threshold]
            for feature in corr_features.index:
                high_corr.append({
                    'feature1': column,
                    'feature2': feature,
                    'correlation': corr_features[feature]
                })
        
        if high_corr:
            return pd.DataFrame(high_corr).sort_values('correlation', ascending=False)
        else:
            return pd.DataFrame()
    
    def save_model(self, filepath='models/model3_feature_selection.pkl'):
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
            'feature_importance': self.feature_importance,
            'selected_features': self.selected_features,
            'all_features': self.all_features,
            'top_n_features': self.top_n_features,
            'is_fitted': self.is_fitted
        }
        
        joblib.dump(model_data, filepath)
        print(f"✓ Model 3 saved to {filepath}")
    
    def load_model(self, filepath='models/model3_feature_selection.pkl'):
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
        self.feature_importance = model_data['feature_importance']
        self.selected_features = model_data['selected_features']
        self.all_features = model_data['all_features']
        self.top_n_features = model_data['top_n_features']
        self.is_fitted = model_data['is_fitted']
        
        print(f"✓ Model 3 loaded from {filepath}")
        
        return self


# Example usage
if __name__ == "__main__":
    # Load model2 output
    # df = pd.read_csv('data/model2_output.csv')
    
    # Initialize and train model
    model3 = FeatureSelectionModel(top_n_features=30)
    # model3.fit(df)
    
    # Apply feature selection
    # result_df = model3.predict(df)
    
    # Plot feature importance
    # model3.plot_feature_importance(top_n=20, save_path='outputs/feature_importance.png')
    
    # Save results
    # result_df.to_csv('data/model3_output.csv', index=False)
    # model3.save_model()
    
    pass