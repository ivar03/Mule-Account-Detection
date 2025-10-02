import pandas as pd
import numpy as np
import shap
import matplotlib.pyplot as plt
import joblib
import os

class ExplainabilityModel:
    """
    MODEL 8: SHAP (SHapley Additive exPlanations) for Model Explainability
    Explains predictions and provides feature importance for individual wallets
    """
    
    def __init__(self):
        """
        Initialize SHAP Explainability model
        """
        self.explainer = None
        self.shap_values = None
        self.base_model = None
        self.feature_names = None
        self.is_fitted = False
        
    def fit(self, model7, df, sample_size=1000):
        """
        Create SHAP explainer from ensemble model
        
        Parameters:
        -----------
        model7 : EnsembleStackingModel
            Trained ensemble model from Model 7
        df : pd.DataFrame
            Data with model7 outputs
        sample_size : int
            Number of samples to use for SHAP background (for efficiency)
            
        Returns:
        --------
        self
        """
        if not model7.is_fitted:
            raise ValueError("Model 7 must be fitted first")
        
        self.base_model = model7
        
        # Extract meta-features
        X_meta = model7.extract_meta_features(df)
        X_scaled = model7.scaler.transform(X_meta)
        
        self.feature_names = model7.meta_features
        
        # Sample background data for SHAP (for efficiency)
        if len(X_scaled) > sample_size:
            background_indices = np.random.choice(len(X_scaled), sample_size, replace=False)
            background_data = X_scaled[background_indices]
        else:
            background_data = X_scaled
        
        # Create SHAP explainer (using TreeExplainer for RandomForest regressor)
        print("Creating SHAP explainer (this may take a moment)...")
        self.explainer = shap.TreeExplainer(
            model7.meta_model_regressor,
            background_data
        )
        
        # Calculate SHAP values for all data
        print("Calculating SHAP values...")
        self.shap_values = self.explainer.shap_values(X_scaled)
        
        self.is_fitted = True
        
        print(f"✓ Model 8 explainability analysis complete:")
        print(f"  Background samples: {len(background_data)}")
        print(f"  SHAP values calculated for: {len(X_scaled)} wallets")
        print(f"  Features explained: {len(self.feature_names)}")
        
        return self
    
    def predict(self, df):
        """
        Add SHAP-based explanations to dataframe
        
        Parameters:
        -----------
        df : pd.DataFrame
            Data with model7 outputs
            
        Returns:
        --------
        pd.DataFrame : Original data with added columns:
            - model8_top_feature_1: Most important feature
            - model8_top_feature_1_impact: Impact value
            - model8_top_feature_2: Second most important feature
            - model8_top_feature_2_impact: Impact value
            - model8_top_feature_3: Third most important feature
            - model8_top_feature_3_impact: Impact value
            - model8_explanation_summary: Text summary of prediction
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")
        
        result_df = df.copy()
        
        # Get top 3 contributing features for each wallet
        top_features_1 = []
        top_impacts_1 = []
        top_features_2 = []
        top_impacts_2 = []
        top_features_3 = []
        top_impacts_3 = []
        explanations = []
        
        for i in range(len(self.shap_values)):
            # Get SHAP values for this wallet
            wallet_shap = self.shap_values[i]
            
            # Get top 3 features by absolute SHAP value
            top_indices = np.argsort(np.abs(wallet_shap))[-3:][::-1]
            
            top_features_1.append(self.feature_names[top_indices[0]])
            top_impacts_1.append(wallet_shap[top_indices[0]])
            
            top_features_2.append(self.feature_names[top_indices[1]] if len(top_indices) > 1 else '')
            top_impacts_2.append(wallet_shap[top_indices[1]] if len(top_indices) > 1 else 0)
            
            top_features_3.append(self.feature_names[top_indices[2]] if len(top_indices) > 2 else '')
            top_impacts_3.append(wallet_shap[top_indices[2]] if len(top_indices) > 2 else 0)
            
            # Create explanation summary
            explanation = self._create_explanation(
                top_indices, wallet_shap, 
                result_df.iloc[i]['model7_risk_score'],
                result_df.iloc[i]['model7_risk_category']
            )
            explanations.append(explanation)
        
        # Add to dataframe
        result_df['model8_top_feature_1'] = top_features_1
        result_df['model8_top_feature_1_impact'] = top_impacts_1
        result_df['model8_top_feature_2'] = top_features_2
        result_df['model8_top_feature_2_impact'] = top_impacts_2
        result_df['model8_top_feature_3'] = top_features_3
        result_df['model8_top_feature_3_impact'] = top_impacts_3
        result_df['model8_explanation_summary'] = explanations
        
        print(f"✓ Model 8 explanations generated:")
        print(f"  Wallets explained: {len(result_df)}")
        print(f"  Mean absolute top feature impact: {np.abs(top_impacts_1).mean():.4f}")
        
        return result_df
    
    def _create_explanation(self, top_indices, shap_values, risk_score, risk_category):
        """
        Create human-readable explanation for a wallet's risk score
        
        Parameters:
        -----------
        top_indices : np.array
            Indices of top contributing features
        shap_values : np.array
            SHAP values for the wallet
        risk_score : float
            Risk score from model 7
        risk_category : str
            Risk category from model 7
            
        Returns:
        --------
        str : Explanation text
        """
        explanation_parts = [f"Risk Score: {risk_score:.2f} ({risk_category})"]
        
        explanation_parts.append("Key factors:")
        
        for i, idx in enumerate(top_indices[:3]):
            feature_name = self.feature_names[idx]
            impact = shap_values[idx]
            
            # Clean feature name
            clean_name = feature_name.replace('model', 'M').replace('_', ' ')
            
            if impact > 0:
                explanation_parts.append(
                    f"  • {clean_name} increases risk (+{impact:.3f})"
                )
            else:
                explanation_parts.append(
                    f"  • {clean_name} decreases risk ({impact:.3f})"
                )
        
        return " | ".join(explanation_parts)
    
    def explain_wallet(self, df, wallet_address):
        """
        Get detailed explanation for a specific wallet
        
        Parameters:
        -----------
        df : pd.DataFrame
            Data with model8 outputs
        wallet_address : str
            Wallet address to explain
            
        Returns:
        --------
        dict : Detailed explanation
        """
        if 'model8_explanation_summary' not in df.columns:
            raise ValueError("Must run predict() first")
        
        wallet_data = df[df['wallet_address'] == wallet_address]
        
        if len(wallet_data) == 0:
            raise ValueError(f"Wallet {wallet_address} not found")
        
        wallet_idx = wallet_data.index[0]
        wallet_position = df.index.get_loc(wallet_idx)
        
        explanation = {
            'wallet_address': wallet_address,
            'risk_score': wallet_data.iloc[0]['model7_risk_score'],
            'risk_category': wallet_data.iloc[0]['model7_risk_category'],
            'laundering_probability': wallet_data.iloc[0]['model7_laundering_probability'],
            'predicted_pattern': wallet_data.iloc[0].get('model4_predicted_class', 'N/A'),
            'top_contributing_features': [],
            'feature_values': {}
        }
        
        # Get SHAP values for this wallet
        wallet_shap = self.shap_values[wallet_position]
        
        # Get all features sorted by absolute impact
        sorted_indices = np.argsort(np.abs(wallet_shap))[::-1]
        
        # Extract meta-features for this wallet
        X_meta = self.base_model.extract_meta_features(df)
        wallet_features = X_meta.iloc[wallet_position]
        
        for idx in sorted_indices[:10]:  # Top 10 features
            feature_name = self.feature_names[idx]
            impact = wallet_shap[idx]
            value = wallet_features[idx]
            
            explanation['top_contributing_features'].append({
                'feature': feature_name,
                'impact': float(impact),
                'value': float(value),
                'direction': 'increases risk' if impact > 0 else 'decreases risk'
            })
            
            explanation['feature_values'][feature_name] = float(value)
        
        explanation['summary'] = wallet_data.iloc[0]['model8_explanation_summary']
        
        return explanation
    
    def plot_feature_importance(self, top_n=20, figsize=(12, 8), save_path=None):
        """
        Plot global feature importance based on mean absolute SHAP values
        
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
        
        # Calculate mean absolute SHAP value for each feature
        mean_abs_shap = np.abs(self.shap_values).mean(axis=0)
        
        # Create dataframe
        importance_df = pd.DataFrame({
            'feature': self.feature_names,
            'importance': mean_abs_shap
        }).sort_values('importance', ascending=False).head(top_n)
        
        # Plot
        plt.figure(figsize=figsize)
        plt.barh(range(len(importance_df)), importance_df['importance'], color='steelblue')
        plt.yticks(range(len(importance_df)), importance_df['feature'])
        plt.xlabel('Mean |SHAP value| (average impact on model output)', fontsize=12)
        plt.title(f'Top {top_n} Feature Importance - SHAP Analysis', fontsize=16)
        plt.gca().invert_yaxis()
        plt.tight_layout()
        
        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✓ Plot saved to {save_path}")
        
        plt.show()
    
    def plot_shap_summary(self, max_display=20, save_path=None):
        """
        Create SHAP summary plot showing feature impact distribution
        
        Parameters:
        -----------
        max_display : int
            Maximum number of features to display
        save_path : str, optional
            Path to save plot
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted first")
        
        # Get feature data
        X_meta = self.base_model.extract_meta_features(
            pd.DataFrame()  # We'll use stored data
        )
        
        plt.figure(figsize=(12, 8))
        shap.summary_plot(
            self.shap_values,
            features=self.base_model.scaler.transform(
                self.base_model.extract_meta_features(
                    pd.read_csv('data/model7_output.csv') if os.path.exists('data/model7_output.csv') else pd.DataFrame()
                )
            ) if os.path.exists('data/model7_output.csv') else None,
            feature_names=self.feature_names,
            max_display=max_display,
            show=False
        )
        
        plt.tight_layout()
        
        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✓ Plot saved to {save_path}")
        
        plt.show()
    
    def plot_wallet_explanation(self, df, wallet_address, save_path=None):
        """
        Create waterfall plot for individual wallet explanation
        
        Parameters:
        -----------
        df : pd.DataFrame
            Data with model8 outputs
        wallet_address : str
            Wallet address to explain
        save_path : str, optional
            Path to save plot
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted first")
        
        wallet_data = df[df['wallet_address'] == wallet_address]
        
        if len(wallet_data) == 0:
            raise ValueError(f"Wallet {wallet_address} not found")
        
        wallet_idx = wallet_data.index[0]
        wallet_position = df.index.get_loc(wallet_idx)
        
        # Create waterfall plot
        shap.plots.waterfall(
            shap.Explanation(
                values=self.shap_values[wallet_position],
                base_values=self.explainer.expected_value,
                data=self.base_model.scaler.transform(
                    self.base_model.extract_meta_features(df)
                )[wallet_position],
                feature_names=self.feature_names
            ),
            show=False
        )
        
        plt.title(f'SHAP Explanation for Wallet: {wallet_address[:16]}...', fontsize=14)
        plt.tight_layout()
        
        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✓ Plot saved to {save_path}")
        
        plt.show()
    
    def get_feature_interactions(self, feature1, feature2):
        """
        Analyze interaction between two features
        
        Parameters:
        -----------
        feature1 : str
            First feature name
        feature2 : str
            Second feature name
            
        Returns:
        --------
        dict : Interaction statistics
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted first")
        
        if feature1 not in self.feature_names or feature2 not in self.feature_names:
            raise ValueError("Feature not found")
        
        idx1 = self.feature_names.index(feature1)
        idx2 = self.feature_names.index(feature2)
        
        # Calculate correlation between SHAP values
        shap_corr = np.corrcoef(
            self.shap_values[:, idx1],
            self.shap_values[:, idx2]
        )[0, 1]
        
        interaction = {
            'feature1': feature1,
            'feature2': feature2,
            'shap_correlation': float(shap_corr),
            'mean_impact_feature1': float(np.abs(self.shap_values[:, idx1]).mean()),
            'mean_impact_feature2': float(np.abs(self.shap_values[:, idx2]).mean()),
            'interaction_strength': 'high' if abs(shap_corr) > 0.7 else 'medium' if abs(shap_corr) > 0.4 else 'low'
        }
        
        return interaction
    
    def generate_investigation_report(self, df, wallet_address, save_path=None):
        """
        Generate comprehensive investigation report for a wallet
        
        Parameters:
        -----------
        df : pd.DataFrame
            Data with all model outputs
        wallet_address : str
            Wallet address to investigate
        save_path : str, optional
            Path to save report
            
        Returns:
        --------
        str : Investigation report text
        """
        explanation = self.explain_wallet(df, wallet_address)
        wallet_data = df[df['wallet_address'] == wallet_address].iloc[0]
        
        report = []
        report.append("=" * 80)
        report.append("MONEY LAUNDERING INVESTIGATION REPORT")
        report.append("=" * 80)
        report.append(f"\nWallet Address: {wallet_address}")
        report.append(f"Report Generated: {pd.Timestamp.now()}")
        report.append("\n" + "-" * 80)
        report.append("RISK ASSESSMENT")
        report.append("-" * 80)
        report.append(f"Risk Score: {explanation['risk_score']:.2f}/100")
        report.append(f"Risk Category: {explanation['risk_category'].upper()}")
        report.append(f"Laundering Probability: {explanation['laundering_probability']:.2%}")
        report.append(f"Predicted Pattern: {explanation['predicted_pattern']}")
        
        report.append("\n" + "-" * 80)
        report.append("MODEL OUTPUTS")
        report.append("-" * 80)
        report.append(f"Anomaly Score (Model 1): {wallet_data.get('model1_anomaly_score', 'N/A'):.4f}")
        report.append(f"Community Risk (Model 2): {wallet_data.get('model2_community_risk', 'N/A'):.4f}")
        report.append(f"Classification Confidence (Model 4): {wallet_data.get('model4_confidence', 'N/A'):.4f}")
        report.append(f"Temporal Transition Risk (Model 5): {wallet_data.get('model5_transition_risk', 'N/A'):.4f}")
        report.append(f"Network Risk (Model 6): {wallet_data.get('model6_network_risk', 'N/A'):.4f}")
        
        report.append("\n" + "-" * 80)
        report.append("TOP CONTRIBUTING FACTORS (SHAP Analysis)")
        report.append("-" * 80)
        
        for i, factor in enumerate(explanation['top_contributing_features'][:5], 1):
            report.append(f"{i}. {factor['feature']}")
            report.append(f"   Value: {factor['value']:.4f}")
            report.append(f"   Impact: {factor['impact']:.4f} ({factor['direction']})")
        
        report.append("\n" + "-" * 80)
        report.append("EXPLANATION SUMMARY")
        report.append("-" * 80)
        report.append(explanation['summary'])
        
        report.append("\n" + "-" * 80)
        report.append("RECOMMENDED ACTIONS")
        report.append("-" * 80)
        
        if explanation['risk_score'] >= 90:
            report.append("• CRITICAL: Immediate investigation required")
            report.append("• Freeze wallet transactions pending review")
            report.append("• Trace fund flows using Model 6 network analysis")
            report.append("• Report to relevant authorities")
        elif explanation['risk_score'] >= 75:
            report.append("• HIGH: Priority investigation recommended")
            report.append("• Enhanced monitoring of transactions")
            report.append("• Request additional documentation")
        elif explanation['risk_score'] >= 50:
            report.append("• MEDIUM: Standard review procedures")
            report.append("• Monitor for pattern changes")
        else:
            report.append("• LOW: Continue routine monitoring")
        
        report.append("\n" + "=" * 80)
        report.append("END OF REPORT")
        report.append("=" * 80)
        
        report_text = "\n".join(report)
        
        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            with open(save_path, 'w') as f:
                f.write(report_text)
            print(f"✓ Report saved to {save_path}")
        
        return report_text
    
    def save_model(self, filepath='models/model8_shap_explainability.pkl'):
        """
        Save explainability model to disk
        
        Parameters:
        -----------
        filepath : str
            Path to save model
        """
        if not self.is_fitted:
            raise ValueError("Cannot save unfitted model")
        
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        model_data = {
            'explainer': self.explainer,
            'shap_values': self.shap_values,
            'base_model': self.base_model,
            'feature_names': self.feature_names,
            'is_fitted': self.is_fitted
        }
        
        joblib.dump(model_data, filepath)
        print(f"✓ Model 8 saved to {filepath}")
    
    def load_model(self, filepath='models/model8_shap_explainability.pkl'):
        """
        Load explainability model from disk
        
        Parameters:
        -----------
        filepath : str
            Path to load model from
        """
        model_data = joblib.load(filepath)
        
        self.explainer = model_data['explainer']
        self.shap_values = model_data['shap_values']
        self.base_model = model_data['base_model']
        self.feature_names = model_data['feature_names']
        self.is_fitted = model_data['is_fitted']
        
        print(f"✓ Model 8 loaded from {filepath}")
        
        return self


# Example usage
if __name__ == "__main__":
    # Load model7 and its output
    # from model7_ensemble_stacking import EnsembleStackingModel
    # model7 = EnsembleStackingModel()
    # model7.load_model()
    # df = pd.read_csv('data/model7_output.csv')
    
    # Initialize and fit explainability model
    model8 = ExplainabilityModel()
    # model8.fit(model7, df, sample_size=1000)
    
    # Add explanations to dataframe
    # result_df = model8.predict(df)
    
    # Plot feature importance
    # model8.plot_feature_importance(top_n=20, save_path='outputs/shap_importance.png')
    
    # Explain specific wallet
    # suspicious_wallet = result_df.nlargest(1, 'model7_risk_score').iloc[0]['wallet_address']
    # explanation = model8.explain_wallet(result_df, suspicious_wallet)
    # print(explanation)
    
    # Generate investigation report
    # report = model8.generate_investigation_report(
    #     result_df, 
    #     suspicious_wallet,
    #     save_path=f'outputs/investigation_report_{suspicious_wallet[:8]}.txt'
    # )
    # print(report)
    
    # Save results
    # result_df.to_csv('data/model8_output_final.csv', index=False)
    # model8.save_model()
    
    pass