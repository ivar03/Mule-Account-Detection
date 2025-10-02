import pandas as pd
import numpy as np
from hmmlearn import hmm
from sklearn.preprocessing import StandardScaler
import joblib
import os

class TemporalSequenceModel:
    """
    MODEL 5: Hidden Markov Model for Temporal Analysis
    Detects wallet state transitions from legitimate → suspicious behavior
    """
    
    def __init__(self, n_states=3, n_iter=100, random_state=42):
        """
        Initialize Hidden Markov Model
        
        Parameters:
        -----------
        n_states : int
            Number of hidden states (e.g., 3: normal, suspicious, highly_suspicious)
        n_iter : int
            Number of training iterations
        random_state : int
            Random seed for reproducibility
        """
        self.n_states = n_states
        self.model = hmm.GaussianHMM(
            n_components=n_states,
            covariance_type="diag",
            n_iter=n_iter,
            random_state=random_state
        )
        self.scaler = StandardScaler()
        self.feature_columns = None
        self.state_labels = ['normal', 'suspicious', 'highly_suspicious'][:n_states]
        self.is_fitted = False
        
    def prepare_sequences(self, df, sequence_col='wallet_address', 
                         time_col='timestamp'):
        """
        Prepare time-ordered sequences for each wallet
        
        Parameters:
        -----------
        df : pd.DataFrame
            Data with model4 outputs
        sequence_col : str
            Column to group sequences by (wallet address)
        time_col : str
            Column to sort sequences by (timestamp)
            
        Returns:
        --------
        list : List of (sequence_features, sequence_length) tuples
        """
        # Select temporal features
        temporal_features = [
            'model1_anomaly_score',
            'model2_community_risk',
            'model4_confidence',
        ]
        
        # Add all probability columns
        prob_cols = [col for col in df.columns if col.startswith('model4_prob_')]
        temporal_features.extend(prob_cols)
        
        # Add numeric transaction features
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        additional_features = [col for col in numeric_cols 
                             if col not in temporal_features 
                             and col not in ['is_laundering', 'block_number']]
        
        temporal_features.extend(additional_features[:5])  # Limit to top 5 additional
        
        self.feature_columns = temporal_features
        
        # Sort by time and group by wallet
        df_sorted = df.sort_values([sequence_col, time_col])
        
        sequences = []
        sequence_lengths = []
        wallet_addresses = []
        
        for wallet, group in df_sorted.groupby(sequence_col):
            if len(group) < 2:  # Skip wallets with only 1 transaction
                continue
            
            seq_features = group[temporal_features].fillna(0).values
            
            sequences.append(seq_features)
            sequence_lengths.append(len(seq_features))
            wallet_addresses.append(wallet)
        
        return sequences, sequence_lengths, wallet_addresses
    
    def fit(self, df):
        """
        Train Hidden Markov Model on wallet sequences
        
        Parameters:
        -----------
        df : pd.DataFrame
            Data with model4 outputs
            
        Returns:
        --------
        self
        """
        sequences, lengths, wallets = self.prepare_sequences(df)
        
        # Concatenate all sequences for training
        X = np.vstack(sequences)
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        # Train HMM
        self.model.fit(X_scaled, lengths=lengths)
        
        self.is_fitted = True
        
        print(f"✓ Model 5 training complete:")
        print(f"  States: {self.n_states}")
        print(f"  Sequences: {len(sequences)}")
        print(f"  Total observations: {len(X)}")
        print(f"  Features per observation: {len(self.feature_columns)}")
        print(f"\n  State transition matrix:")
        print(self.model.transmat_)
        
        return self
    
    def predict(self, df):
        """
        Predict hidden states and detect suspicious transitions
        
        Parameters:
        -----------
        df : pd.DataFrame
            Data with model4 outputs
            
        Returns:
        --------
        pd.DataFrame : Original data with added columns:
            - model5_hidden_state: Predicted hidden state (0, 1, 2, ...)
            - model5_state_label: State label (normal, suspicious, highly_suspicious)
            - model5_state_probability: Probability of current state
            - model5_transition_risk: Risk score based on state transition
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")
        
        sequences, lengths, wallets = self.prepare_sequences(df)
        
        # Prepare results storage
        wallet_states = {}
        wallet_state_probs = {}
        wallet_transitions = {}
        
        # Predict states for each sequence
        for seq, length, wallet in zip(sequences, lengths, wallets):
            X_scaled = self.scaler.transform(seq)
            
            # Predict most likely state sequence
            states = self.model.predict(X_scaled)
            
            # Calculate state probabilities
            state_probs = self.model.predict_proba(X_scaled).max(axis=1)
            
            # Calculate transition risk (increases when moving to higher states)
            transition_risk = np.zeros(len(states))
            for i in range(1, len(states)):
                if states[i] > states[i-1]:  # Transition to more suspicious state
                    transition_risk[i] = (states[i] - states[i-1]) / (self.n_states - 1)
            
            wallet_states[wallet] = states
            wallet_state_probs[wallet] = state_probs
            wallet_transitions[wallet] = transition_risk
        
        # Add predictions to dataframe
        result_df = df.copy()
        
        # Map states back to original dataframe
        state_list = []
        state_prob_list = []
        transition_risk_list = []
        
        for idx, row in result_df.iterrows():
            wallet = row['wallet_address']
            
            if wallet in wallet_states:
                # Find position in sequence (by timestamp)
                wallet_df = result_df[result_df['wallet_address'] == wallet].sort_values('timestamp')
                position = wallet_df.index.get_loc(idx)
                
                state_list.append(wallet_states[wallet][position])
                state_prob_list.append(wallet_state_probs[wallet][position])
                transition_risk_list.append(wallet_transitions[wallet][position])
            else:
                state_list.append(0)  # Default to normal state
                state_prob_list.append(0.5)
                transition_risk_list.append(0.0)
        
        result_df['model5_hidden_state'] = state_list
        result_df['model5_state_label'] = [self.state_labels[int(s)] if int(s) < len(self.state_labels) 
                                           else 'unknown' for s in state_list]
        result_df['model5_state_probability'] = state_prob_list
        result_df['model5_transition_risk'] = transition_risk_list
        
        # Calculate statistics
        suspicious_transitions = (result_df['model5_transition_risk'] > 0).sum()
        high_risk_states = (result_df['model5_hidden_state'] >= self.n_states - 1).sum()
        
        print(f"✓ Model 5 state prediction complete:")
        print(f"  Suspicious transitions detected: {suspicious_transitions}")
        print(f"  High-risk states: {high_risk_states}")
        print(f"  Mean transition risk: {result_df['model5_transition_risk'].mean():.4f}")
        
        return result_df
    
    def detect_anomalous_transitions(self, df, threshold=0.5):
        """
        Identify wallets with anomalous state transitions
        
        Parameters:
        -----------
        df : pd.DataFrame
            Data with model5 outputs
        threshold : float
            Transition risk threshold for flagging
            
        Returns:
        --------
        pd.DataFrame : Wallets with suspicious transitions
        """
        suspicious = df[df['model5_transition_risk'] > threshold].copy()
        
        if len(suspicious) > 0:
            suspicious_summary = suspicious.groupby('wallet_address').agg({
                'model5_transition_risk': 'max',
                'model5_hidden_state': 'max',
                'model4_predicted_class': lambda x: x.mode()[0] if len(x) > 0 else 'unknown'
            }).reset_index()
            
            suspicious_summary.columns = ['wallet_address', 'max_transition_risk', 
                                         'max_state', 'dominant_pattern']
            
            return suspicious_summary.sort_values('max_transition_risk', ascending=False)
        
        return pd.DataFrame()
    
    def save_model(self, filepath='models/model5_hmm.pkl'):
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
            'state_labels': self.state_labels,
            'n_states': self.n_states,
            'is_fitted': self.is_fitted
        }
        
        joblib.dump(model_data, filepath)
        print(f"✓ Model 5 saved to {filepath}")
    
    def load_model(self, filepath='models/model5_hmm.pkl'):
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
        self.state_labels = model_data['state_labels']
        self.n_states = model_data['n_states']
        self.is_fitted = model_data['is_fitted']
        
        print(f"✓ Model 5 loaded from {filepath}")
        
        return self


# Example usage
if __name__ == "__main__":
    # Load model4 output
    # df = pd.read_csv('data/model4_output.csv')
    
    # Initialize and train model
    model5 = TemporalSequenceModel(n_states=3)
    # model5.fit(df)
    
    # Predict states and transitions
    # result_df = model5.predict(df)
    
    # Detect anomalous transitions
    # suspicious = model5.detect_anomalous_transitions(result_df, threshold=0.5)
    # print(suspicious)
    
    # Save results
    # result_df.to_csv('data/model5_output.csv', index=False)
    # model5.save_model()
    
    pass