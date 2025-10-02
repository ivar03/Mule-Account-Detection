import pandas as pd
import numpy as np
import networkx as nx
from networkx.algorithms import community
import joblib
import os

class CommunityDetectionModel:
    """
    MODEL 2: Louvain Community Detection
    Identifies connected wallet clusters and suspicious networks
    """
    
    def __init__(self, resolution=1.0, random_state=42):
        """
        Initialize Community Detection model
        
        Parameters:
        -----------
        resolution : float
            Resolution parameter for Louvain algorithm (higher = more communities)
        random_state : int
            Random seed for reproducibility
        """
        self.resolution = resolution
        self.random_state = random_state
        self.graph = None
        self.communities = None
        self.community_stats = None
        self.is_fitted = False
        
    def build_graph(self, df, transaction_col='from_address', to_col='to_address', 
                    amount_col='amount', threshold=0):
        """
        Build transaction graph from data
        
        Parameters:
        -----------
        df : pd.DataFrame
            Transaction data with wallet addresses
        transaction_col : str
            Column name for source wallet
        to_col : str
            Column name for destination wallet
        amount_col : str
            Column name for transaction amount
        threshold : float
            Minimum transaction amount to include
            
        Returns:
        --------
        nx.Graph : Transaction graph
        """
        # Filter transactions above threshold
        df_filtered = df[df[amount_col] >= threshold].copy()
        
        # Create directed graph
        G = nx.DiGraph()
        
        # Add edges with transaction attributes
        for _, row in df_filtered.iterrows():
            from_addr = row[transaction_col]
            to_addr = row[to_col]
            amount = row[amount_col]
            
            if G.has_edge(from_addr, to_addr):
                # Aggregate multiple transactions
                G[from_addr][to_addr]['weight'] += amount
                G[from_addr][to_addr]['count'] += 1
            else:
                G.add_edge(from_addr, to_addr, weight=amount, count=1)
        
        # Convert to undirected for community detection
        self.graph = G.to_undirected()
        
        print(f"✓ Graph built: {self.graph.number_of_nodes()} nodes, "
              f"{self.graph.number_of_edges()} edges")
        
        return self.graph
    
    def detect_communities(self):
        """
        Detect communities using Louvain algorithm
        
        Returns:
        --------
        dict : Node to community mapping
        """
        if self.graph is None:
            raise ValueError("Graph must be built before community detection")
        
        # Apply Louvain community detection
        self.communities = community.louvain_communities(
            self.graph, 
            resolution=self.resolution,
            seed=self.random_state
        )
        
        # Create node to community mapping
        node_to_community = {}
        for comm_id, comm_nodes in enumerate(self.communities):
            for node in comm_nodes:
                node_to_community[node] = comm_id
        
        self.is_fitted = True
        
        print(f"✓ Communities detected: {len(self.communities)}")
        print(f"  Modularity: {community.modularity(self.graph, self.communities):.4f}")
        
        return node_to_community
    
    def calculate_community_stats(self, df):
        """
        Calculate statistics for each community
        
        Parameters:
        -----------
        df : pd.DataFrame
            Data with model1 outputs (anomaly scores)
            
        Returns:
        --------
        pd.DataFrame : Community statistics
        """
        node_to_comm = self.detect_communities()
        
        # Map wallets to communities
        df['model2_community_id'] = df['wallet_address'].map(node_to_comm)
        
        # Calculate community-level statistics
        community_stats = []
        
        for comm_id in range(len(self.communities)):
            comm_wallets = df[df['model2_community_id'] == comm_id]
            
            if len(comm_wallets) == 0:
                continue
            
            stats = {
                'community_id': comm_id,
                'size': len(comm_wallets),
                'avg_anomaly_score': comm_wallets['model1_anomaly_score'].mean(),
                'anomaly_rate': comm_wallets['model1_is_anomaly'].mean(),
                'total_transaction_volume': comm_wallets.get('total_amount_sent', 0).sum(),
                'avg_transaction_count': comm_wallets.get('transaction_count', 0).mean(),
            }
            
            # Calculate network centrality for community
            comm_nodes = list(self.communities[comm_id])
            subgraph = self.graph.subgraph(comm_nodes)
            
            if subgraph.number_of_edges() > 0:
                stats['density'] = nx.density(subgraph)
                stats['avg_clustering'] = nx.average_clustering(subgraph)
            else:
                stats['density'] = 0
                stats['avg_clustering'] = 0
            
            community_stats.append(stats)
        
        self.community_stats = pd.DataFrame(community_stats)
        
        return self.community_stats
    
    def predict(self, df):
        """
        Assign communities and calculate risk scores
        
        Parameters:
        -----------
        df : pd.DataFrame
            Data with model1 outputs
            
        Returns:
        --------
        pd.DataFrame : Original data with added columns:
            - model2_community_id: Community assignment
            - model2_community_risk: Community risk score (0-1)
            - model2_community_size: Number of wallets in community
        """
        if not self.is_fitted:
            # Build graph and detect communities
            self.build_graph(df)
            self.calculate_community_stats(df)
        
        # Map communities to dataframe
        node_to_comm = {}
        for comm_id, comm_nodes in enumerate(self.communities):
            for node in comm_nodes:
                node_to_comm[node] = comm_id
        
        result_df = df.copy()
        result_df['model2_community_id'] = result_df['wallet_address'].map(node_to_comm)
        
        # Add community statistics
        comm_risk = self.community_stats.set_index('community_id')['avg_anomaly_score'].to_dict()
        comm_size = self.community_stats.set_index('community_id')['size'].to_dict()
        
        result_df['model2_community_risk'] = result_df['model2_community_id'].map(comm_risk)
        result_df['model2_community_size'] = result_df['model2_community_id'].map(comm_size)
        
        # Fill NaN for wallets not in any community
        result_df['model2_community_risk'].fillna(0, inplace=True)
        result_df['model2_community_size'].fillna(1, inplace=True)
        
        suspicious_communities = (self.community_stats['avg_anomaly_score'] > 0.7).sum()
        
        print(f"✓ Model 2 detection complete:")
        print(f"  Suspicious communities: {suspicious_communities}")
        print(f"  Mean community risk: {result_df['model2_community_risk'].mean():.4f}")
        
        return result_df
    
    def get_community_subgraph(self, community_id):
        """
        Extract subgraph for a specific community
        
        Parameters:
        -----------
        community_id : int
            Community ID
            
        Returns:
        --------
        nx.Graph : Subgraph of community
        """
        if community_id >= len(self.communities):
            raise ValueError(f"Community {community_id} does not exist")
        
        comm_nodes = list(self.communities[community_id])
        return self.graph.subgraph(comm_nodes)
    
    def save_model(self, filepath='models/model2_community_detection.pkl'):
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
            'graph': self.graph,
            'communities': self.communities,
            'community_stats': self.community_stats,
            'resolution': self.resolution,
            'is_fitted': self.is_fitted
        }
        
        joblib.dump(model_data, filepath)
        print(f"✓ Model 2 saved to {filepath}")
    
    def load_model(self, filepath='models/model2_community_detection.pkl'):
        """
        Load trained model from disk
        
        Parameters:
        -----------
        filepath : str
            Path to load model from
        """
        model_data = joblib.load(filepath)
        
        self.graph = model_data['graph']
        self.communities = model_data['communities']
        self.community_stats = model_data['community_stats']
        self.resolution = model_data['resolution']
        self.is_fitted = model_data['is_fitted']
        
        print(f"✓ Model 2 loaded from {filepath}")
        
        return self


# Example usage
if __name__ == "__main__":
    # Load model1 output
    # df = pd.read_csv('data/model1_output.csv')
    
    # Initialize and train model
    model2 = CommunityDetectionModel(resolution=1.0)
    # result_df = model2.predict(df)
    
    # Save results
    # result_df.to_csv('data/model2_output.csv', index=False)
    # model2.save_model()
    
    pass