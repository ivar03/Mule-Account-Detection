import pandas as pd
import numpy as np
import networkx as nx
import joblib
import os

class NetworkTracingModel:
    """
    MODEL 6: Graph Traversal + PageRank for Fund Flow Tracing
    Traces fund flows and identifies key players in laundering networks
    """
    
    def __init__(self, alpha=0.85, max_depth=5):
        """
        Initialize Network Tracing model
        
        Parameters:
        -----------
        alpha : float
            Damping parameter for PageRank (0.85 = 85% probability of following links)
        max_depth : int
            Maximum depth for backward/forward tracing
        """
        self.alpha = alpha
        self.max_depth = max_depth
        self.graph = None
        self.pagerank_scores = None
        self.betweenness_scores = None
        self.is_fitted = False
        
    def build_transaction_graph(self, df, from_col='from_address', 
                                to_col='to_address', amount_col='amount'):
        """
        Build directed transaction graph
        
        Parameters:
        -----------
        df : pd.DataFrame
            Transaction data with addresses and amounts
        from_col : str
            Source address column
        to_col : str
            Destination address column
        amount_col : str
            Transaction amount column
            
        Returns:
        --------
        nx.DiGraph : Transaction graph
        """
        G = nx.DiGraph()
        
        # Group transactions between same wallet pairs
        transactions = df.groupby([from_col, to_col]).agg({
            amount_col: ['sum', 'count', 'mean']
        }).reset_index()
        
        transactions.columns = [from_col, to_col, 'total_amount', 'tx_count', 'avg_amount']
        
        # Add edges with transaction attributes
        for _, row in transactions.iterrows():
            G.add_edge(
                row[from_col],
                row[to_col],
                weight=row['total_amount'],
                count=row['tx_count'],
                avg_amount=row['avg_amount']
            )
        
        # Add node attributes from wallet features
        wallet_features = df.groupby('wallet_address').agg({
            'model1_anomaly_score': 'mean',
            'model2_community_risk': 'mean',
            'model4_confidence': 'mean',
            'model5_transition_risk': 'mean'
        }).to_dict('index')
        
        for node in G.nodes():
            if node in wallet_features:
                for attr, value in wallet_features[node].items():
                    G.nodes[node][attr] = value
        
        self.graph = G
        
        print(f"✓ Transaction graph built:")
        print(f"  Nodes (wallets): {G.number_of_nodes()}")
        print(f"  Edges (transactions): {G.number_of_edges()}")
        print(f"  Density: {nx.density(G):.6f}")
        
        return G
    
    def fit(self, df):
        """
        Calculate graph centrality metrics
        
        Parameters:
        -----------
        df : pd.DataFrame
            Data with model5 outputs
            
        Returns:
        --------
        self
        """
        # Build graph
        self.build_transaction_graph(df)
        
        # Calculate PageRank (importance/centrality)
        self.pagerank_scores = nx.pagerank(
            self.graph, 
            alpha=self.alpha,
            weight='weight'
        )
        
        # Calculate betweenness centrality (bridge/intermediary role)
        self.betweenness_scores = nx.betweenness_centrality(
            self.graph,
            weight='weight'
        )
        
        self.is_fitted = True
        
        # Show top central nodes
        top_pagerank = sorted(self.pagerank_scores.items(), 
                            key=lambda x: x[1], reverse=True)[:5]
        top_betweenness = sorted(self.betweenness_scores.items(), 
                                key=lambda x: x[1], reverse=True)[:5]
        
        print(f"✓ Model 6 centrality analysis complete:")
        print(f"\n  Top 5 by PageRank (importance):")
        for node, score in top_pagerank:
            print(f"    {node[:16]}... : {score:.6f}")
        
        print(f"\n  Top 5 by Betweenness (intermediary role):")
        for node, score in top_betweenness:
            print(f"    {node[:16]}... : {score:.6f}")
        
        return self
    
    def predict(self, df):
        """
        Add network centrality scores to wallets
        
        Parameters:
        -----------
        df : pd.DataFrame
            Data with model5 outputs
            
        Returns:
        --------
        pd.DataFrame : Original data with added columns:
            - model6_pagerank: PageRank score (importance)
            - model6_betweenness: Betweenness centrality (intermediary)
            - model6_in_degree: Number of incoming transactions
            - model6_out_degree: Number of outgoing transactions
            - model6_network_risk: Combined network risk score
        """
        if not self.is_fitted:
            self.fit(df)
        
        result_df = df.copy()
        
        # Map centrality scores
        result_df['model6_pagerank'] = result_df['wallet_address'].map(self.pagerank_scores)
        result_df['model6_betweenness'] = result_df['wallet_address'].map(self.betweenness_scores)
        
        # Calculate degree metrics
        in_degree = dict(self.graph.in_degree())
        out_degree = dict(self.graph.out_degree())
        
        result_df['model6_in_degree'] = result_df['wallet_address'].map(in_degree)
        result_df['model6_out_degree'] = result_df['wallet_address'].map(out_degree)
        
        # Fill NaN values for wallets not in graph
        result_df['model6_pagerank'].fillna(0, inplace=True)
        result_df['model6_betweenness'].fillna(0, inplace=True)
        result_df['model6_in_degree'].fillna(0, inplace=True)
        result_df['model6_out_degree'].fillna(0, inplace=True)
        
        # Calculate network risk score (combination of centrality and previous model scores)
        result_df['model6_network_risk'] = (
            result_df['model6_pagerank'] * 0.3 +
            result_df['model6_betweenness'] * 0.3 +
            result_df['model1_anomaly_score'] * 0.2 +
            result_df['model2_community_risk'] * 0.2
        )
        
        # Normalize to 0-1
        max_risk = result_df['model6_network_risk'].max()
        if max_risk > 0:
            result_df['model6_network_risk'] = result_df['model6_network_risk'] / max_risk
        
        high_risk_wallets = (result_df['model6_network_risk'] > 0.7).sum()
        
        print(f"✓ Model 6 network analysis complete:")
        print(f"  High network risk wallets: {high_risk_wallets}")
        print(f"  Mean PageRank: {result_df['model6_pagerank'].mean():.6f}")
        print(f"  Mean network risk: {result_df['model6_network_risk'].mean():.4f}")
        
        return result_df
    
    def trace_backwards(self, seed_wallet, depth=None):
        """
        Trace fund sources backwards from a suspicious wallet
        
        Parameters:
        -----------
        seed_wallet : str
            Starting wallet address
        depth : int, optional
            Maximum trace depth (default: self.max_depth)
            
        Returns:
        --------
        dict : Traced wallets with distances and paths
        """
        if self.graph is None:
            raise ValueError("Graph must be built before tracing")
        
        if seed_wallet not in self.graph:
            return {}
        
        depth = depth or self.max_depth
        
        # Find all predecessors within depth
        traced = {}
        
        for node in self.graph.nodes():
            if node == seed_wallet:
                continue
            
            try:
                # Check if path exists
                if nx.has_path(self.graph, node, seed_wallet):
                    path = nx.shortest_path(self.graph, node, seed_wallet, weight='weight')
                    
                    if len(path) - 1 <= depth:  # Path length within depth
                        traced[node] = {
                            'distance': len(path) - 1,
                            'path': path,
                            'total_flow': sum(
                                self.graph[path[i]][path[i+1]]['weight'] 
                                for i in range(len(path)-1)
                            )
                        }
            except nx.NetworkXNoPath:
                continue
        
        return traced
    
    def trace_forwards(self, seed_wallet, depth=None):
        """
        Trace fund destinations forward from a suspicious wallet
        
        Parameters:
        -----------
        seed_wallet : str
            Starting wallet address
        depth : int, optional
            Maximum trace depth (default: self.max_depth)
            
        Returns:
        --------
        dict : Traced wallets with distances and paths
        """
        if self.graph is None:
            raise ValueError("Graph must be built before tracing")
        
        if seed_wallet not in self.graph:
            return {}
        
        depth = depth or self.max_depth
        
        # Find all successors within depth
        traced = {}
        
        for node in self.graph.nodes():
            if node == seed_wallet:
                continue
            
            try:
                # Check if path exists
                if nx.has_path(self.graph, seed_wallet, node):
                    path = nx.shortest_path(self.graph, seed_wallet, node, weight='weight')
                    
                    if len(path) - 1 <= depth:  # Path length within depth
                        traced[node] = {
                            'distance': len(path) - 1,
                            'path': path,
                            'total_flow': sum(
                                self.graph[path[i]][path[i+1]]['weight'] 
                                for i in range(len(path)-1)
                            )
                        }
            except nx.NetworkXNoPath:
                continue
        
        return traced
    
    def get_connected_subgraph(self, seed_wallets, depth=2):
        """
        Extract subgraph of connected wallets around seed wallets
        
        Parameters:
        -----------
        seed_wallets : list
            List of wallet addresses to trace from
        depth : int
            Neighborhood depth
            
        Returns:
        --------
        nx.DiGraph : Subgraph of connected wallets
        """
        if self.graph is None:
            raise ValueError("Graph must be built before extraction")
        
        # Collect all neighbors within depth
        connected_nodes = set(seed_wallets)
        
        for seed in seed_wallets:
            if seed not in self.graph:
                continue
            
            # Get ego graph (neighborhood)
            ego = nx.ego_graph(self.graph, seed, radius=depth)
            connected_nodes.update(ego.nodes())
        
        # Extract subgraph
        subgraph = self.graph.subgraph(connected_nodes).copy()
        
        return subgraph
    
    def identify_money_flow_paths(self, source_wallets, sink_wallets, k=5):
        """
        Find top k paths between source and sink wallets
        
        Parameters:
        -----------
        source_wallets : list
            List of source wallet addresses
        sink_wallets : list
            List of destination wallet addresses
        k : int
            Number of top paths to return
            
        Returns:
        --------
        list : Top k paths with flow amounts
        """
        if self.graph is None:
            raise ValueError("Graph must be built before path analysis")
        
        all_paths = []
        
        for source in source_wallets:
            if source not in self.graph:
                continue
            
            for sink in sink_wallets:
                if sink not in self.graph or source == sink:
                    continue
                
                try:
                    # Find all simple paths (no cycles)
                    paths = nx.all_simple_paths(
                        self.graph, 
                        source, 
                        sink, 
                        cutoff=self.max_depth
                    )
                    
                    for path in paths:
                        # Calculate total flow along path
                        flow = sum(
                            self.graph[path[i]][path[i+1]]['weight'] 
                            for i in range(len(path)-1)
                        )
                        
                        all_paths.append({
                            'source': source,
                            'sink': sink,
                            'path': path,
                            'length': len(path) - 1,
                            'flow': flow
                        })
                        
                except nx.NetworkXNoPath:
                    continue
        
        # Sort by flow amount and return top k
        all_paths.sort(key=lambda x: x['flow'], reverse=True)
        
        return all_paths[:k]
    
    def calculate_personalized_pagerank(self, seed_wallets, alpha=0.85):
        """
        Calculate Personalized PageRank from seed wallets
        (identifies wallets most connected to suspicious seeds)
        
        Parameters:
        -----------
        seed_wallets : list
            List of suspicious wallet addresses
        alpha : float
            Damping parameter
            
        Returns:
        --------
        dict : Wallet to personalized PageRank score mapping
        """
        if self.graph is None:
            raise ValueError("Graph must be built before PageRank")
        
        # Create personalization dict (uniform over seed wallets)
        personalization = {node: 0 for node in self.graph.nodes()}
        
        valid_seeds = [w for w in seed_wallets if w in self.graph]
        
        if not valid_seeds:
            return personalization
        
        for seed in valid_seeds:
            personalization[seed] = 1.0 / len(valid_seeds)
        
        # Calculate personalized PageRank
        ppr_scores = nx.pagerank(
            self.graph,
            alpha=alpha,
            personalization=personalization,
            weight='weight'
        )
        
        return ppr_scores
    
    def save_model(self, filepath='models/model6_network_tracing.pkl'):
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
            'pagerank_scores': self.pagerank_scores,
            'betweenness_scores': self.betweenness_scores,
            'alpha': self.alpha,
            'max_depth': self.max_depth,
            'is_fitted': self.is_fitted
        }
        
        joblib.dump(model_data, filepath)
        print(f"✓ Model 6 saved to {filepath}")
    
    def load_model(self, filepath='models/model6_network_tracing.pkl'):
        """
        Load trained model from disk
        
        Parameters:
        -----------
        filepath : str
            Path to load model from
        """
        model_data = joblib.load(filepath)
        
        self.graph = model_data['graph']
        self.pagerank_scores = model_data['pagerank_scores']
        self.betweenness_scores = model_data['betweenness_scores']
        self.alpha = model_data['alpha']
        self.max_depth = model_data['max_depth']
        self.is_fitted = model_data['is_fitted']
        
        print(f"✓ Model 6 loaded from {filepath}")
        
        return self


# Example usage
if __name__ == "__main__":
    # Load model5 output
    # df = pd.read_csv('data/model5_output.csv')
    
    # Initialize and train model
    model6 = NetworkTracingModel(alpha=0.85, max_depth=5)
    # model6.fit(df)
    
    # Add network features
    # result_df = model6.predict(df)
    
    # Trace suspicious wallet
    # suspicious_wallet = "0x1234..."
    # backward_trace = model6.trace_backwards(suspicious_wallet, depth=3)
    # forward_trace = model6.trace_forwards(suspicious_wallet, depth=3)
    
    # Find money flow paths
    # sources = ["0xabc...", "0xdef..."]
    # sinks = ["0x123...", "0x456..."]
    # paths = model6.identify_money_flow_paths(sources, sinks, k=10)
    
    # Save results
    # result_df.to_csv('data/model6_output.csv', index=False)
    # model6.save_model()
    
    pass