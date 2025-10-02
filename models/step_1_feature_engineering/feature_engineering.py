import pandas as pd
import numpy as np
import graph_tool.all as gt
from scipy.stats import entropy
from collections import Counter, defaultdict, deque
from datetime import timedelta
import warnings
import time
from multiprocessing import Pool, cpu_count
warnings.filterwarnings('ignore')

class BlockchainFeatureEngineer:
    """
    Comprehensive feature engineering for blockchain transaction data
    Generates graph, transaction, temporal, and pattern-specific features per wallet
    Uses graph-tool for high-performance graph operations
    """
    
    def __init__(self, transactions_df, wallets_df):
        """
        Initialize with transaction and wallet dataframes
        
        Args:
            transactions_df: DataFrame with columns [transaction_id, timestamp, from_address, 
                            to_address, amount, transaction_label, pattern]
            wallets_df: DataFrame with columns [wallet_id, address, type, pattern, is_illicit]
        """
        self.transactions_df = transactions_df.copy()
        self.wallets_df = wallets_df.copy()
        
        # Ensure timestamp is datetime
        if not pd.api.types.is_datetime64_any_dtype(self.transactions_df['timestamp']):
            self.transactions_df['timestamp'] = pd.to_datetime(self.transactions_df['timestamp'])
        
        # Build transaction graph
        self.graph = None
        self.vertex_map = None  # Maps address -> vertex index
        self.reverse_map = None  # Maps vertex index -> address
        self.wallet_features = None
        
        print(">> Feature Engineering initialized (graph-tool)")
        print(f"   Transactions: {len(self.transactions_df):,}")
        print(f"   Wallets: {len(self.wallets_df):,}")
    
    def build_transaction_graph(self):
        """Build directed graph from transactions using graph-tool"""
        print("\n>> Building transaction graph (graph-tool)...")
        
        # Create directed graph
        self.graph = gt.Graph(directed=True)
        
        # Create property maps for edge weights and counts
        edge_weight = self.graph.new_edge_property("double")
        edge_count = self.graph.new_edge_property("int")
        
        # Create vertex mapping
        self.vertex_map = {}
        self.reverse_map = {}
        
        # Add all wallets as vertices
        for idx, addr in enumerate(self.wallets_df['address']):
            v = self.graph.add_vertex()
            self.vertex_map[addr] = int(v)
            self.reverse_map[int(v)] = addr
        
        # Build edge dictionary for aggregation
        edge_dict = defaultdict(lambda: {'weight': 0, 'count': 0})
        
        for _, txn in self.transactions_df.iterrows():
            from_addr = txn['from_address']
            to_addr = txn['to_address']
            amount = txn['amount']
            
            if from_addr in self.vertex_map and to_addr in self.vertex_map:
                key = (from_addr, to_addr)
                edge_dict[key]['weight'] += amount
                edge_dict[key]['count'] += 1
        
        # Add edges with aggregated weights
        for (from_addr, to_addr), data in edge_dict.items():
            from_v = self.vertex_map[from_addr]
            to_v = self.vertex_map[to_addr]
            
            e = self.graph.add_edge(from_v, to_v)
            edge_weight[e] = data['weight']
            edge_count[e] = data['count']
        
        # Store property maps
        self.graph.edge_properties["weight"] = edge_weight
        self.graph.edge_properties["count"] = edge_count
        
        print(f"   Graph built: {self.graph.num_vertices():,} nodes, {self.graph.num_edges():,} edges")
        return self.graph

    def compute_graph_features(self):
        """Compute scalable graph-based features for each wallet using graph-tool"""
        print("\n>> Computing graph features (graph-tool optimized)...")
        
        if self.graph is None:
            self.build_transaction_graph()
        
        graph_features = {}
        
        # Precompute degrees (instant with graph-tool)
        print("   Precomputing degrees...")
        in_degrees = self.graph.get_in_degrees(self.graph.get_vertices())
        out_degrees = self.graph.get_out_degrees(self.graph.get_vertices())
        total_degrees = self.graph.get_total_degrees(self.graph.get_vertices())

        # Degree centrality
        print("   Computing degree centrality...")
        n = self.graph.num_vertices()
        degree_centrality = total_degrees / (n - 1) if n > 1 else np.zeros(n)

        # Betweenness centrality (sampled approximation, parallel)
        print("   Approximating betweenness centrality...")
        betweenness = gt.betweenness(self.graph, weight=self.graph.ep.weight)[0].a

        # Clustering coefficient (fast on undirected view)
        print("   Computing clustering coefficient...")
        clustering = gt.local_clustering(gt.GraphView(self.graph, directed=False)).a

        # PageRank (parallel, fast)
        print("   Computing PageRank...")
        try:
            pagerank = gt.pagerank(self.graph, weight=self.graph.ep.weight).a
        except:
            print("   Warning: PageRank failed, using uniform distribution")
            pagerank = np.ones(n) / n

        # Average neighbor degree
        print("   Computing average neighbor degrees...")
        avg_neighbor_degree = np.zeros(n)
        for v in self.graph.vertices():
            v_idx = int(v)
            in_neighbors = list(v.in_neighbors())
            out_neighbors = list(v.out_neighbors())
            neighbors = in_neighbors + out_neighbors
            if len(neighbors) > 0:
                avg_neighbor_degree[v_idx] = np.mean([total_degrees[int(nb)] for nb in neighbors])

        print("   Extracting per-wallet graph features...")
        for addr, v_idx in self.vertex_map.items():
            graph_features[addr] = {
                "degree_centrality": degree_centrality[v_idx],
                "in_degree": int(in_degrees[v_idx]),
                "out_degree": int(out_degrees[v_idx]),
                "total_degree": int(total_degrees[v_idx]),
                "betweenness_centrality": betweenness[v_idx],
                "clustering_coefficient": clustering[v_idx],
                "pagerank_score": pagerank[v_idx],
                "avg_neighbor_degree": avg_neighbor_degree[v_idx],
            }

        return pd.DataFrame.from_dict(graph_features, orient="index")

    
    def compute_transaction_features(self):
        """Compute transaction-based features for each wallet"""
        print("\n>> Computing transaction features...")
        
        transaction_features = {}
        
        for addr in self.wallets_df['address']:
            # Get sent and received transactions
            sent_txns = self.transactions_df[self.transactions_df['from_address'] == addr]
            received_txns = self.transactions_df[self.transactions_df['to_address'] == addr]
            all_txns = pd.concat([sent_txns, received_txns])
            
            # Basic transaction stats
            n_sent = len(sent_txns)
            n_received = len(received_txns)
            n_total = n_sent + n_received
            
            total_sent = sent_txns['amount'].sum() if n_sent > 0 else 0
            total_received = received_txns['amount'].sum() if n_received > 0 else 0
            total_volume = total_sent + total_received
            
            avg_amount = all_txns['amount'].mean() if n_total > 0 else 0
            max_amount = all_txns['amount'].max() if n_total > 0 else 0
            std_amount = all_txns['amount'].std() if n_total > 0 else 0
            
            # Ratio features
            sent_received_ratio = total_sent / total_received if total_received > 0 else 0
            
            # Unique counterparties
            unique_counterparties = len(set(sent_txns['to_address']) | set(received_txns['from_address']))
            
            # Gini coefficient (concentration of transaction amounts)
            amounts = all_txns['amount'].values
            if len(amounts) > 0:
                gini = self._compute_gini(amounts)
            else:
                gini = 0
            
            # Transaction frequency (per day)
            if n_total > 0:
                time_span = (all_txns['timestamp'].max() - all_txns['timestamp'].min()).total_seconds() / 86400
                txn_frequency = n_total / max(time_span, 1)
            else:
                txn_frequency = 0
            
            transaction_features[addr] = {
                'total_sent': total_sent,
                'total_received': total_received,
                'total_volume': total_volume,
                'n_transactions': n_total,
                'n_sent': n_sent,
                'n_received': n_received,
                'avg_transaction_amount': avg_amount,
                'max_transaction_amount': max_amount,
                'std_transaction_amount': std_amount,
                'sent_received_ratio': sent_received_ratio,
                'unique_counterparties': unique_counterparties,
                'gini_coefficient': gini,
                'transaction_frequency_per_day': txn_frequency
            }
        
        return pd.DataFrame.from_dict(transaction_features, orient='index')
    
    def compute_temporal_features(self):
        """Compute temporal features for each wallet"""
        print("\n>> Computing temporal features...")
        
        temporal_features = {}
        reference_date = self.transactions_df['timestamp'].max()
        
        for addr in self.wallets_df['address']:
            sent_txns = self.transactions_df[self.transactions_df['from_address'] == addr]
            received_txns = self.transactions_df[self.transactions_df['to_address'] == addr]
            all_txns = pd.concat([sent_txns, received_txns]).sort_values('timestamp')
            
            if len(all_txns) == 0:
                temporal_features[addr] = {
                    'account_age_days': 0,
                    'days_since_first_txn': 0,
                    'days_since_last_txn': 0,
                    'transaction_velocity': 0,
                    'avg_time_between_txns_hours': 0,
                    'active_days': 0,
                    'activity_ratio': 0,
                    'hourly_pattern_entropy': 0
                }
                continue
            
            first_txn = all_txns['timestamp'].min()
            last_txn = all_txns['timestamp'].max()
            
            # Account age and recency
            account_age_days = (last_txn - first_txn).total_seconds() / 86400
            days_since_first = (reference_date - first_txn).total_seconds() / 86400
            days_since_last = (reference_date - last_txn).total_seconds() / 86400
            
            # Transaction velocity (volume per day)
            transaction_velocity = all_txns['amount'].sum() / max(account_age_days, 1)
            
            # Average time between transactions
            if len(all_txns) > 1:
                time_diffs = all_txns['timestamp'].diff().dropna()
                avg_time_between = time_diffs.mean().total_seconds() / 3600  # hours
            else:
                avg_time_between = 0
            
            # Activity ratio (active days / total days)
            active_days = all_txns['timestamp'].dt.date.nunique()
            activity_ratio = active_days / max(account_age_days, 1)
            
            # Hourly pattern entropy
            hours = all_txns['timestamp'].dt.hour
            hour_counts = Counter(hours)
            hour_probs = np.array(list(hour_counts.values())) / len(hours)
            hourly_entropy = entropy(hour_probs)
            
            temporal_features[addr] = {
                'account_age_days': account_age_days,
                'days_since_first_txn': days_since_first,
                'days_since_last_txn': days_since_last,
                'transaction_velocity': transaction_velocity,
                'avg_time_between_txns_hours': avg_time_between,
                'active_days': active_days,
                'activity_ratio': activity_ratio,
                'hourly_pattern_entropy': hourly_entropy
            }
        
        return pd.DataFrame.from_dict(temporal_features, orient='index')
    
    @staticmethod
    def _compute_chain_length_worker(args):
        """
        Worker function for parallel chain length computation
        Uses graph_dict (lightweight adjacency list) instead of graph-tool object
        """
        addr, graph_dict = args
        
        if addr not in graph_dict:
            return (addr, 0)
        
        try:
            max_depth = 0
            visited = {addr}
            queue = deque([(addr, 0)])
            nodes_explored = 0
            max_nodes = 10000
            start_time = time.time()
            max_time = 2  # 2 seconds per wallet
            
            while queue and max_depth < 20:
                if time.time() - start_time > max_time:
                    break
                if nodes_explored >= max_nodes:
                    break
                
                current, depth = queue.popleft()
                max_depth = max(max_depth, depth)
                nodes_explored += 1
                
                if current in graph_dict:
                    for successor in graph_dict[current]:
                        if successor not in visited:
                            visited.add(successor)
                            queue.append((successor, depth + 1))
            
            return (addr, max_depth)
        except Exception as e:
            return (addr, 0)
    
    def _convert_graph_to_dict(self):
        """Convert graph-tool graph to lightweight adjacency list dictionary"""
        print("   Converting graph to adjacency list...")
        graph_dict = {}
        
        for v in self.graph.vertices():
            v_idx = int(v)
            addr = self.reverse_map[v_idx]
            successors = [self.reverse_map[int(succ)] for succ in v.out_neighbors()]
            graph_dict[addr] = successors
        
        return graph_dict
    
    def _precompute_chain_lengths_parallel(self):
        """Precompute chain lengths using multiprocessing"""
        print("   Precomputing chain lengths (parallel)...")
        
        if self.graph is None:
            return {}
        
        # Convert graph to picklable format
        graph_dict = self._convert_graph_to_dict()
        
        # Prepare arguments
        addresses = list(self.wallets_df['address'])
        args_list = [(addr, graph_dict) for addr in addresses]
        
        # Determine number of processes
        n_processes = max(1, cpu_count() - 1)
        print(f"   Using {n_processes} CPU cores")
        print(f"   Processing {len(addresses):,} wallets...")
        
        # Process in parallel
        chain_lengths = {}
        chunk_size = 100
        
        with Pool(processes=n_processes) as pool:
            results = pool.imap(
                self._compute_chain_length_worker,
                args_list,
                chunksize=chunk_size
            )
            
            for idx, (addr, chain_length) in enumerate(results):
                chain_lengths[addr] = chain_length
                
                if (idx + 1) % 2000 == 0:
                    progress = (idx + 1) / len(addresses) * 100
                    print(f"      Progress: {idx+1:,}/{len(addresses):,} ({progress:.1f}%)")
        
        print(f"   Chain length computation complete!")
        return chain_lengths
    
    def compute_pattern_specific_features(self):
        """Compute pattern-specific features for illicit detection"""
        print("\n>> Computing pattern-specific features...")
        
        # Precompute chain lengths in parallel
        chain_lengths = self._precompute_chain_lengths_parallel()
        
        # Precompute mixing services
        print("   Identifying mixing services...")
        mixing_services = set()
        if self.graph:
            in_degs = self.graph.get_in_degrees(self.graph.get_vertices())
            out_degs = self.graph.get_out_degrees(self.graph.get_vertices())
            
            for addr, v_idx in self.vertex_map.items():
                if in_degs[v_idx] > 100 and out_degs[v_idx] > 100:
                    mixing_services.add(addr)
        
        print(f"   Found {len(mixing_services)} mixing services")
        
        # Index transactions for fast lookup
        print("   Indexing transactions...")
        sent_groups = self.transactions_df.groupby('from_address')
        received_groups = self.transactions_df.groupby('to_address')
        
        pattern_features = {}
        total_wallets = len(self.wallets_df['address'])
        
        print(f"   Processing {total_wallets:,} wallets...")
        start_time = time.time()
        
        for idx, addr in enumerate(self.wallets_df['address']):
            if idx % 2000 == 0 and idx > 0:
                elapsed = time.time() - start_time
                rate = idx / elapsed
                remaining = (total_wallets - idx) / rate / 60
                print(f"   Progress: {idx:,}/{total_wallets:,} ({100*idx/total_wallets:.1f}%) - "
                      f"Est. {remaining:.1f} min remaining")
            
            # Get transactions using optimized lookup
            try:
                sent_txns = sent_groups.get_group(addr)
            except KeyError:
                sent_txns = pd.DataFrame(columns=self.transactions_df.columns)
            try:
                received_txns = received_groups.get_group(addr)
            except KeyError:
                received_txns = pd.DataFrame(columns=self.transactions_df.columns)
            
            # Use precomputed chain length
            chain_length = chain_lengths.get(addr, 0)
            
            # Splitting ratio (average outputs per input)
            if len(received_txns) > 0:
                splitting_ratio = len(sent_txns) / len(received_txns)
            else:
                splitting_ratio = 0
            
            # Amount decay rate (for peel chains)
            amount_decay_rate = self._compute_amount_decay(addr, sent_txns)
            
            # Dormancy period
            dormancy_days = self._compute_dormancy_period(addr, sent_txns, received_txns)
            
            # Sub-threshold transactions (below $10K)
            threshold = 10000
            n_sub_threshold = len(sent_txns[sent_txns['amount'] < threshold]) if len(sent_txns) > 0 else 0
            
            # Mixing service interaction
            mixing_interaction = int(any(to_addr in mixing_services 
                                        for to_addr in sent_txns['to_address'].unique()))
            
            # Rapid movement detection (transactions within 1 hour)
            rapid_movement_count = self._count_rapid_movements(addr, sent_txns)
            
            # Smurfing pattern (many similar amounts)
            smurfing_score = self._compute_smurfing_score(addr, sent_txns)
            
            pattern_features[addr] = {
                'chain_length': chain_length,
                'splitting_ratio': splitting_ratio,
                'amount_decay_rate': amount_decay_rate,
                'dormancy_period_days': dormancy_days,
                'n_sub_threshold_txns': n_sub_threshold,
                'mixing_service_interaction': mixing_interaction,
                'rapid_movement_count': rapid_movement_count,
                'smurfing_score': smurfing_score
            }
        
        total_time = time.time() - start_time
        print(f"\n   Completed all {total_wallets:,} wallets in {total_time/60:.1f} minutes!")
        return pd.DataFrame.from_dict(pattern_features, orient='index')
    
    def _compute_gini(self, amounts):
        """Compute Gini coefficient for transaction amounts"""
        if len(amounts) == 0:
            return 0
        sorted_amounts = np.sort(amounts)
        n = len(amounts)
        index = np.arange(1, n + 1)
        return (2 * np.sum(index * sorted_amounts)) / (n * np.sum(sorted_amounts)) - (n + 1) / n
    
    def _compute_amount_decay(self, addr, sent_txns):
        """Compute amount decay rate in transaction chains"""
        if len(sent_txns) < 2:
            return 0
        
        sent_txns = sent_txns.sort_values('timestamp')
        amounts = sent_txns['amount'].values
        
        if len(amounts) < 2:
            return 0
        
        # Compute percentage change between consecutive transactions
        decay_rates = []
        for i in range(len(amounts) - 1):
            if amounts[i] > 0:
                decay = (amounts[i] - amounts[i+1]) / amounts[i]
                decay_rates.append(decay)
        
        return np.mean(decay_rates) if decay_rates else 0
    
    def _compute_dormancy_period(self, addr, sent_txns, received_txns):
        """Compute longest dormancy period (gap between transactions)"""
        all_txns = pd.concat([sent_txns, received_txns])
        
        if len(all_txns) < 2:
            return 0
        
        all_txns = all_txns.sort_values('timestamp')
        time_diffs = all_txns['timestamp'].diff().dropna()
        
        if len(time_diffs) == 0:
            return 0
        
        max_gap_days = time_diffs.max().total_seconds() / 86400
        return max_gap_days
    
    def _count_rapid_movements(self, addr, sent_txns):
        """Count transactions that occur within 1 hour of each other"""
        if len(sent_txns) < 2:
            return 0
        
        sent_txns = sent_txns.sort_values('timestamp')
        rapid_count = 0
        
        for i in range(len(sent_txns) - 1):
            time_diff = (sent_txns.iloc[i+1]['timestamp'] - sent_txns.iloc[i]['timestamp']).total_seconds() / 3600
            if time_diff < 1:
                rapid_count += 1
        
        return rapid_count
    
    def _compute_smurfing_score(self, addr, sent_txns):
        """Detect smurfing pattern (many similar amounts just below threshold)"""
        if len(sent_txns) < 5:
            return 0
        
        # Check for transactions clustered around $9000-$9900
        threshold_range = sent_txns[
            (sent_txns['amount'] >= 8000) & 
            (sent_txns['amount'] <= 9900)
        ]
        
        if len(sent_txns) > 0:
            smurfing_ratio = len(threshold_range) / len(sent_txns)
            return smurfing_ratio
        
        return 0
    
    def engineer_all_features(self):
        """
        Main method to engineer all features
        Returns: Complete feature DataFrame with all features per wallet
        """
        print("\n" + "="*60)
        print("FEATURE ENGINEERING PIPELINE (graph-tool + parallel)")
        print("="*60)
        
        # Compute all feature groups
        graph_features_df = self.compute_graph_features()
        transaction_features_df = self.compute_transaction_features()
        temporal_features_df = self.compute_temporal_features()
        pattern_features_df = self.compute_pattern_specific_features()
        
        # Merge all features
        print("\n>> Merging all feature groups...")
        self.wallet_features = self.wallets_df.set_index('address').join([
            graph_features_df,
            transaction_features_df,
            temporal_features_df,
            pattern_features_df
        ])
        
        # Fill NaN values
        self.wallet_features = self.wallet_features.fillna(0)
        
        # Reset index to make address a column
        self.wallet_features = self.wallet_features.reset_index()
        
        print("\n" + "="*60)
        print("FEATURE ENGINEERING COMPLETE")
        print("="*60)
        print(f"Total features created: {len(self.wallet_features.columns) - 6}")  # Exclude original wallet columns
        print(f"Feature DataFrame shape: {self.wallet_features.shape}")
        
        # Display feature summary
        feature_groups = {
            'Graph Features': 8,
            'Transaction Features': 13,
            'Temporal Features': 8,
            'Pattern-Specific Features': 8
        }
        
        print("\nFeature breakdown:")
        for group, count in feature_groups.items():
            print(f"  {group}: {count}")
        
        return self.wallet_features
    
    def save_features(self, filepath='refined_data/blockchain_wallet_features.csv'):
        """Save engineered features to CSV"""
        if self.wallet_features is None:
            raise ValueError("Features not yet engineered. Run engineer_all_features() first.")
        
        self.wallet_features.to_csv(filepath, index=False)
        print(f"\n>> Features saved to: {filepath}")
        return filepath
    
    def get_feature_names(self):
        """Return list of all engineered feature names"""
        if self.wallet_features is None:
            return []
        
        # Exclude original wallet metadata columns
        exclude_cols = ['wallet_id', 'address', 'type', 'pattern', 'is_illicit', 'created_at']
        feature_cols = [col for col in self.wallet_features.columns if col not in exclude_cols]
        
        return feature_cols


# ============================================================================
# USAGE EXAMPLE
# ============================================================================

if __name__ == "__main__":
    # Load the generated datasets
    print("Loading datasets...")
    wallets_df = pd.read_csv("/home/ivar03/Desktop/mule_bank_account_detection_pipeline/Mule-Account-Detection/data/dataset/blockchain_wallets.csv")
    transactions_df = pd.read_csv("/home/ivar03/Desktop/mule_bank_account_detection_pipeline/Mule-Account-Detection/data/dataset/blockchain_transactions.csv")
    
    print(f"Loaded {len(wallets_df):,} wallets and {len(transactions_df):,} transactions")
    
    # Initialize feature engineer
    feature_engineer = BlockchainFeatureEngineer(transactions_df, wallets_df)
    
    # Engineer all features
    wallet_features = feature_engineer.engineer_all_features()
    
    # Save features
    feature_engineer.save_features('/home/ivar03/Desktop/mule_bank_account_detection_pipeline/Mule-Account-Detection/models/step_1_feature_engineering/refined_data/blockchain_wallet_features.csv')
    
    # Display sample
    print("\nSample features (first 5 wallets):")
    feature_cols = feature_engineer.get_feature_names()
    print(wallet_features[['address', 'is_illicit'] + feature_cols[:10]].head())
    
    print("\nFeature engineering complete!")