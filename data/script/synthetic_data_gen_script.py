import pandas as pd
import numpy as np
import uuid
from datetime import datetime, timedelta
import random
from collections import defaultdict
import hashlib
import sys

# Fix encoding for Windows
if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8')

class BlockchainSyntheticDataGenerator:
    """
    Generate synthetic blockchain transaction data for money laundering detection
    Target: 700K+ transactions, 50K wallets, 1-2 year timespan
    """
    
    def __init__(self, 
                 n_wallets=50000,
                 n_transactions=2000000,
                 illicit_wallet_ratio=0.15,  # 15% illicit wallets
                 illicit_txn_ratio=0.18,     # 18% illicit transactions (realistic target)
                 start_date='2023-01-01',
                 duration_days=730):          # 2 years
        
        self.n_wallets = n_wallets
        self.n_transactions = n_transactions
        self.illicit_wallet_ratio = illicit_wallet_ratio
        self.illicit_txn_ratio = illicit_txn_ratio
        self.start_date = datetime.strptime(start_date, '%Y-%m-%d')
        self.duration_days = duration_days
        self.end_date = self.start_date + timedelta(days=duration_days)
        
        # Data storage
        self.wallets = {}
        self.transactions = []
        self.wallet_balances = defaultdict(float)
        
        # Counters
        self.transaction_counter = 0
        self.block_counter = 1
        
        print(">> Initializing Blockchain Synthetic Data Generator")
        print(f">> Target: {n_transactions:,} transactions across {n_wallets:,} wallets")
        print(f">> Illicit ratio: {illicit_wallet_ratio*100:.1f}% wallets, {illicit_txn_ratio*100:.1f}% transactions")
        
    def generate_address(self):
        """Generate a realistic-looking blockchain address"""
        return "0x" + hashlib.sha256(uuid.uuid4().bytes).hexdigest()[:40]
    
    def generate_transaction_hash(self):
        """Generate transaction hash"""
        return "0x" + hashlib.sha256(str(self.transaction_counter).encode()).hexdigest()
    
    def random_timestamp(self, start=None, end=None):
        """Generate random timestamp within range"""
        if start is None:
            start = self.start_date
        if end is None:
            end = self.end_date
        
        time_diff = (end - start).total_seconds()
        random_seconds = random.uniform(0, time_diff)
        return start + timedelta(seconds=random_seconds)
    
    def create_wallets(self):
        """Create wallet addresses with assigned patterns"""
        print("\n>> Creating wallets...")
        
        n_illicit = int(self.n_wallets * self.illicit_wallet_ratio)
        n_legitimate = self.n_wallets - n_illicit
        
        # Legitimate wallet types
        legitimate_types = {
            'merchant': int(n_legitimate * 0.25),
            'individual': int(n_legitimate * 0.45),
            'exchange': int(n_legitimate * 0.15),
            'saver': int(n_legitimate * 0.15)
        }
        
        # Illicit wallet patterns
        illicit_patterns = {
            'layering': int(n_illicit * 0.30),
            'smurfing': int(n_illicit * 0.25),
            'rapid_movement': int(n_illicit * 0.15),
            'mixing': int(n_illicit * 0.15),
            'peel_chain': int(n_illicit * 0.10),
            'dormant_active': int(n_illicit * 0.05)
        }
        
        wallet_id = 0
        
        # Create legitimate wallets
        for wallet_type, count in legitimate_types.items():
            for _ in range(count):
                address = self.generate_address()
                self.wallets[address] = {
                    'wallet_id': wallet_id,
                    'address': address,
                    'type': wallet_type,
                    'pattern': 'legitimate',
                    'is_illicit': False,
                    'created_at': self.random_timestamp()
                }
                self.wallet_balances[address] = random.uniform(5000, 100000)
                wallet_id += 1
        
        # Create illicit wallets with higher balances
        for pattern, count in illicit_patterns.items():
            for _ in range(count):
                address = self.generate_address()
                self.wallets[address] = {
                    'wallet_id': wallet_id,
                    'address': address,
                    'type': 'illicit',
                    'pattern': pattern,
                    'is_illicit': True,
                    'created_at': self.random_timestamp()
                }
                # Illicit wallets start with more funds
                self.wallet_balances[address] = random.uniform(100000, 500000)
                wallet_id += 1
        
        print(f">> Created {len(self.wallets):,} wallets")
        print(f"   - Legitimate: {n_legitimate:,}")
        print(f"   - Illicit: {n_illicit:,}")
        
        return self.wallets
    
    def add_transaction(self, from_addr, to_addr, amount, timestamp, label='legitimate', pattern='normal'):
        """Add a transaction to the dataset"""
        
        # Validate balance
        if self.wallet_balances.get(from_addr, 0) < amount:
            return None
        
        # Update balances
        self.wallet_balances[from_addr] -= amount
        self.wallet_balances[to_addr] = self.wallet_balances.get(to_addr, 0) + amount
        
        # Generate gas fee
        gas_fee = amount * random.uniform(0.001, 0.005)
        
        txn = {
            'transaction_id': self.generate_transaction_hash(),
            'block_number': self.block_counter,
            'timestamp': timestamp,
            'from_address': from_addr,
            'to_address': to_addr,
            'amount': round(amount, 2),
            'gas_fee': round(gas_fee, 4),
            'transaction_label': label,
            'pattern': pattern
        }
        
        self.transactions.append(txn)
        self.transaction_counter += 1
        
        if self.transaction_counter % random.randint(10, 20) == 0:
            self.block_counter += 1
        
        return txn
    
    def generate_merchant_transactions(self, merchant_addr, n_txns):
        """Generate transactions for a merchant"""
        legitimate_wallets = [addr for addr, info in self.wallets.items() 
                             if not info['is_illicit']]
        
        for _ in range(n_txns):
            if len(legitimate_wallets) == 0:
                break
            customer = random.choice(legitimate_wallets)
            amount = random.uniform(10, 500)
            timestamp = self.random_timestamp()
            self.add_transaction(customer, merchant_addr, amount, timestamp, 'legitimate', 'merchant_payment')
    
    def generate_individual_transactions(self, individual_addr, n_txns):
        """Generate sporadic transactions for individual users"""
        all_wallets = list(self.wallets.keys())
        
        for _ in range(n_txns):
            if random.random() < 0.5:
                recipient = random.choice(all_wallets)
                amount = random.uniform(50, 5000)
                timestamp = self.random_timestamp()
                self.add_transaction(individual_addr, recipient, amount, timestamp, 'legitimate', 'p2p_transfer')
            else:
                sender = random.choice(all_wallets)
                amount = random.uniform(50, 5000)
                timestamp = self.random_timestamp()
                self.add_transaction(sender, individual_addr, amount, timestamp, 'legitimate', 'p2p_transfer')
    
    def generate_exchange_transactions(self, exchange_addr, n_txns):
        """Generate high-volume exchange transactions"""
        all_wallets = list(self.wallets.keys())
        
        for _ in range(n_txns):
            counterparty = random.choice(all_wallets)
            amount = random.uniform(100, 50000)
            timestamp = self.random_timestamp()
            
            if random.random() < 0.5:
                self.add_transaction(counterparty, exchange_addr, amount, timestamp, 'legitimate', 'exchange_deposit')
            else:
                self.add_transaction(exchange_addr, counterparty, amount, timestamp, 'legitimate', 'exchange_withdrawal')
    
    def generate_saver_transactions(self, saver_addr, n_txns):
        """Generate saver pattern"""
        legitimate_wallets = [addr for addr, info in self.wallets.items() 
                             if not info['is_illicit']]
        
        for _ in range(int(n_txns * 0.8)):
            sender = random.choice(legitimate_wallets)
            amount = random.uniform(500, 10000)
            timestamp = self.random_timestamp()
            self.add_transaction(sender, saver_addr, amount, timestamp, 'legitimate', 'savings_deposit')
        
        for _ in range(int(n_txns * 0.2)):
            recipient = random.choice(legitimate_wallets)
            amount = random.uniform(100, 2000)
            timestamp = self.random_timestamp()
            self.add_transaction(saver_addr, recipient, amount, timestamp, 'legitimate', 'savings_withdrawal')
    
    def generate_layering_pattern(self, source_addr):
        """Layering pattern with balance check"""
        if self.wallet_balances.get(source_addr, 0) < 50000:
            return
            
        initial_amount = min(self.wallet_balances[source_addr] * 0.8, random.uniform(50000, 200000))
        n_intermediaries = random.randint(10, 30)
        n_hops = random.randint(3, 5)
        
        intermediaries = [self.generate_address() for _ in range(n_intermediaries)]
        for inter in intermediaries:
            self.wallet_balances[inter] = 0
        
        split_amount = initial_amount / n_intermediaries
        base_time = self.random_timestamp()
        
        # Split phase
        for inter in intermediaries:
            timestamp = base_time + timedelta(minutes=random.randint(1, 30))
            self.add_transaction(source_addr, inter, split_amount, timestamp, 'illicit', 'layering_split')
        
        # Hop phase
        for hop in range(n_hops - 1):
            next_intermediaries = [self.generate_address() for _ in range(n_intermediaries)]
            for next_inter in next_intermediaries:
                self.wallet_balances[next_inter] = 0
            
            for current, next_addr in zip(intermediaries, next_intermediaries):
                timestamp = base_time + timedelta(hours=hop+1, minutes=random.randint(1, 59))
                amount = self.wallet_balances.get(current, 0) * 0.95
                if amount > 0:
                    self.add_transaction(current, next_addr, amount, timestamp, 'illicit', 'layering_hop')
            
            intermediaries = next_intermediaries
        
        # Recombine phase
        final_destination = self.generate_address()
        self.wallet_balances[final_destination] = 0
        
        for inter in intermediaries:
            timestamp = base_time + timedelta(hours=n_hops+1, minutes=random.randint(1, 59))
            amount = self.wallet_balances.get(inter, 0) * 0.95
            if amount > 0:
                self.add_transaction(inter, final_destination, amount, timestamp, 'illicit', 'layering_recombine')
    
    def generate_smurfing_pattern(self, source_addr):
        """Smurfing pattern"""
        if self.wallet_balances.get(source_addr, 0) < 100000:
            return
            
        total_amount = min(self.wallet_balances[source_addr] * 0.7, random.uniform(100000, 500000))
        threshold = 9500
        n_transactions = int(total_amount / threshold) + random.randint(5, 15)
        
        all_wallets = list(self.wallets.keys())
        base_time = self.random_timestamp()
        
        for i in range(n_transactions):
            recipient = random.choice(all_wallets)
            amount = random.uniform(8000, 9900)
            timestamp = base_time + timedelta(days=random.randint(0, 30), hours=random.randint(0, 23))
            result = self.add_transaction(source_addr, recipient, amount, timestamp, 'illicit', 'smurfing')
            if result is None:
                break
    
    def generate_rapid_movement_pattern(self, source_addr):
        """Rapid movement pattern"""
        if self.wallet_balances.get(source_addr, 0) < 20000:
            return
            
        amount = min(self.wallet_balances[source_addr] * 0.8, random.uniform(20000, 100000))
        n_hops = random.randint(5, 15)
        
        current_addr = source_addr
        base_time = self.random_timestamp()
        
        for hop in range(n_hops):
            next_addr = self.generate_address()
            self.wallet_balances[next_addr] = 0
            timestamp = base_time + timedelta(minutes=hop * random.randint(2, 10))
            amount = amount * 0.98
            result = self.add_transaction(current_addr, next_addr, amount, timestamp, 'illicit', 'rapid_movement')
            if result is None:
                break
            current_addr = next_addr
    
    def generate_mixing_service_pattern(self, source_addr):
        """Mixing service pattern"""
        mixer_addr = self.generate_address()
        self.wallet_balances[mixer_addr] = 0
        
        n_inputs = random.randint(5, 20)
        input_amounts = []
        base_time = self.random_timestamp()
        all_wallets = list(self.wallets.keys())
        
        for i in range(n_inputs):
            amount = random.uniform(5000, 50000)
            input_amounts.append(amount)
            timestamp = base_time + timedelta(minutes=random.randint(1, 120))
            
            if i == 0:
                sender = source_addr
            else:
                sender = random.choice(all_wallets)
            
            self.add_transaction(sender, mixer_addr, amount, timestamp, 'illicit', 'mixing_input')
        
        mix_time = base_time + timedelta(hours=random.randint(2, 48))
        total_mixed = sum(input_amounts) * 0.95
        n_outputs = random.randint(5, 20)
        
        for i in range(n_outputs):
            output_addr = self.generate_address()
            self.wallet_balances[output_addr] = 0
            amount = total_mixed / n_outputs
            timestamp = mix_time + timedelta(minutes=random.randint(1, 180))
            self.add_transaction(mixer_addr, output_addr, amount, timestamp, 'illicit', 'mixing_output')
    
    def generate_peel_chain_pattern(self, source_addr):
        """Peel chain pattern"""
        if self.wallet_balances.get(source_addr, 0) < 50000:
            return
            
        initial_amount = min(self.wallet_balances[source_addr] * 0.7, random.uniform(50000, 200000))
        peel_percentage = random.uniform(0.03, 0.10)
        
        current_addr = source_addr
        remaining_amount = initial_amount
        base_time = self.random_timestamp()
        hop = 0
        
        while remaining_amount > 1000 and hop < 20:
            peel_amount = remaining_amount * peel_percentage
            remaining_amount -= peel_amount
            
            peel_recipient = self.generate_address()
            self.wallet_balances[peel_recipient] = 0
            timestamp = base_time + timedelta(hours=hop, minutes=random.randint(1, 59))
            self.add_transaction(current_addr, peel_recipient, peel_amount, timestamp, 'illicit', 'peel_chain_peel')
            
            next_addr = self.generate_address()
            self.wallet_balances[next_addr] = 0
            timestamp = base_time + timedelta(hours=hop, minutes=random.randint(30, 59))
            result = self.add_transaction(current_addr, next_addr, remaining_amount, timestamp, 'illicit', 'peel_chain_transfer')
            if result is None:
                break
                
            current_addr = next_addr
            hop += 1
    
    def generate_dormant_active_pattern(self, source_addr):
        """Dormant-then-active pattern"""
        large_amount = random.uniform(50000, 300000)
        
        all_wallets = list(self.wallets.keys())
        sender = random.choice(all_wallets)
        deposit_time = self.random_timestamp(end=self.start_date + timedelta(days=365))
        self.add_transaction(sender, source_addr, large_amount, deposit_time, 'illicit', 'dormant_deposit')
        
        dormancy_days = random.randint(90, 180)
        activation_time = deposit_time + timedelta(days=dormancy_days)
        
        n_burst_txns = random.randint(10, 50)
        for i in range(n_burst_txns):
            recipient = self.generate_address()
            self.wallet_balances[recipient] = 0
            amount = random.uniform(1000, 10000)
            timestamp = activation_time + timedelta(days=random.randint(0, 7), hours=random.randint(0, 23))
            result = self.add_transaction(source_addr, recipient, amount, timestamp, 'illicit', 'dormant_active_burst')
            if result is None:
                break
    
    def refill_illicit_wallets(self):
        """Refill illicit wallets aggressively"""
        for addr, info in self.wallets.items():
            if info['is_illicit'] and self.wallet_balances.get(addr, 0) < 50000:
                self.wallet_balances[addr] += random.uniform(100000, 300000)
    
    def generate_all_transactions(self):
        """Generate all transactions"""
        print("\n>> Generating transactions...")
        
        legitimate_wallets = {addr: info for addr, info in self.wallets.items() 
                             if not info['is_illicit']}
        illicit_wallets = {addr: info for addr, info in self.wallets.items() 
                          if info['is_illicit']}
        
        n_legitimate_txns = int(self.n_transactions * (1 - self.illicit_txn_ratio))
        
        merchants = [addr for addr, info in legitimate_wallets.items() if info['type'] == 'merchant']
        individuals = [addr for addr, info in legitimate_wallets.items() if info['type'] == 'individual']
        exchanges = [addr for addr, info in legitimate_wallets.items() if info['type'] == 'exchange']
        savers = [addr for addr, info in legitimate_wallets.items() if info['type'] == 'saver']
        
        print(f"   Generating legitimate transactions ({n_legitimate_txns:,} target)...")
        
        merchant_txns_each = int((n_legitimate_txns * 0.30) / len(merchants)) if merchants else 0
        for merchant in merchants[:int(len(merchants) * 0.5)]:
            self.generate_merchant_transactions(merchant, merchant_txns_each)
        
        individual_txns_each = int((n_legitimate_txns * 0.40) / len(individuals)) if individuals else 0
        for individual in individuals[:int(len(individuals) * 0.6)]:
            self.generate_individual_transactions(individual, individual_txns_each)
        
        exchange_txns_each = int((n_legitimate_txns * 0.20) / len(exchanges)) if exchanges else 0
        for exchange in exchanges:
            self.generate_exchange_transactions(exchange, exchange_txns_each)
        
        saver_txns_each = int((n_legitimate_txns * 0.10) / len(savers)) if savers else 0
        for saver in savers[:int(len(savers) * 0.5)]:
            self.generate_saver_transactions(saver, saver_txns_each)
        
        print(f"   >> Generated {len(self.transactions):,} legitimate transactions")
        print(f"   Generating illicit transactions...")
        
        illicit_by_pattern = defaultdict(list)
        for addr, info in illicit_wallets.items():
            illicit_by_pattern[info['pattern']].append(addr)
        
        # Generate illicit patterns with multiple rounds and refills
        for round_num in range(1):
            self.refill_illicit_wallets()
            
            for addr in illicit_by_pattern['layering']:
                self.generate_layering_pattern(addr)
            
            for addr in illicit_by_pattern['smurfing']:
                self.generate_smurfing_pattern(addr)
            
            for addr in illicit_by_pattern['rapid_movement']:
                self.generate_rapid_movement_pattern(addr)
            
            for addr in illicit_by_pattern['mixing']:
                self.generate_mixing_service_pattern(addr)
            
            for addr in illicit_by_pattern['peel_chain']:
                self.generate_peel_chain_pattern(addr)
            
            for addr in illicit_by_pattern['dormant_active']:
                self.generate_dormant_active_pattern(addr)
            
            print(f"   >> Round {round_num + 1} complete: {len(self.transactions):,} total transactions")
        
        illicit_count = sum(1 for t in self.transactions if t['transaction_label'] == 'illicit')
        print(f"   >> Final: {len(self.transactions):,} total transactions")
        print(f"   >> Illicit transaction ratio: {(illicit_count / len(self.transactions) * 100):.2f}%")
        
    def create_dataframes(self):
        """Convert to pandas DataFrames"""
        print("\n>> Creating DataFrames...")
        
        wallets_df = pd.DataFrame(self.wallets.values())
        transactions_df = pd.DataFrame(self.transactions)
        transactions_df = transactions_df.sort_values('timestamp').reset_index(drop=True)
        
        print(f">> Wallets DataFrame: {len(wallets_df):,} rows")
        print(f">> Transactions DataFrame: {len(transactions_df):,} rows")
        
        return wallets_df, transactions_df
    
    def generate_dataset(self):
        """Main function to generate complete dataset"""
        print("\n" + "="*60)
        print("BLOCKCHAIN MONEY LAUNDERING DETECTION DATASET GENERATOR")
        print("="*60)
        
        self.create_wallets()
        self.generate_all_transactions()
        wallets_df, transactions_df = self.create_dataframes()
        
        print("\n" + "="*60)
        print("DATASET STATISTICS")
        print("="*60)
        print(f"Total Wallets: {len(wallets_df):,}")
        print(f"  - Legitimate: {len(wallets_df[~wallets_df['is_illicit']]):,} ({len(wallets_df[~wallets_df['is_illicit']])/len(wallets_df)*100:.1f}%)")
        print(f"  - Illicit: {len(wallets_df[wallets_df['is_illicit']]):,} ({len(wallets_df[wallets_df['is_illicit']])/len(wallets_df)*100:.1f}%)")
        print(f"\nTotal Transactions: {len(transactions_df):,}")
        print(f"  - Legitimate: {len(transactions_df[transactions_df['transaction_label']=='legitimate']):,} ({len(transactions_df[transactions_df['transaction_label']=='legitimate'])/len(transactions_df)*100:.1f}%)")
        print(f"  - Illicit: {len(transactions_df[transactions_df['transaction_label']=='illicit']):,} ({len(transactions_df[transactions_df['transaction_label']=='illicit'])/len(transactions_df)*100:.1f}%)")
        print(f"\nTime Range: {transactions_df['timestamp'].min()} to {transactions_df['timestamp'].max()}")
        print(f"Total Transaction Volume: ${transactions_df['amount'].sum():,.2f}")
        print(f"Average Transaction Amount: ${transactions_df['amount'].mean():,.2f}")
        print("\nPattern Distribution:")
        print(transactions_df['pattern'].value_counts())
        print("\n" + "="*60)
        
        return wallets_df, transactions_df
    
    def save_to_csv(self, wallets_df, transactions_df, 
                    wallets_file='blockchain_wallets.csv',
                    transactions_file='blockchain_transactions.csv'):
        """Save datasets to CSV files"""
        print(f"\n>> Saving datasets...")
        wallets_df.to_csv(wallets_file, index=False)
        transactions_df.to_csv(transactions_file, index=False)
        print(f">> Saved: {wallets_file}")
        print(f">> Saved: {transactions_file}")


if __name__ == "__main__":
    generator = BlockchainSyntheticDataGenerator(
        n_wallets=50000,
        n_transactions=2000000,
        illicit_wallet_ratio=0.1,
        illicit_txn_ratio=0.15,
        start_date='2023-01-01',
        duration_days=730
    )
    
    wallets_df, transactions_df = generator.generate_dataset()
    generator.save_to_csv(wallets_df, transactions_df)
    
    print("\nWallets Preview:")
    print(wallets_df.head(10))
    print("\nTransactions Preview:")
    print(transactions_df.head(10))
    print("\nDataset generation complete!")