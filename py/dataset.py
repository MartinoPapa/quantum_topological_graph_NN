# dataset.py
import numpy as np
import networkx as nx
import pandas as pd
import torch
import os
import pickle
from config import *

class BaseGraphDataset:
    def __init__(self):
        self.graphs = []
        self.labels = []
        self.num_nodes = NUM_NODES
        self.G_global = nx.DiGraph()

    def finalize_and_sample(self):
        print(f"Grafo Globale: {self.G_global.number_of_nodes()} nodi, {self.G_global.number_of_edges()} archi.")
        print(f"Campionamento (Random Walk) di {TOTAL_GRAPHS} sottografi (Target Fraud: {int(TOTAL_GRAPHS * FRAUD_RATIO)})...")
        
        # Rimuovi nodi isolati o non etichettati
        nodes_to_remove = [n for n, d in self.G_global.nodes(data=True) if d.get('isFraud', -1) == -1 and self.G_global.degree(n) == 0]
        self.G_global.remove_nodes_from(nodes_to_remove)
        
        self._sample_subgraphs_random_walk()

    def _sample_subgraphs_random_walk(self):
        fraud_nodes = [n for n, d in self.G_global.nodes(data=True) if d.get('isFraud') == 1]
        normal_nodes = [n for n, d in self.G_global.nodes(data=True) if d.get('isFraud') == 0]
        
        if not fraud_nodes:
            print("ATTENZIONE: Nessun nodo fraudolento trovato nel grafo globale!")
            
        degrees = dict(self.G_global.degree())
        
        # Ordina per grado per dare priorità agli hub (high-risk clusters)
        normal_nodes.sort(key=lambda n: degrees.get(n, 0), reverse=True)
        np.random.shuffle(fraud_nodes)
        
        target_fraud_count = int(TOTAL_GRAPHS * FRAUD_RATIO)
        target_normal_count = TOTAL_GRAPHS - target_fraud_count
        
        G_undirected = self.G_global.to_undirected(as_view=True)
        global_edge_count = self.G_global.number_of_edges()
        
        # --- Estrazione Sottografi Fraudolenti ---
        count_fraud = 0; fraud_idx = 0
        while count_fraud < target_fraud_count and fraud_idx < len(fraud_nodes):
            subG = self._biased_random_walk(fraud_nodes[fraud_idx], G_undirected, degrees)
            fraud_idx += 1
            if self._is_valid_subgraph(subG, global_edge_count):
                self.graphs.append(subG)
                self.labels.append(1)
                count_fraud += 1

        # --- Estrazione Sottografi Normali ---
        count_normal = 0; norm_idx = 0
        while count_normal < target_normal_count and norm_idx < len(normal_nodes):
            subG = self._biased_random_walk(normal_nodes[norm_idx], G_undirected, degrees)
            norm_idx += 1
            if self._is_valid_subgraph(subG, global_edge_count):
                f_flags = nx.get_node_attributes(subG, 'isFraud').values()
                if 1 not in f_flags:
                    self.graphs.append(subG)
                    self.labels.append(0)
                    count_normal += 1

    def _biased_random_walk(self, start_node, G_view, degrees):
        """
        Implementa il Random Walk per preservare frequent transactions (pesi) 
        e high-degree nodes (gradi), come richiesto dal paper.
        """
        sampled_nodes = {start_node}
        current_node = start_node
        
        # Preveniamo loop infiniti se il walker rimane bloccato in un vicolo cieco
        max_steps = self.num_nodes * 10 
        steps = 0
        
        while len(sampled_nodes) < self.num_nodes and steps < max_steps:
            neighbors = list(G_view.neighbors(current_node))
            
            if not neighbors:
                # Vicolo cieco: Teletrasporto a un nodo già campionato a caso
                current_node = list(sampled_nodes)[np.random.randint(len(sampled_nodes))]
                steps += 1
                continue
                
            # Calcolo delle probabilità di transizione biasate
            weights = []
            for n in neighbors:
                edge_data = G_view.get_edge_data(current_node, n, default={})
                w = edge_data.get('weight', 1.0)  # Frequent transactions
                d = degrees.get(n, 1)             # High-degree nodes (community structure)
                
                # Moltiplichiamo peso e grado per favorire transazioni pesanti verso hub
                weights.append(w * d) 
            
            # Normalizzazione softmax-style semplice per ottenere probabilità [0, 1]
            weights = np.array(weights)
            if weights.sum() == 0:
                probs = np.ones(len(neighbors)) / len(neighbors)
            else:
                probs = weights / weights.sum()
            
            # Salto al prossimo nodo
            next_node = np.random.choice(neighbors, p=probs)
            sampled_nodes.add(next_node)
            current_node = next_node
            steps += 1
            
        return self.G_global.subgraph(list(sampled_nodes)).copy()

    def _is_valid_subgraph(self, subG, global_edge_count):
        if subG.number_of_nodes() < 4: return False
        if subG.number_of_edges() < SAMPLED_EDGES: return False
        
        # Eq (28) constraint: |E'| <= kappa * |E|
        # In a real dynamic scenario, kappa is updated via F1 score.
        # Here we apply a static safety check to ensure subgraphs aren't massive.
        kappa_max = 0.05 # Max 5% of global edges
        if subG.number_of_edges() > (kappa_max * global_edge_count):
            return False
            
        if not nx.is_weakly_connected(subG):
            largest_cc = max(nx.weakly_connected_components(subG), key=len)
            if len(largest_cc) < self.num_nodes * 0.8: return False
            
        return True

    def get_matrix_A(self, g):
        # [This method remains exactly the same as the previous BaseGraphDataset]
        g = nx.convert_node_labels_to_integers(g)
        N = self.num_nodes
        adj = np.zeros((N, N)) 
        
        if g.number_of_nodes() > 0:
            nodes = sorted(list(g.nodes))
            for u, v, data in g.edges(data=True):
                i, j = nodes.index(u), nodes.index(v)
                adj[i, j] = data.get('weight', 1.0)
            
            in_deg = dict(g.in_degree(weight='weight'))
            out_deg = dict(g.out_degree(weight='weight'))
            total_weight = g.size(weight='weight') + 1e-9
            if total_weight == 0: total_weight = g.number_of_edges() + 1e-9
            
            for i, node in enumerate(nodes):
                deg = in_deg.get(node, 0) + out_deg.get(node, 0)
                adj[i, i] = deg / total_weight

        feature_vec = adj.flatten() 
        target_dim = 2**TOTAL_QUBITS 
        if len(feature_vec) < target_dim:
            feature_vec = np.pad(feature_vec, (0, target_dim - len(feature_vec)))
        else:
            feature_vec = feature_vec[:target_dim]
            
        norm = np.linalg.norm(feature_vec)
        if norm < 1e-9: norm = 1.0
        return feature_vec / norm
# ==========================================
# 1. PAYSIM DATASET
# ==========================================
class PaySimDataset(BaseGraphDataset):
    def __init__(self, csv_file, pool_file="pool_nodes.pkl"):
        super().__init__()
        print("Caricamento PaySim Dataset...")
        
        if os.path.exists(pool_file):
            with open(pool_file, "rb") as f:
                self.allowed_nodes = pickle.load(f)
        else:
            raise FileNotFoundError(f"Esegui prima lo script per generare {pool_file}")

        chunk_iter = pd.read_csv(csv_file, chunksize=500000, usecols=['amount', 'nameOrig', 'nameDest', 'isFraud'])
        for chunk in chunk_iter:
            mask = chunk['nameOrig'].isin(self.allowed_nodes) & chunk['nameDest'].isin(self.allowed_nodes)
            filtered = chunk[mask].copy()
            if filtered.empty: continue
            
            filtered['amount'] = np.log1p(filtered['amount'])
            for row in filtered.itertuples():
                u, v, w, f = row.nameOrig, row.nameDest, row.amount, row.isFraud
                if self.G_global.has_edge(u, v):
                    self.G_global[u][v]['weight'] += w
                    if f == 1: self.G_global.nodes[u]['isFraud'] = self.G_global.nodes[v]['isFraud'] = 1
                else:
                    self.G_global.add_edge(u, v, weight=w)
                    self.G_global.nodes[u]['isFraud'] = max(f, self.G_global.nodes.get(u, {}).get('isFraud', 0))
                    self.G_global.nodes[v]['isFraud'] = max(f, self.G_global.nodes.get(v, {}).get('isFraud', 0))
                    
        self.finalize_and_sample()

# ==========================================
# 2. ELLIPTIC BITCOIN DATASET
# ==========================================
class EllipticDataset(BaseGraphDataset):
    def __init__(self, edges_file, classes_file):
        super().__init__()
        print("Caricamento Elliptic Bitcoin Dataset...")
        
        df_classes = pd.read_csv(classes_file)
        df_classes = df_classes[df_classes['class'] != 'unknown']
        df_classes['class'] = df_classes['class'].map({'1': 1, '2': 0})
        label_dict = dict(zip(df_classes['txId'], df_classes['class']))
        
        df_edges = pd.read_csv(edges_file)
        self.G_global.add_edges_from(zip(df_edges['txId1'], df_edges['txId2']))
        
        for node in self.G_global.nodes():
            self.G_global.nodes[node]['isFraud'] = label_dict.get(node, -1)
            
        self.finalize_and_sample()

# ==========================================
# 3. IBM AMLSIM DATASET
# ==========================================
class AMLSimDataset(BaseGraphDataset):
    def __init__(self, tx_file):
        super().__init__()
        print("Caricamento IBM AMLSim Dataset...")
        
        # AMLSim usa solitamente colonne come: sender_id, receiver_id, amount, is_sar (Suspicious Activity Report)
        df = pd.read_csv(tx_file, usecols=['sender_id', 'receiver_id', 'amount', 'is_sar'])
        df['amount'] = np.log1p(df['amount'])
        
        for _, row in df.iterrows():
            u, v, w, f = row['sender_id'], row['receiver_id'], row['amount'], row['is_sar']
            if self.G_global.has_edge(u, v):
                self.G_global[u][v]['weight'] += w
            else:
                self.G_global.add_edge(u, v, weight=w)
            
            self.G_global.nodes[u]['isFraud'] = max(f, self.G_global.nodes.get(u, {}).get('isFraud', 0))
            self.G_global.nodes[v]['isFraud'] = max(f, self.G_global.nodes.get(v, {}).get('isFraud', 0))
            
        self.finalize_and_sample()

# ==========================================
# 4. ETHEREUM PHISHING DATASET
# ==========================================
class EthereumPhishingDataset(BaseGraphDataset):
    def __init__(self, edges_file, nodes_file):
        super().__init__()
        print("Caricamento Ethereum Phishing Dataset...")
        
        # File dei nodi solitamente contiene 'address' e 'is_phishing' (1/0)
        df_nodes = pd.read_csv(nodes_file)
        label_dict = dict(zip(df_nodes['address'], df_nodes['is_phishing']))
        
        # File degli archi solitamente contiene 'from_address', 'to_address', 'value'
        df_edges = pd.read_csv(edges_file)
        
        for _, row in df_edges.iterrows():
            u, v, w = row['from_address'], row['to_address'], row.get('value', 1.0)
            self.G_global.add_edge(u, v, weight=w)
            
        for node in self.G_global.nodes():
            self.G_global.nodes[node]['isFraud'] = label_dict.get(node, 0)
            
        self.finalize_and_sample()

def get_dataset(dataset_name, **kwargs):
    dataset_name = dataset_name.lower()
    if dataset_name == "paysim":
        return PaySimDataset(kwargs['csv_file'], kwargs.get('pool_file', "pool_nodes.pkl"))
    elif dataset_name == "elliptic":
        return EllipticDataset(kwargs['edges_file'], kwargs['classes_file'])
    elif dataset_name == "amlsim":
        return AMLSimDataset(kwargs['tx_file'])
    elif dataset_name == "ethereum":
        return EthereumPhishingDataset(kwargs['edges_file'], kwargs['nodes_file'])
    else:
        raise ValueError(f"Dataset '{dataset_name}' non supportato. Scegli tra: paysim, elliptic, amlsim, ethereum.")