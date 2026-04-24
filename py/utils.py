import sys
import torch
import numpy as np
import networkx as nx

class Logger(object):
    def __init__(self, filename):
        self.terminal = sys.stdout
        self.log = open(filename, "w")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
        self.log.flush()

    def flush(self):
        self.terminal.flush()
        self.log.flush()

def compute_topological_features(rho_np, epsilon_steps=3):
    N = rho_np.shape[0]
    dists = np.zeros((N, N))
    rho_abs = np.abs(rho_np)
    
    for i in range(N):
        for j in range(i+1, N):
            denom = np.sqrt(rho_abs[i,i] * rho_abs[j,j]) + 1e-9
            fid = rho_abs[i,j] / denom
            d = np.sqrt(max(0, 1 - fid**2))
            dists[i,j] = dists[j,i] = d
            
    features = []
    epsilons = np.linspace(0.1, 0.9, epsilon_steps)
    
    # 2. Estrazione degli invarianti topologici tramite complessi di Vietoris-Rips
    for eps in epsilons:
        adj = (dists <= eps).astype(int)
        np.fill_diagonal(adj, 0)
        G = nx.from_numpy_array(adj)
        
        b0 = nx.number_connected_components(G)
        b1 = len(nx.cycle_basis(G))
        b2 = sum(1 for _ in nx.enumerate_all_cliques(G) if len(_) == 3)
        chi = b0 - b1 + b2
        
        features.extend([b0/N, b1/N, b2/N, chi/N])
        
    return torch.tensor(features, dtype=torch.float32)