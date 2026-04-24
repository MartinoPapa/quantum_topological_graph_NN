# model.py
import torch
import torch.nn as nn
import pennylane as qml
from config import *

if USE_IBMQ:
    from qiskit_ibm_provider import IBMProvider
    dev = qml.device('qiskit.ibmq', wires=TOTAL_QUBITS, backend=IBM_BACKEND)
else:
    dev = qml.device("default.mixed", wires=TOTAL_QUBITS)

class MemoryBank:
    def __init__(self, capacity, feature_dim):
        self.capacity = capacity
        self.bank = torch.empty(0, feature_dim) 
        
    def update(self, new_features):
        if new_features.shape[0] == 0: return
        new_entries = new_features.detach().cpu()
        max_update = max(1, int(self.capacity * BANK_PERCENTAGE_UPDATE))
        if new_entries.shape[0] > max_update:
            indices = torch.randperm(new_entries.shape[0])[:max_update]
            entries_to_add = new_entries[indices]
        else:
            entries_to_add = new_entries
        combined = torch.cat([self.bank, entries_to_add], dim=0)
        if combined.shape[0] > self.capacity:
            self.bank = combined[-self.capacity:]
        else:
            self.bank = combined

    def get_nearest_distance(self, features):
        if self.bank.shape[0] == 0:
            return torch.zeros(features.shape[0], device=features.device)
        bank_device = self.bank.to(features.device)
        dists = torch.cdist(features, bank_device, p=2).pow(2)
        min_dists, _ = torch.min(dists, dim=1)
        return min_dists

@qml.qnode(dev, interface="torch")
def qtgnn_circuit(inputs, theta_e, theta_h, theta_p, layers):
    # Encoding quantistico dei dati del grafo
    qml.AmplitudeEmbedding(features=inputs, wires=range(TOTAL_QUBITS), normalize=True)
    
    # Entanglement Seeding (GAES)
    for i in range(NUM_QUBITS):
        qml.IsingXX(theta_e[i], wires=[i, i + NUM_QUBITS])
    
    sys_wires = range(NUM_QUBITS)
    
    # Variational Quantum Graph Convolution (VQGC)
    for l in range(layers):
        for i in sys_wires:
            qml.RZ(theta_h[l, 0, i], wires=i)
        
        cnt = 0
        for i in range(NUM_QUBITS):
            for j in range(i + 1, NUM_QUBITS):
                qml.IsingZZ(theta_h[l, 1, cnt], wires=[i, j])
                cnt = (cnt + 1) % theta_h.shape[-1]
        
        # Dinamiche non-lineari tramite Phase Damping
        t0 = theta_p[l, 0]
        t1 = theta_p[l, 1]
        denom = torch.exp(t0) + torch.exp(t1)
        p1 = torch.exp(t1) / denom 
        
        for i in sys_wires:
            qml.PhaseDamping(p1, wires=i)
            
    # Restituisce la matrice di densità ridotta (per simulatore)
    return qml.density_matrix(wires=sys_wires)

class QTGNN(nn.Module):
    def __init__(self, num_layers=3):
        super().__init__()
        self.n_qubits = NUM_QUBITS
        self.layers = num_layers
        self.theta_e = nn.Parameter(torch.randn(self.n_qubits) * 0.1)
        max_params = self.n_qubits * (self.n_qubits - 1) // 2
        self.theta_h = nn.Parameter(torch.randn(num_layers, 2, max_params) * 0.1)
        self.theta_p = nn.Parameter(torch.tensor([[THETA_0_INIT, THETA_1_INIT]] * num_layers))
        
        # Dimensione features: 2 (Quantistiche) + (N_BETTI+1)*3 (Topologiche)
        self.feat_dim = 2 + (N_BETTI + 1) * 3 
        self.classifier = nn.Sequential(
            nn.Linear(self.feat_dim, 1)
        )
        self.memory_bank = MemoryBank(K_MEMORY_BANK, self.feat_dim)

    def get_quantum_features(self, rho):
        # Calcolo Entropia e Purezza (Quantum Features)
        rho = (rho + rho.conj().transpose(-1, -2)) / 2.0
        z_q = torch.real(torch.trace(rho @ rho))
        try:
            _, evals, _ = torch.linalg.svd(rho)
        except:
            rho_jitter = rho + 1e-7 * torch.eye(rho.shape[-1], device=rho.device)
            _, evals, _ = torch.linalg.svd(rho_jitter)
        evals = torch.clamp(evals, min=1e-9)
        evals = evals / torch.sum(evals)
        c_q = -torch.sum(evals * torch.log(evals))
        return torch.stack([z_q, c_q]).float()

    def forward(self, inputs):
        rho = qtgnn_circuit(inputs, self.theta_e, self.theta_h, self.theta_p, self.layers)
        q_feats = self.get_quantum_features(rho)
        return rho, q_feats