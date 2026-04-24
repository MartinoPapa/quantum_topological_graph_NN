## Datasets
### REAL
- The Elliptic Bitcoin Dataset (https://www.kaggle.com/datasets/ellipticco/elliptic-data-set)
real-world dataset mapping Bitcoin transactions, where nodes are transactions and edges are the flow of Bitcoin. It includes labeled illicit transactions (e.g., ransomware, darknet markets). Because illicit actors actively try to obfuscate their trails using mixers and cyclic transfers, the graph contains the exact rich topological structures.

! This dataset is different from the others because it maps the transactions as nodes and edges as flow of Bitcoins.

! edges and nodes already defined

- Ethereum Phishing Transaction Networks (https://www.kaggle.com/datasets/xblock/ethereum-phishing-transaction-network)
accounts and transactions in Ethereum are treated as nodes and edges, thus detection of phishing accounts can be modeled as a node classification problem.

! edges and nodes already defined

### SYNTHETIC:
- IBM AMLSim (https://github.com/IBM/AMLSim/blob/master/README.md)
everyday synthetic transactions are specific "alert patterns" and anomalous subgraphs that mimic actual money laundering typologies (like structuring, smurfing, or cyclical transactions).
- PaySim, already tested, has the problem of not having loops or complex structures

## Implementation details
- we will train the model on a simulator. Only the testing will be performed on an actual quantum computer to reduce costs

## TODO
- understand the concept behind the bitcoin dataset
- explain the economic reasons behind the idea of looking for loops and complex structures (expecially in the bitcoin dataset which is mapped differently)
- decide the appropiate number of qubits