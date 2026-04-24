import numpy as np
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(os.path.dirname(BASE_DIR), 'datasets')

USE_IBMQ_TRAIN = False #using true for training on ibmq would require too much time and money, so we train on a simulator and then test on ibmq
USE_IBMQ_TEST = False
IBM_BACKEND = "ibm_brisbane" 

TRAINING_DIM = 700 
VALIDATION_DIM = 10 
TEST_DIM = 100 
FRAUD_RATIO = 0.3 
SAMPLED_EDGES = 8 

TOTAL_GRAPHS = TRAINING_DIM + VALIDATION_DIM + TEST_DIM
NUM_NODES = 16                     
NUM_QUBITS = int(np.log2(NUM_NODES)) 
TOTAL_QUBITS = 2 * NUM_QUBITS      

BATCH_SIZE = 16                    
NUM_LAYERS = 3                     
N_BETTI = 1                        
LATENT_DIM = 2 + (N_BETTI + 1) * 3 

K_MEMORY_BANK = 10             
BANK_PERCENTAGE_UPDATE = 0.1       
WARMUP_STEPS = 5                   

EPOCHS = 20                      
LEARNING_RATE = 0.005           

LAMBDA_1 = 0.1 
LAMBDA_2 = 0.01 

THETA_0_INIT = 1.0 
THETA_1_INIT = -5.0   

TAU_THRESHOLD = 0.4