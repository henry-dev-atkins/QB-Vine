import time

import torch
import logging
from sklearn.datasets import load_wine
from main.AllDimRecursion import BP_all_dim
from main.backends import BackendMPI

# Configure logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

def load_data():
    wine = load_wine()
    data = wine.data
    return data

def train_model(train_data, test_data):
    backend = BackendMPI()
    bp_all_dim = BP_all_dim(backend, train_data, test_data)
    print(f"example.train_model: Training data shape: {train_data.shape}")
    print(f"example.train_model: Training data sample: {train_data.shape}")
    results = bp_all_dim.calculate(n_dim=train_data.shape[1])
    return results

def evaluate_model(results):
    avg_pdfs, avg_cdfs, rho, dim_index, avg_cdfs_train = results
    print(f"Selected rho: {rho}")
    print(f"Average PDF: {avg_pdfs}")

if __name__ == "__main__":
    data = load_data()[:20, 0:3]
    train_data = torch.tensor(data)
    test_data = torch.tensor(data)
    results = train_model(train_data, test_data)
    evaluate_model(results)