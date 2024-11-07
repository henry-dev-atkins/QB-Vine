import torch
from main.AllDimRecursion import BP_all_dim
from main.backends import *

# I have Implemented what I *think* is the correct way to use the codebase, 
# although it is untested due to import/install issues.


def load_data():
    from sklearn.datasets import load_wine
    wine = load_wine()
    data = wine.data
    return data

def train_model(train_data, test_data):
    backend = mpi.get_mpi_manager()
    bp_all_dim = BP_all_dim(backend, train_data, test_data)
    results = bp_all_dim.calculate(n_dim=train_data.shape[1])
    return results

def evaluate_model(results):
    avg_pdfs, avg_cdfs, rho, dim_index, avg_cdfs_train = results
    print(f"Selected rho: {rho}")
    print(f"Average PDF: {avg_pdfs}")

if __name__ == "__main__":
    data = load_data()
    train_data = torch.tensor(data).unsqueeze(0)
    test_data = torch.tensor(data[0]).unsqueeze(0)
    results = train_model(train_data, test_data)
    evaluate_model(results)
