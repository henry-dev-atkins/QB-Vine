import time
import torch
import logging
from sklearn.datasets import load_wine

from main.AllDimRecursion import BP_all_dim
from main.backends import BackendMPI
from main.QBV import QBV



logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

def evaluate_model(results):
    avg_pdfs, avg_cdfs, rho, dim_index, avg_cdfs_train = results
    print(f"Selected rho: {rho}")
    print(f"Average PDF: {avg_pdfs}")

if __name__ == "__main__":
    from sklearn.preprocessing import MinMaxScaler
    from sklearn.metrics import mean_squared_error
    data = MinMaxScaler().fit_transform(load_wine().data)

    X = torch.tensor(data[:, :-1])
    y = torch.tensor(data[:, -1])

    QBV = QBV()
    results = QBV.fit(X, y)
    preds = QBV.predict(X)
    print(f"Mean Squared Error: {mean_squared_error(y, preds)}")

