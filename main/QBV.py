import numpy as np
import torch

from main.GaussianCop import DistributionFunctions
from main.AllDimRecursion import BP_all_dim
from main.backends import BackendMPI


class QBV:
    """
    A wrapper using the Quasi-Bayes Vine model.
    """
    def __init__(self) -> None:
        self.marginals = []
        self.copula = None
        self.copula_bandwidth = 3.0
        self.means = None
        self.stds = None

    @staticmethod
    def _val_tensor(data) -> torch.Tensor:
        """
        Converts input to Torch Tensor if necessary.
        """
        if isinstance(data, torch.Tensor):
            return data
        elif isinstance(data, np.ndarray):
            return torch.from_numpy(data)
        else:
            raise TypeError("Input data must be a NumPy array or a Torch tensor")


    def _val_data(self, input_x, input_y) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Validates and converts input data to Torch Tensors.
        """
        input_x = self._val_tensor(data=input_x)
        input_y = self._val_tensor(data=input_y)

        if input_x.ndimension() != 2:
            raise ValueError("input_x must be a 2D tensor (matrix)")
        if input_y.ndimension() != 1:
            raise ValueError(f"input_y must be a 1D tensor (vector), but is {input_y.shape}")
        if input_x.shape[0] != input_y.shape[0]:
            raise ValueError("input_x and input_y must have the same number of rows")

        return input_x, input_y


    @staticmethod
    def _get_test_train(_data: torch.Tensor, test_pct: float = 0.2) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Splits the data into test and train sets using Torch Tensors.
        """
        _train_rows = int(len(_data) * (1 - test_pct))
        train_data = _data[:_train_rows, :]
        test_data = _data[_train_rows:, :]
        return train_data, test_data

    def fit(self, input_x, input_y, test_pct: float = 0.2) -> list:
        """
        Fits the model to the data with preprocessing and stores fitted parameters.
        """
        input_x, input_y = self._val_data(input_x, input_y)

        # Make input_y a column in input_x for the fit
        input_y = input_y.view(-1, 1)
        _data = torch.cat([input_x, input_y], dim=1)  # Concatenate along columns
        self.means = _data.mean(dim=0)
        self.stds = _data.std(dim=0)

        train_data, test_data = self._get_test_train(_data=_data, test_pct=test_pct)

        backend = BackendMPI()
        bp_all_dim = BP_all_dim(backend, train_data, test_data)
        print(f"_data shape: {_data.shape}, n_dim: {_data.shape[1]}")

        results: list = bp_all_dim.calculate(n_dim=_data.shape[1])
        transposed_results = list(zip(*results))

        for idx, (pdf, cdf, theta, dim_index, avg_cdfs_train) in enumerate(transposed_results):
            self.marginals.append({
                'pdf': pdf,
                'cdf': cdf,
                'theta': theta,  # Store optimized correlation parameter
                'dim_index': dim_index,
                'avg_cdfs_train': avg_cdfs_train
            })

        # TODO: Wtf is this?
        self.copula_bandwidth = None
        return results

    def predict(self, input_x):
        """
        Predicts the joint density for the input data.
        Normalizes and ensures data compatibility with marginals.
        """
        input_x = self._val_tensor(data=input_x)
        input_x = (input_x - self.means[:-1]) / self.stds[:-1]
        samples, dims = input_x.shape

        fitted_dims = [marginal['dim_index'] for marginal in self.marginals]

        # Ensure consistency between fitted marginals and input dimensions
        if dims in fitted_dims:
            fitted_dims.remove(dims)

        input_x = input_x[:, fitted_dims]

        joint_densities = []

        for s in range(samples):
            marginal_cdfs = []
            marginal_pdfs = []

            for idx, dim in enumerate(fitted_dims):
                cdf = self.marginals[idx]['cdf']
                pdf = self.marginals[idx]['pdf']

                index = torch.round(input_x[s, idx]).long()
                marginal_cdfs.append(torch.tensor(cdf[index], dtype=torch.float32))
                marginal_pdfs.append(torch.tensor(pdf[index], dtype=torch.float32))

            # Convert to torch Tensors for copula handling
            marginal_cdfs = torch.tensor(marginal_cdfs)
            marginal_pdfs = torch.tensor(marginal_pdfs)

            copula_density = 1.0
            for i in range(marginal_cdfs.shape[0] - 1):  # Iterate across pairs of dimensions
                rho = torch.tensor([self.marginals[i]['theta']])
                u = marginal_cdfs[i]
                v = marginal_cdfs[i + 1]

                pair_density = DistributionFunctions.GC_density(rho=rho, u=u.unsqueeze(0), v=v.unsqueeze(0))
                copula_density *= pair_density

            joint_density = copula_density * torch.prod(marginal_pdfs)
            joint_densities.append(joint_density)

        return (torch.tensor(joint_densities) + self.means[-1]) * self.stds[-1]