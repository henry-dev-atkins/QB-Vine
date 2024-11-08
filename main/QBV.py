import numpy as np
import torch
from main.AllDimRecursion import BP_all_dim
from main.backends import BackendMPI


class QBV:
    """
    A wrapper using the Quasi-Bayes Vine model.
    """

    def __init__(self) -> None:
        self.model = None
        pass

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
        Fits the model to the data.
        """
        input_x, input_y = self._val_data(input_x, input_y)

        # Make input_y a column in input_x for the fit
        input_y = input_y.view(-1, 1)
        _data = torch.cat([input_x, input_y], dim=1)  # Concatenate along columns

        train_data, test_data = self._get_test_train(_data=_data, test_pct=test_pct)

        backend = BackendMPI()
        bp_all_dim = BP_all_dim(backend, train_data, test_data)
        results = bp_all_dim.calculate(n_dim=_data.shape[1])
        return results


    def predict(self, input_x: torch.Tensor) -> torch.Tensor:
        """
        A method to predict the target variable.
        """
        # Placeholder for actual prediction logic
        pass
