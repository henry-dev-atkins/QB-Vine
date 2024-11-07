import numpy as np
from typing import Union
import

from AllDimRecursion import AllDimRecursion

class QBV:
    """
    A wrapper using the Quasi-Bayes Vine model.
    """
    def __init__(self)->None:
        pass


    def fit(self, input_x: Union[np.ndarray, spark., input_y:np.ndarray)->None:
        """
        A method to fit the model to the data.
        """
        _complete_data = np.concatenate((input_x, input_y), axis=1)
        # Something like the below.
        _all_dim_recursion = AllDimRecursion(_complete_data)


    def predict(self, input_x: np.ndarray)->np.ndarray:
        """
        A method to predict the target variable.
        """
        pass