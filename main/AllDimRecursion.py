from GaussianCop import *
from OneRecursion import *
from utils import *
import pandas as pd
import torch
from tqdm import tqdm
import numpy as np
import torch.optim as optim
import time

from abc import ABCMeta, abstractmethod, abstractproperty

class Method(metaclass = ABCMeta):
    """
        This abstract base class represents an inference method.

    """

    def __getstate__(self):
        """Cloudpickle is used with the MPIBackend. This function ensures that the backend itself
        is not pickled
        """
        state = self.__dict__.copy()
        del state['backend']
        return state



class BP_all_dim(Method):
    #n_samples = None
    def __init__(self, backend, train_data_all_dim, test_data_all_dim):
        self.backend = backend
        self.train_data_all_dim = train_data_all_dim
        self.test_data_all_dim = test_data_all_dim

        self.train_data_all_dim_bds = self.backend.broadcast(train_data_all_dim)
        self.test_data_all_dim_bds = self.backend.broadcast(test_data_all_dim)

    def calculate(self, n_dim):
        self.n_dim = n_dim

        index_arr = range( self.n_dim)
        dim_index_pds = self.backend.parallelize(index_arr)

        avg_pdf_cdf_theta_dimidx_testgrid_traincdf = self.backend.map(self._BP_single_dim, dim_index_pds)
        avg_pdf_cdf_theta_dimidx_testgrid_traincdf_collected = self.backend.collect(avg_pdf_cdf_theta_dimidx_testgrid_traincdf)
        avg_pdf, avg_cdf, theta_hist, dimidx, grids = [list(t) for t in zip(*avg_pdf_cdf_theta_dimidx_testgrid_traincdf_collected)]

        #avg_pdf_cdf_theta = self.backend.collect(parameters_simulations_pds)
        #parameters, simulations = [list(t) for t in zip(*parameters_simulations)]

        return avg_pdf, avg_cdf, theta_hist, dimidx, grids

    # def _sub_calc(self, a):
    #     return pow(a, self.exponent) + 10

    def _BP_single_dim(self, dim_index):
        '''
        Optimizes the rho parameter for a single dimension of the data. Then computes the pdf and cdf on the test data.

        Inputs:
        traindata_single_dim_allperms_and_testdata: [train_data_allperms, test_data],
        where train_data_allperms is a tensor of shape (num_perms, num_samples) and test_data is a tensor of shape (num_samples,)\n
        max_iter: int, maximum number of iterations for the optimization, default is 50 - optional\n
        learning_rate: float, learning rate for the optimizer, default is 3 - optional\n
        plot_it: bool, if True, plots the pdf and cdf of the data and the test data, default is False - optional

        Outputs:
        out: list, [pdf, cdf, rho], where pdf and cdf are evaluated on the test data, and rho is the optimized parameter
        '''
        max_iter = 50
        learning_rate = 3
        train_data = self.train_data_all_dim[:, :, int(dim_index)]
        test_data = self.test_data_all_dim[:, int(dim_index)]

        train_data = self.train_data_all_dim_bds.value()[:, :, int(dim_index)]
        test_data = self.test_data_all_dim_bds.value()[:, int(dim_index)]

        start = time.time()  # to measure runtime

        rho = torch.tensor(1., requires_grad=True)  # imitiolize rho, is in reals, but sigmoid is applied on it in the model

        optimizer = optim.Adam([rho], lr=learning_rate)

        # to track the scores and rhos
        scores_hist = []
        theta_hist = []
        stop_counter = 0  # for early stopping in case of convergence

        for i in (range(max_iter)):
            optimizer.zero_grad()

            # over all perms
            conditional_cdf_train_list = []
            cdf_perms = []
            for perm in (range(train_data.shape[0])):
                conditional_cdf_train = MarRecur().get_CDFn_on_trainset(train_data[perm, :],
                                                                        torch.sigmoid(rho))  # gets u^n values
                conditional_cdf_train_list.append(conditional_cdf_train)

                grid, grid_cdf = MarRecur().get_CDF_on_grid_single_perm(grid_size=1000,
                                                                        cdf_traindata_oneperm=conditional_cdf_train,
                                                                        current_rho=torch.sigmoid(rho),
                                                                        observed_data=train_data[perm,:]) # gets cdf values on grid, to use in escore
                
                grid_cdf.requires_grad_(True)
                cdf_perms.append(grid_cdf)

            # escore avergred over all perms
            out = MarRecur().escore_over_avgperms(num_samples=100,
                                                  observed_data=train_data[perm, :],
                                                  grid=grid,
                                                  cdf_grid_listperms=cdf_perms,
                                                  crps='not integral')

            out[0].backward()
            optimizer.step()
            scores_hist.append(out[0].item())
            theta_hist.append(torch.sigmoid(rho).item())
            if i > 0:
                with torch.no_grad():

                    if torch.sigmoid(rho).item() - theta_hist[-2] < 1e-2:
                        stop_counter += 1

                if torch.sigmoid(rho).item() > 0.999 or stop_counter > 2:  # Optimisation finished
                    print('Dim=',dim_index,' converged|' ' evals:', 1 + i, '| selected rho:', theta_hist[-2], '| time taken:',
                          time.time() - start)
                    print('Reason:', '| rho too close to 1', torch.sigmoid(rho).item() > 0.999,
                          '| stop_counter>2 (diff between itterations<0.01) for 3 times:', stop_counter > 2)

                    # Compute the pdf and cdf on test data
                    pdfs = []
                    cdfs = []
                    test_grid = torch.cat([test_data, torch.linspace(train_data.min() - 1, train_data.max() + 1,
                                                                    10000)])  
                    for perm in (range(train_data.shape[0])):
                        conditional_cdf_train = conditional_cdf_train_list[perm]  # reuse the conditional cdf of theta[-2]
                        pdf, cdf = MarRecur().eval_PDFandCDF_on_test_single_perm(test_data=test_grid,
                                                                                 cdf_traindata_oneperm=conditional_cdf_train,
                                                                                 current_rho=torch.tensor(theta_hist[-2]))
                        pdfs.append(pdf)
                        cdfs.append(cdf)
                    # average the pdfs and cdfs over permutations
                    avg_pdfs = torch.stack(pdfs).mean(dim=0)
                    avg_cdfs = torch.stack(cdfs).mean(dim=0)

                    out = [avg_pdfs.detach().numpy(), avg_cdfs.detach().numpy(), theta_hist[-2],dim_index, test_grid.detach().numpy()]
                    print('Dim=',dim_index,' finished|','| selected rho:', theta_hist[-2], '| time taken:',
                          time.time() - start)
                    break

        return out
