from .utils import *
from .GaussianCop import *
import torch
from sklearn.model_selection import train_test_split


class MarRecur:
    '''
    Class for univariate Quasi-Bayesian recursion.
    '''

    def __init__(self):
        pass

    def train_test_permute(self, data, test_size=0.5, random_seed=3, perm_nb=10):
        '''
        Standardize the data, split it into train and test sets and create permutations of the train set.

        Inputs
        ---
        data: the data to be split, torch.tensor\n
        test_size: the proportion of the data to be used as test set, float - optional, set at 0.2\n
        random_seed: the random seed for reproducibility, int - optional, set at 3\n
        perm_nb: the number of permutations to create, int - optional, set at 10
        '''

        n = len(data)
        n_test = int(np.ceil(n * test_size))
        n_train = n - n_test
        train_idx, test_idx = train_test_split(np.arange(n),
                                               test_size=test_size,
                                               train_size=n_train,
                                               random_state=random_seed)
        train_data = data[train_idx].clone().detach().float()
        mean_train = train_data.mean(dim=0)
        std_train = train_data.std(dim=0)
        train_data = (train_data - mean_train) / std_train

        train_data_permutations = create_permutatons(obs=train_data,
                                                     perms=perm_nb)

        test_data = data[test_idx].clone().detach().float()
        test_data = (test_data - mean_train) / std_train

        return train_data_permutations, test_data

    # What is needed for training, as a single itteration of optimisation for a given rho value:

    # For each perm:
    #   1. Get the conditioning values CDF^n(x^n) @step_n for chosen rho, with get_CDFn_on_trainset.
    #   2. Get the grid CDF for chosen rho, with get_CDF_on_grid_single_perm
    # Combining permutation:
    # -> Get the energy score for chosen rho, with escore_over_avgperms

    def get_CDFn_on_trainset(self, observations, rho, init_dist='Cauchy', a=1.):
        '''
        Computes CDF^n(x^n), where n is the itteration of the recursion, for a given permutation of a 1D marginal array of observations given a single rho value.
        These are needed as conditioning values in the conditional copula for the R-BP recursion.

        Inputs
        ---
        observations: on permutation of data for a single dimension, shape (num_data,)\n
        rho: correlation coefficient, float\n
        init_dist: initialization distribution, string ('Normal', 'Cauchy', 'Lomax')\n
        a: shape parameter for lomax distribution, scalar - optional

        Output
        ---
        Cdf values for each observation, shape (num_data,)
        '''
        flt = 1e-6
        num_data = observations.shape[0]
        cdf_n_vals = torch.zeros([num_data])

        # Initialize the CDF values
        if init_dist == 'Normal':
            cdf = DistributionFunctions.cdf_std_normal(observations).reshape(num_data)
        elif init_dist == 'Cauchy':
            cdf = torch.distributions.cauchy.Cauchy(loc=0.0, scale=1.0).cdf(observations).reshape(num_data)
        elif init_dist == 'Lomax':
            cdf = cdf_lomax(observations, a)
        # Clip the CDF values to avoid nans
        cdf = torch.clip(cdf, min=flt, max=1. + flt)
        cdf_n_vals[0] = cdf[0]

        # Compute the CDF^n(x^n) values for each step n of the recursion
        for k in range(1, num_data):
            Cop = DistributionFunctions.cGC_distribution(rho=rho, u=cdf[1:], v=cdf[0]).reshape(num_data - k)
            cdf = (1 - alpha(k)) * cdf[1:] + alpha(k) * Cop
            cdf = torch.clip(cdf, min=flt, max=1. + flt)
            cdf_n_vals[k] = cdf[0]
        return cdf_n_vals

    def get_CDF_on_grid_single_perm(self, grid_size, cdf_traindata_oneperm, current_rho, observed_data, extrap_tail=.1,
                                    init_dist='Cauchy', a=1.):
        '''
        Intermediate function that gets CDF values at values of a grid, for a single dimension, single permutation of observations and a single rho.
        These are needed to numerically invert the CDF for the R-BP recursion to sample from it.

        Inputs
        ---

        grid_size: int, the number of grid points to evaluate the CDF at.\n
        cdf_traindata_oneperm: a 1d array of cdf values for one permutation with a particular rho. of shape (num_data,)\n
        current_rho: float, the rho value to optimize.\n
        observed_data: torch.tensor, the data for the dimension. Used to get extrapolation values for the inverse CDF.\n
        extrap_tail: float, optional, the amount to extrapolate the CDF by. -set at 0.1\n
        init_dist: string, optional, the initialization distribution, either 'Normal', 'Cauchy', or 'Lomax'. -set at Cauchy\n
        a: float, optional, the shape parameter for the Lomax distribution, if used. - set as 1.

        Outputs
        ---
        xgrids: torch.tensor, the grid of points. of shape (grid_size,)\n
        cdf_of_grid: torch.tensor, the CDF values at the grid points. of shape (grid_size,)
        '''

        flt = 1e-6

        num_data = cdf_traindata_oneperm.shape[0]
        min = torch.min(observed_data) - extrap_tail
        max = torch.max(observed_data) + extrap_tail
        xgrids = torch.linspace(min, max, grid_size)

        # Initialize the CDF values
        if init_dist == 'Normal':
            cdf = torch.distributions.normal.Normal(loc=0, scale=1).cdf(xgrids).reshape(grid_size)
        if init_dist == 'Cauchy':
            cdf = torch.distributions.cauchy.Cauchy(loc=0.0, scale=1.0).cdf(xgrids).reshape(grid_size)
        if init_dist == 'Lomax':
            cdf = cdf_lomax(xgrids, a)
        # Clip the CDF values to avoid nans
        cdf = torch.clip(cdf, min=flt, max=1. + flt)

        # recursion for p_i i>0
        for k in range(0, num_data):
            Cop = DistributionFunctions.cGC_distribution(rho=current_rho,
                                                         u=cdf,
                                                         v=cdf_traindata_oneperm[k]).reshape(grid_size)
            cdf = (1 - alpha(k + 1)) * cdf + alpha(k + 1) * Cop
            cdf = torch.clip(cdf, min=flt, max=1. + flt)
        cdf_of_grid = cdf

        return xgrids, cdf_of_grid

    def escore_over_avgperms(self, num_samples, observed_data, grid, cdf_grid_listperms, crps='integral'):
        '''
        Computes the energy score for a single rho value.
        This is done by averaging the cdf values at grid points to get a linear interpolation of the inverse cdf P^-1.
        Then doing inverse CDF sampling throught it.

        Inputs
        ---
        num_samples: int, the number of samples used to average the energy score expectation.\n
        observed_data: torch.tensor, the observed data, in any order. of shape (num_data,)\n
        grid: torch.tensor, the grid of points. of shape (grid_size,)\n
        cdf_grid_listperms: list of torch.tensors, the cdf values at grid points for each permutation. [(grid_size,),...]

        Outputs
        ---
        Escore: float, the energy score for the rho value.
        '''
        # Average the cdf values over permutations
        avg_cdf_overperms = torch.zeros(grid.shape[0])
        for i in range(len(cdf_grid_listperms)):
            avg_cdf_overperms += cdf_grid_listperms[i]
        avg_cdf_overperms = avg_cdf_overperms / len(cdf_grid_listperms)
        # Linear interpolation of the inverse cdf and inverse CDF sampling

        if crps == 'integral':
            crps_score = crps_integral(grid, avg_cdf_overperms, observed_data)
            return crps_score, avg_cdf_overperms

        else:
            avg_cdf_overperms[torch.argmin(avg_cdf_overperms)] = 0.
            avg_cdf_overperms[torch.argmax(avg_cdf_overperms)] = 1.
            uniform_samples = torch.rand(num_samples)
            inv = xi.Interp1D(avg_cdf_overperms, grid, method='linear')
            inv_cdf_samples = inv(uniform_samples)
            crps_score = crps_samples(inv_cdf_samples, observed_data)

            return crps_score, inv_cdf_samples, uniform_samples, avg_cdf_overperms

    def eval_PDFandCDF_on_test_single_perm(self, test_data, cdf_traindata_oneperm, current_rho, init_dist='Cauchy',
                                           a=1.):
        '''
        Evaluates CDF and PDF values at tets points, for a single dimension, single permutation of observations and a fixed rho.

        Inputs
        ---
        test_data: torch.tensor, the test data for the dimension. of shape (num_test_data,)\n
        cdf_traindata_oneperm: a 1d array of cdf values for one permutation with a particular rho. of shape (num_data,)\n
        current_rho: float, the rho value to use.\n
        init_dist: string, optional, the initialization distribution, either 'Normal', 'Cauchy', or 'Lomax'. -set at Cauchy\n
        a: float, optional, the shape parameter for the Lomax distribution, if used. - set as 1.

        Outputs
        ---
        test_pdf: torch.tensor, the PDF values at the test points. of shape (num_test_data,)\n
        test_cdf: torch.tensor, the CDF values at the test points. of shape (num_test_data,)
        '''

        flt = 1e-6

        num_data_train = cdf_traindata_oneperm.shape[0]
        num_data_test = test_data.shape[0]

        # Initialize the PDF and CDF values
        if init_dist == 'Normal':
            cdf = torch.distributions.normal.Normal(loc=0, scale=1).cdf(test_data).reshape(num_data_test)
            pdf = DistributionFunctions.pdf_std_normal(test_data).reshape(num_data)
        if init_dist == 'Cauchy':
            cdf = torch.distributions.cauchy.Cauchy(loc=0.0, scale=1.0).cdf(test_data).reshape(num_data_test)
            pdf = torch.distributions.cauchy.Cauchy(loc=0.0, scale=1.0).log_prob(test_data).exp().reshape(num_data_test)
        if init_dist == 'Lomax':
            cdf = cdf_lomax(test_data, a)
            pdf = pdf_lomax(test_data, a)

        # Clip the CDF values to avoid nans
        cdf = torch.clip(cdf, min=flt, max=1. + flt)

        # recursion for p_i i>0
        for k in range(0, num_data_train):
            cop = DistributionFunctions.GC_density(rho=current_rho,
                                                   u=cdf,
                                                   v=cdf_traindata_oneperm[k]).reshape(num_data_test)
            Cop = DistributionFunctions.cGC_distribution(rho=current_rho,
                                                         u=cdf,
                                                         v=cdf_traindata_oneperm[k]).reshape(num_data_test)
            cdf = (1 - alpha(k + 1)) * cdf + alpha(k + 1) * Cop
            cdf = torch.clip(cdf, min=flt, max=1. + flt)
            pdf = (1 - alpha(k + 1)) * pdf + alpha(k + 1) * cop * pdf

        return pdf, cdf

    # What I am not sure is necessary
    def pdf_cdf_single_dim(self, observations_oneperm, rho, init_dist='Normal', a=1.):
        '''
        run one predictive recursion in 1D given a single rho value and a single permutation.
        Returns final avergare log-density and cdf.

        Input
        ---
        observations_oneperm: one permutation of the data, shape (num_data,)
        rho: correlation coefficient, scalar
        init_dist: initialization distribution, string ('Normal', 'Cauchy', 'Lomax')
        a: shape parameter for lomax distribution, scalar - optional

        Output
        ---
        Summed log predictive score (negative log likelihood equivalent) over observations_oneperm
        '''
        flt = 1e-6
        num_data = observations_oneperm.shape[0]
        pl = torch.zeros([1])

        if init_dist == 'Normal':
            cdf = DistributionFunctions.cdf_std_normal(observations_oneperm).reshape(num_data)
            pdf = DistributionFunctions.pdf_std_normal(observations_oneperm).reshape(num_data)
        elif init_dist == 'Cauchy':
            cdf = torch.distributions.cauchy.Cauchy(loc=0.0, scale=1.0).cdf(observations_oneperm).reshape(num_data)
            pdf = torch.distributions.cauchy.Cauchy(loc=0.0, scale=1.0).log_prob(observations_oneperm).exp().reshape(
                num_data)
        elif init_dist == 'Lomax':
            cdf = cdf_lomax(observations_oneperm, a)
            pdf = pdf_lomax(observations_oneperm, a)

        for k in range(1, num_data):
            Cop = DistributionFunctions.cGC_distribution(rho=rho, u=cdf[1:], v=cdf[0]).reshape(num_data - k)
            cop = DistributionFunctions.GC_density(rho=rho, u=cdf[1:], v=cdf[0]).reshape(num_data - k)
            cdf = (1 - alpha(k)) * cdf[1:] + alpha(k) * Cop
            cdf = torch.clip(cdf, min=flt, max=1. + flt)
            pdf = (1 - alpha(k)) * pdf[1:] + alpha(k) * cop * pdf[1:]
            pl = pl + torch.log(pdf[0])

        return pl, cdf

    def evaluate_prcopula_oneperm(self, test_points, cdf_obs_oneperm, rho, init_dist='Normal', a=1.):
        '''
        evaluate pdfs and cdfs on a single permutation of data for a single rho value.

        Input
        ---
        test_points: test points, shape (num_evals,)
        cdf_obs_oneperm: conditional CDF of observations for a single permutation, shape (num_data,)
        rho: correlation coefficient, scalar torch.tensor
        init_dist: initialization distribution, string ('Normal', 'Cauchy', 'Lomax') - optional
        a: shape parameter for Lomax distribution, scalar - optional

        Output
        ---
        pdfs and cdfs at test points, given the permutation of cdf_obs_oneperm, shape (num_evals,)
        '''
        flt = 1e-6
        num_evals = test_points.shape[0]
        num_data = cdf_obs_oneperm.shape[0]

        if init_dist == 'Normal':
            pdf = DistributionFunctions.pdf_std_normal(test_points).reshape(num_evals)
            cdf = DistributionFunctions.cdf_std_normal(test_points).reshape(num_evals)
        elif init_dist == 'Cauchy':
            pdf = torch.distributions.cauchy.Cauchy(loc=0.0, scale=1.0).log_prob(test_points).exp().reshape(num_evals)
            cdf = torch.distributions.cauchy.Cauchy(loc=0.0, scale=1.0).cdf(test_points).reshape(num_evals)
        elif init_dist == 'Lomax':
            pdf = pdf_lomax(test_points, a)
            cdf = cdf_lomax(test_points, a)

        cdf = torch.clip(cdf, min=flt, max=1. + flt)

        for k in range(num_data):
            cop = DistributionFunctions.GC_density(rho=rho, u=cdf, v=cdf_obs_oneperm[k]).reshape(num_evals)
            Cop = DistributionFunctions.cGC_distribution(rho=rho, u=cdf, v=cdf_obs_oneperm[k]).reshape(num_evals)
            cdf = (1 - alpha(k + 1)) * cdf + alpha(k + 1) * Cop
            cdf = torch.clip(cdf, min=flt, max=1. + flt)
            pdf = (1 - alpha(k + 1)) * pdf + alpha(k + 1) * cop * pdf

        return pdf, cdf


# Example usage:
if __name__ == '__main__':
    mar_recur = MarRecur()

    # Example data and parameters
    observations = torch.randn(10)
    rho = 0.5
    init_dist = 'Cauchy'
    a = 1.0

    # Training example
    pl = mar_recur.train_rho(observations, rho, init_dist, a)
    print("Training Log Likelihood:", pl)

    # Context extraction example
    context = mar_recur.get_context(observations, rho, init_dist, a)
    print("Context:", context)

    # Evaluation example
    test_points = torch.randn(10)
    dens, cdfs = mar_recur.evaluate_prcopula(test_points, context, rho, init_dist, a)
    print("Densities:", dens)
    print("CDFs:", cdfs)