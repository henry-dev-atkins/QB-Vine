import torch
import numpy as np
import pandas as pd

def cdf_lomax(x, a):
  '''
  calculate cdf of lomax distribution
  ---Input
  x: the input value
  a: the shape parameter
  ---Output
  cdf of lomax distribution
  '''
  return 1 - (1+x) ** (-a)

def pdf_lomax(x, a):
  '''
  calculate pdf of lomax distribution
  ---Input
  x: the input value
  a: the shape parameter
  ---Output
  pdf of lomax distribution
  '''
  return a * (1 + x) **(-a-1)

def alpha(step):
  '''
  alpha value derived by (Fong et al. 2021)
  --- Input 
  step: the step index
  ---Output
  squence weight
  '''
  i = step
  alpha = (2 - 1/i) * (1/(i+1))
  return torch.tensor(alpha, dtype = torch.float32)

def drop_corr(y,threshold= 0.98):
    '''
    Drop data with high correlation.
    ---Input
    y: data
    threshold: correlation threshold
    ---Output
    return: data without high correlation
    '''
    data = pd.DataFrame(y)
    # Create correlation matrix
    corr_matrix = data.corr().abs()

    # Select upper triangle of correlation matrix
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))

    # Find index of feature columns with correlation greater than 0.95
    to_drop = [col for col in upper.columns if any(upper[col] > threshold)]
    y = data.drop(columns = to_drop).values

    return(y)

def create_permutatons(obs, perms):
  '''
  Create permutations of data.
  ---Input
  obs: data
  perms: number of permutations
  ---Output
  perms permutations of data
  '''
  permutations = []
  L = obs.shape[0]

  for _ in range(perms):

    permutation = torch.randperm(L)
    sequence = obs[permutation, :]
    permutations.append(sequence)

  return torch.stack(permutations)



def Energy_Score_pytorch(beta, observations_y, simulations_Y):
        '''
        This version is basically the CRPS as the obs are 1D. Check that inputs are 1D if using this.
        '''
        n = len(observations_y)
        m = len(simulations_Y)

        # First part |Y-y|. Gives the L2 dist scaled by power beta. Is a vector of length n/one value per location.
        diff_Y_y = torch.pow(
            torch.abs( # Absolute value because 1D obs
                (observations_y.unsqueeze(1) -
                simulations_Y.unsqueeze(0)).float(),
                ),
            beta)

        # Second part |Y-Y'|. 2* because pdist counts only once.
        diff_Y_Y = torch.pow(
            torch.abs( # Absolute value because 1D obs
                (simulations_Y.unsqueeze(1) -
                simulations_Y.unsqueeze(0)).float(),
                ),
            beta)
        Energy = 2 * torch.mean(diff_Y_y) - torch.sum(diff_Y_Y) / (m * (m - 1))
        return Energy

def crps_samples(samples,obs):
    ''' 
    Computes an empirical approximation to the CRPS(X,y) based on samples X compared to an observation y,
    averaged over observations.
    
    Inputs
    ---
    samples: torch.tensor (n_samples,)
        samples from the predictive distribution
    obs: torch.tensor (n_obs,)
        observations

    Outputs
    ---
    crps: torch.tensor (1,)
        CRPS(X,y) value
    '''
    # first term |X-y|
    crps1 = torch.abs(samples.unsqueeze(0)-obs.unsqueeze(1)).mean()
    # second term E|X-X'|
    crps2 = torch.abs(samples.unsqueeze(0)-samples.unsqueeze(1)).sum()/(len(samples)*(len(samples)-1))
    # Final CRPS = |X-y| - 1/2 * E|X-X'|
    return crps1 - 0.5*crps2


def crps_integral(grid,cdf_grid,obs):
    ''' 
    Computes the CRPS(X,y) using numerical integration with the trapezoidal rule on the CDF of the model.
    
    Inputs
    ---
    grid: torch.tensor (n_grid,)
        grid of points where the CDF is evaluated to provide an approximation to the integral
    cdf_grid: torch.tensor (n_grid,)
        values of the model's CDF at the grid points
    obs: torch.tensor (n_obs,)
        observations

    Outputs
    ---
    crps: torch.tensor (1,)
        CRPS(X,y) value
    '''

    # Compute Sum(y) [CDF(x) - I(x>y)]^2
    cdf_diff = torch.mean((cdf_grid - (grid>obs.unsqueeze(1)).float())**2, dim=0)

    # Compute the integral using the trapezoidal rule
    crps = torch.trapz(cdf_diff, grid)

    return crps