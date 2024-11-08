from .utils import *
import torch


class DistributionFunctions:
    def __init__(self):
        pass

    @staticmethod
    def inverse_std_normal(cumulative_prob):
        """
        Inverse of the standard normal distribution.
        --- Input
        cumulative_prob: the cumulative probability
        --- Output
        the inverse cdf of the standard normal distribution
        """
        cumulative_prob_double = cumulative_prob.double()
        return torch.clamp(torch.erfinv(2 * cumulative_prob_double - 1) * torch.sqrt(torch.tensor(2.0)), -10,
                           10).float()

    @staticmethod
    def cdf_std_normal(input):
        """
        CDF of the standard normal distribution.
        --- Input
        input: the input value
        --- Output
        the cdf of the standard normal distribution
        """
        return torch.distributions.normal.Normal(loc=0, scale=1).cdf(input)

    @staticmethod
    def pdf_std_normal(input):
        """
        PDF of the standard normal distribution.
        --- Input
        input: the input value
        --- Output
        the pdf of the standard normal distribution
        """
        return torch.distributions.normal.Normal(loc=0, scale=1).log_prob(input).exp()

    @staticmethod
    def bvn_density(rho, u, v, shift=0.0, scale=1.0):
        """
        Bivariate normal density function with correlation rho given copula-scale input.
        --- Input
        rho: the correlation coefficient
        u: the [0,1]-valued input in 1st dimension
        v: the [0,1]-valued input in 2nd dimension
        shift: the shift parameter
        scale: the scale parameter
        --- Output
        pdf value
        """
        mean = torch.tensor([shift, shift])
        covariance_matrix = torch.tensor([[scale, rho], [rho, scale]])
        multivariate_normal = torch.distributions.MultivariateNormal(mean, covariance_matrix)

        l = len(u)
        input = torch.cat([u.reshape(l, 1), v * torch.ones(l, 1)], dim=1)

        return multivariate_normal.log_prob(DistributionFunctions.inverse_std_normal(input)).exp()

    @staticmethod
    def GC_density(rho, u, v, shift=0.0, scale=1.0):
        """
        PDF of Gaussian Copula.
        --- Input
        rho: the correlation parameter
        u: the input value in 1st dimension
        v: the input value in 2nd dimension
        shift: the shift parameter
        scale: the scale parameter
        --- Output
        pdf value
        """
        l = len(u)

        v_d = DistributionFunctions.pdf_std_normal(DistributionFunctions.inverse_std_normal(v))
        u_d = DistributionFunctions.pdf_std_normal(DistributionFunctions.inverse_std_normal(u)).reshape(l, 1)
        low = u_d * v_d

        up = DistributionFunctions.bvn_density(rho=rho, u=u, v=v).reshape(l, 1)

        return up / low

    @staticmethod
    def cGC_distribution(rho, u, v, shift=0.0, scale=1.0):
        """
        Conditional Gaussian Copula distribution function. C(u|v)
        --- Input
        rho: the correlation parameter
        u: the input value in 1st dimension
        v: the input value in 2nd dimension
        shift: the shift parameter
        scale: the scale parameter
        --- Output
        conditional cdf value of u conditional on v
        """
        upper = DistributionFunctions.inverse_std_normal(u).reshape(len(u),
                                                                    1) - rho * DistributionFunctions.inverse_std_normal(
            v)
        lower = torch.sqrt(1 - rho ** 2)
        input = upper / lower

        return DistributionFunctions.cdf_std_normal(input)


# Example usage:
if __name__ == '__main__':
    dist = DistributionFunctions()

    u = torch.tensor([0.1, 0.2, 0.3])
    v = torch.tensor([0.4])
    rho = 0.5

    # Example of using one of the methods:
    density = dist.GC_density(rho, u, v)
    print("GC Density:", density)