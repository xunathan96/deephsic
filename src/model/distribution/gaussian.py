__all__ = ['Gaussian']
import torch
import torch.nn as nn
import distribution


class Gaussian(nn.Module, distribution.Gaussian):

    def __init__(self, dim=1):
        super().__init__()
        self.dim = dim

        # mean parameter
        mu = torch.empty(self.dim)
        nn.init.normal_(mu)
        self.mean = nn.Parameter(mu)

        # lower triangular matrix parameter
        lower_arr = torch.empty(self.dim*(self.dim+1)//2)
        nn.init.normal_(lower_arr)
        self.L_arr = nn.Parameter(lower_arr)

        # track covariance gradients
        #self.cov = self.L @ self.L.T
        #self.cov.retain_grad()

    # needed property so that we can recompute the computation graph / gradients for every forward
    @property
    def L(self):
        device = self.L_arr.device
        lower_mat = torch.zeros((self.dim, self.dim), device=device)
        lower_mat[torch.tril_indices(self.dim, self.dim, device=device).tolist()] = self.L_arr
        return lower_mat
    
    @property
    def cov(self):
        return self.L @ self.L.T

    @property
    def precision(self):
        return torch.linalg.inv(self.cov)

    @property
    def log_det(self):
        return torch.logdet(self.cov)




def main(args):
    from mcmc import MALA
    import matplotlib.pyplot as plt

    x = torch.Tensor([
        [0,0],
        [1,1],
        [1,-1],
    ])

    modelGaussian = Gaussian(dim=2)
    logp = modelGaussian.log_prob(x)
    print('MEAN:', modelGaussian.mean)
    print('COV:', modelGaussian.cov)
    print('logp(x):', logp)

    # simulate MALA
    mala = MALA(stationary_distribution=modelGaussian, step_size=args.step_size)
    x0 = 2*torch.rand((args.n_samples, 2)) - 1
    x = mala.simulate(x0, burn_in=args.burn_in, retain_graph=True)
    print(x)

    # plot real and model gaussian
    gaussian = Gaussian(modelGaussian.mean, modelGaussian.cov)
    y = gaussian.sample(n_samples=1000)
    x = x.detach().cpu()
    y = y.detach().cpu()
    plt.scatter(x[:,0], x[:,1], s=0.5)
    plt.scatter(y[:,0], y[:,1], s=0.5)
    plt.axis('equal')
    plt.show()

