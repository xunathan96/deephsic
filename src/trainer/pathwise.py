import torch
import numpy as np
from tqdm import tqdm

from config.config import Config
from .base import BaseTrainer

import matplotlib.pyplot as plt
from distribution import Distribution, Gaussian, Dirichlet
from mcmc import MCMC

class Pathwise(BaseTrainer):

    def __init__(self, config: Config):
        super().__init__(config)
        self.model: Distribution
        self.mcmc: MCMC = self.cfg['mcmc'].build(stationary_distribution=self.model)

        if self.cfg['distribution'] == 'gaussian':
            mean = torch.tensor([0.,0.], device=self.device)
            cov = torch.tensor([
                [1, 0.8],
                [0.8, 1],
            ], device=self.device)
            cov = torch.tensor([
                [2.2, -1.9],
                [-1.9, 1.8],
            ], device=self.device)
            self.distr = Gaussian(mean, cov)
        elif self.cfg['distribution'] == 'dirichlet':
            alpha = torch.tensor([2., 10.])
            self.distr = Dirichlet(concentration=alpha)

    def train_one_epoch(self, epoch: int, args):
        self.model.train()

        # simulate samples via MCMC
        if self.cfg['distribution'] == 'gaussian':
            x0 = 2*torch.rand((args.n_samples, self.cfg['model']['dim'])) - 1
        elif self.cfg['distribution'] == 'dirichlet':
            x0 = torch.rand((args.n_samples, self.cfg['model']['dim']-1))
        x0 = x0.to(self.device)

        x = self.mcmc.simulate(x0=x0,
                               burn_in=args.burn_in,
                               retain_graph=True)

        if self.cfg['distribution'] == 'dirichlet':
            x = torch.cat((x, 1-x), dim=-1)

        log_qx = self.model.log_prob(x) # (N,)
        log_px = self.distr.log_prob(x) # (N,)
        reverse_kl = torch.mean(log_qx - log_px)
        self.backprop(reverse_kl, self.optimizer)

        print(f"[{epoch+1}]    loss: {reverse_kl.item():.4f}")
        return reverse_kl.item()


    @torch.no_grad
    def validation(self, epoch: int):
        return super().validation(epoch)
    
    @torch.no_grad
    def inference(self):
        x = self.model.sample(n_samples=1000)
        y = self.distr.sample(n_samples=1000)
        x = x.detach().cpu().numpy()
        y = y.detach().cpu().numpy()

        fig, axis = plt.subplots(1, 2)
        axis[0].scatter(x[:,0], x[:,1], s=0.5, c='#ff7f0e') # default orange
        axis[0].set_title('Learned model')
        axis[1].scatter(y[:,0], y[:,1], s=0.5, c='#1f77b4') # default blue
        axis[1].set_title(f"Target {self.cfg['distribution']}")
        #plt.scatter(x[:,0], x[:,1], s=0.5)
        #plt.scatter(y[:,0], y[:,1], s=0.5)
        plt.axis('equal')
        plt.show()
        return x
    
    def compute_metrics(self, pred, label):
        return super().compute_metrics(pred, label)


    def pathwise_mcmc(self, args):
        print('MEAN:', self.model.mean)
        print('COV:', self.model.cov)

        # simulate MCMC
        x0 = (2*torch.rand((args.n_samples, self.cfg['model']['dim'])) - 1).to(self.device)
        x = self.mcmc.simulate(x0,
                               burn_in=args.burn_in,
                               retain_graph=True)

        # E[X]
        x_bar = torch.mean(x**2, dim=0)    # NOTE: first or second moment
        self.model.zero_grad()
        x_bar.sum().backward()
        print('mean:', x_bar)
        print('grad mean:', self.model.mean.grad)
        print('grad cov:', self.model.cov.grad)

        gaussian = Gaussian(self.model.mean, self.model.cov)
        y = gaussian.sample(n_samples=1000)
        x = x.detach().cpu()
        y = y.detach().cpu()
        plt.scatter(x[:,0], x[:,1], s=0.5)
        plt.scatter(y[:,0], y[:,1], s=0.5)
        plt.axis('equal')
        plt.show()



    def pathwise_rparam(self, args):
        print('MEAN:', self.model.mean)
        print('COV:', self.model.cov)

        # reparameterized sample
        x = self.model.sample(n_samples=args.n_samples)

        # E[X]
        x_bar = torch.mean(x, dim=0)    # NOTE: cant do X**2 since rsample only uses L not COV
        self.model.zero_grad()
        x_bar.sum().backward()
        print('mean:', x_bar)
        print('grad mean:', self.model.mean.grad)
        print('grad cov:', self.model.cov.grad)

        gaussian = Gaussian(self.model.mean, self.model.cov)
        y = gaussian.sample(n_samples=1000)
        x = x.detach().cpu()
        y = y.detach().cpu()
        plt.scatter(x[:,0], x[:,1], s=0.5)
        plt.scatter(y[:,0], y[:,1], s=0.5)
        plt.axis('equal')
        plt.show()



    def log_gradients(self, args, func = 'mean'):
        print('MEAN:', self.model.mean)
        print('COV:', self.model.cov)

        grad_mean = grad_cov = None
        mu_err_list = []
        cov_err_list = []

        xN = (2*torch.rand((args.n_samples, self.cfg['model']['dim'])) - 1).to(self.device)
        for n in tqdm(range(args.burn_in)):
            xN = self.mcmc.step(xN, retain_graph=True)

            if func=='mean':
                x_bar = torch.mean(xN, dim=0)
            elif func=='var':
                x_bar = torch.mean(xN**2, dim=0)
            self.model.zero_grad()
            x_bar.sum().backward(retain_graph=True)

            if self.model.mean.grad is not None:
                grad_mean = self.model.mean.grad.detach()
                mu_err = torch.sqrt(torch.sum((grad_mean-1)**2, dim=-1))
                mu_err_list.append(mu_err.item())

            if self.model.cov.grad is not None:
                grad_cov = self.model.cov.grad.detach()
                grad_var = torch.diagonal(grad_cov)
                cov_err = torch.sqrt(torch.sum((grad_var-1)**2, dim=-1))
                cov_err_list.append(cov_err.item())

        print('grad mean:', grad_mean)
        print('grad cov:', grad_cov)

        xaxis = np.arange(args.burn_in)
        mean_err = np.array(mu_err_list)
        cov_err = np.array(cov_err_list)
        plt.plot(xaxis, mean_err)
        plt.title('Gradient Estimation Error')
        plt.xlabel('Langevin step')
        plt.tight_layout()
        plt.show()

        gaussian = Gaussian(self.model.mean, self.model.cov)
        y = gaussian.sample(n_samples=1000)
        y = y.detach().cpu()
        plt.scatter(y[:,0], y[:,1], s=0.5)
        plt.axis('equal')
        plt.tight_layout()
        plt.show()

