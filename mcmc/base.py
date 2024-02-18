from abc import ABC, abstractmethod
import torch
__all__ = ['MCMC']

class MCMC(ABC):

    @abstractmethod
    def step(self,
             x: torch.Tensor,
             retain_graph: bool = False):
        ...


    def simulate(self,
                 x0: torch.Tensor,
                 burn_in: int,
                 retain_graph: bool = False):
        r"""generates samples via MCMC with a given burn-in period
        and initial value X of size (N, D)."""
        xN = x0
        for n in range(burn_in):
            xN = self.step(xN, retain_graph)
        return xN





def autograd_backward():

    x = torch.tensor([3.], requires_grad=True)
    y = x**3

    y.backward(create_graph=True)
    print('dy/dx:', x.grad)

    dy = x.grad.clone()     # necessary (why?)
    x.grad.zero_()          # necessary (to prevent gradient accumulation)
    dy.sum().backward(create_graph=True)
    #dy.backward(gradient=torch.ones_like(dy))
    print('d2y/dx2:', x.grad)   # now has 2nd order gradients

    dy = x.grad.clone()
    x.grad.zero_()
    dy.sum().backward()
    print('d3y/dx3:', x.grad)   # now has 3rd order gradients (but no computation graph)


def autograd_grad():
    from torch.autograd import grad

    x1 = torch.tensor([
        [2., 3.],
        [1., -2.]
    ], requires_grad=True)
    x2 = torch.tensor([
        [6., 4.],
        [-2., -5.]
    ], requires_grad=True)

    y = 3*x1**3 - x2**2     # (N, D)

    dy = grad(y.sum(), x2, create_graph=True)[0]
    d2y = grad(dy.sum(), x2, create_graph=True)[0]
    #dy = grad(y, x2, grad_outputs=torch.ones_like(y), create_graph=True)[0]
    #d2y = grad(dy, x2, grad_outputs=torch.ones_like(dy), create_graph=True)[0]
    print('dy:', dy)
    print('d2y:', d2y)


def autograd_require_grad():
    X = torch.Tensor([
        [0,0],
        [1,1],
        [1,-1],
    ])
    X.requires_grad_(True)
    Y = X**2                # tensors have both grad_fn AND requires_grad==True
    X.requires_grad_(False) # X.grad == None but backward() works since there is grad_fn
    X.requires_grad_(True)  # X.grad works and is correct answer
    Y.sum().backward()
    print(X.grad)

