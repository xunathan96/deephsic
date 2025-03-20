import numpy as np
from typing import Callable
from scipy.stats import ortho_group
# import sys
# sys.path.append('../..')
from .base import ToyDataset, ToyIterator
from data.transforms import numpy_to_tensor
__all__ = ['ISAGenerator']



class ISAGenerator(ToyIterator):
    
    def __init__(self,
                 dim: int,
                 std: float,
                 rot: float,
                 transform: Callable = numpy_to_tensor,
                 seed: int = None):
        super().__init__(transform, seed)
        assert 0 <= rot <= 1
        self.dim = dim
        self.std = std
        self.theta = rot * np.pi/4
        c, s = np.cos(self.theta), np.sin(self.theta)
        self.R = np.array(((c, -s), (s, c)))


    def __next__(self):
        x, y = self.sample_mixture()
        joint = self.R @ np.concatenate((x, y), axis=0)     # (2,)
        noise = self.rng.normal(0, 1, size=(2, self.dim-1)) # (2, d-1)
        joint = np.concatenate((joint[:,np.newaxis], noise), axis=-1)   # (2,d)
        if self.dim > 1:
            Ox = ortho_group.rvs(dim=self.dim, random_state=self.seed)
            Oy = ortho_group.rvs(dim=self.dim, random_state=self.seed)
            x = Ox @ joint[0]
            y = Oy @ joint[1]
            x, y = self.transform(np.stack((x, y), axis=0))
        else:
            x, y = self.transform(joint)        
        return x, y


    def sample_mixture(self):
        alpha = self.rng.uniform(size=2)
        means = 2 * (alpha < 0.5).astype(int) - 1
        joint = means + self.rng.normal(0, self.std, size=2)
        return np.split(joint, 2) #joint[0], joint[1]





def generate_ISA(n,d,sigma_normal,alpha):
    # https://github.com/renyixin666/HSIC-LK/blob/main/dataset.py
    
    x = np.concatenate((np.random.normal(-1, sigma_normal, n//2), np.random.normal(1, sigma_normal, n//2)))
    y = np.concatenate((np.random.normal(-1, sigma_normal, n//2), np.random.normal(1, sigma_normal, n//2)))
    p = np.random.permutation(n)
    y_p = y[p]

    D = np.zeros([2,n])
    D[0,:] = x
    D[1,:] = y_p

    theta = np.pi/4*alpha
    c, s = np.cos(theta), np.sin(theta)
    R = np.array(((c, -s), (s, c)))

    D_R = R@D
    X_mix = D_R[0,:].reshape(-1,1)
    Y_mix = D_R[1,:].reshape(-1,1)

    X_z = np.random.randn(n,d-1)
    Y_z = np.random.randn(n,d-1)

    X_con = np.concatenate((X_mix,X_z), axis=1) # (n,d)
    Y_con = np.concatenate((Y_mix,Y_z), axis=1)

    m_x = ortho_group.rvs(dim=d)
    m_y = ortho_group.rvs(dim=d)

    X = (m_x@X_con.T).T
    Y = (m_y@Y_con.T).T
    
    return X,Y




def main():
    from torch.utils.data import DataLoader
    import matplotlib.pyplot as plt

    dataset = ISAGenerator(dim=1, std=0.1, rot=0.5)
    testloader = DataLoader(dataset, batch_size=128)
    testiter = iter(testloader)
    X, Y = next(testiter)

    plt.scatter(X, Y)
    plt.show()
    





if __name__ == '__main__':
    main()

