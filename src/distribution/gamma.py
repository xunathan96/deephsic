import torch
from torch.distributions import Gamma
from torch.autograd import gradcheck
from scipy import stats
__all__ = ['GammaCDF', 'GammaInvCDF']


###############################
#       PyTorch Comment       #
###############################
# https://github.com/pytorch/pytorch/issues/41637#issuecomment-908603489

def d_igamma_dp_series_expansion(p: torch.Tensor, x: torch.Tensor, eps: float = 1e-5, n_max: int = 100) -> torch.Tensor:
    """
    Algorithm AS 187: Derivatives of the Incomplete Gamma Integral
    Author(s): R. J. Moore
    Stable URL: http://www.jstor.org/stable/2348014

    Calculate d igamma(p,x) / dp using a series expansion (cf. Moore, 1982)
    This is valid for p <= x <= 1 and also for x < p.
    Notation from Moore 1982
    Args:
        p: actual observations
        x: model prediction
        eps: the minimal accepted accuracy when the series expansion is
        n_max: max number of iteration
    Returns:
        Calculate d igamma(p,x) valid for p <= x <= 1
    """
    log_x = torch.log(x)
    p_plus_1 = p + 1.0
    log_f = p * log_x - torch.lgamma(p_plus_1) - x
    f = log_f.exp()
    df_dp = f * (log_x - torch.polygamma(0, p_plus_1))

    C_old = torch.ones_like(p)
    d_C_old_dp = torch.zeros_like(p)
    S = C_old
    dS_dp = d_C_old_dp

    idx_notconv = torch.arange(0, p.numel(), dtype=torch.long)

    converged = False

    for n in range(1, n_max):
        ppn = torch.reciprocal(p + n)
        C_new = x * ppn * C_old
        d_C_new_dp = C_new * (torch.reciprocal(C_old) * d_C_old_dp - ppn)
        # update indices
        idx_notconv = torch.where(torch.abs(C_new) > S * eps)[0]

        if idx_notconv.numel() > 0:
            S[idx_notconv] = (S + C_new)[idx_notconv]
            dS_dp[idx_notconv] = (dS_dp + d_C_new_dp)[idx_notconv]
        else:
            converged = True
            break

        C_old = C_new
        d_C_old_dp = d_C_new_dp

    return S * df_dp + f * dS_dp


def d_igamma_dp_cf_expansion(p: torch.Tensor, x: torch.Tensor, eps: float = 1e-5, n_max: int = 100) -> torch.Tensor:
    """
    Algorithm AS 187: Derivatives of the Incomplete Gamma Integral
    Author(s): R. J. Moore
    Stable URL: http://www.jstor.org/stable/2348014

    Calculate d igamma(p,x) / dp using a continued fraction expansion (cf. Moore, 1982)
    This is valid for the entire input domain outside of { {p <= x <= 1} U {x < p} }.
    Notation from Moore 1982
    Args:
        p: actual observations
        x: model prediction
        eps: the minimal accepted accuracy when the series expansion is
        n_max: max number of iteration
    Returns:
        Calculate d igamma(p,x) valid for p <= x <= 1
    """
    log_x = torch.log(x)
    log_f = p * log_x - torch.lgamma(p) - x
    f = log_f.exp()
    df_dp = f * (log_x - torch.polygamma(0, p))

    A0 = torch.ones_like(p)
    B0 = x
    A1 = x + 1.0
    B1 = x * (2.0 - p + x)

    # init derivatives
    dA0_dp = torch.zeros_like(p)
    dA1_dp = torch.zeros_like(p)
    dB0_dp = torch.zeros_like(p)
    dB1_dp = -x

    S = torch.zeros_like(p)
    S_old = torch.zeros_like(p)
    S_temp = torch.zeros_like(p)
    dS_dp = torch.zeros_like(p)

    # tensor indices that haven't converged yet (initially all of them)
    idx_notconv = torch.arange(0, p.numel(), dtype=torch.long)

    converged = False

    for n in range(2, n_max):
        a, b = (n - 1) * (p - n), 2.0 * n - p + x
        A2 = b * A1 + a * A0
        B2 = b * B1 + a * B0
        dA2_dp = b * dA1_dp - A1 + a * dA0_dp + (n - 1) * A0
        dB2_dp = b * dB1_dp - B1 + a * dB0_dp + (n - 1) * B0

        S_old[idx_notconv] = (A1 / B1)[idx_notconv]
        S_temp[idx_notconv] = (A2 / B2)[idx_notconv]

        # update indices
        idx_notconv = torch.where(torch.abs(S_temp - S_old) > S_temp * eps)[0]

        if idx_notconv.numel() > 0:
            S[idx_notconv] = S_temp[idx_notconv]
            # TODO: refactor in the future - this might be numerically unstable
            dS_dp[idx_notconv] = (dA2_dp / B2 - S_temp * (dB2_dp / B2))[idx_notconv]
        else:
            converged = True
            break

        # update intermediates and partials
        A0, B0 = A1, B1
        A1, B1 = A2, B2
        dA0_dp, dB0_dp = dA1_dp, dB1_dp
        dA1_dp, dB1_dp = dA2_dp, dB2_dp

    return -S * df_dp - f * dS_dp


def d_igamma_dp(p: torch.Tensor, x: torch.Tensor, eps: float = 1e-5, n_max: int = 200) -> torch.Tensor:
    """
    Algorithm AS 187: Derivatives of the Incomplete Gamma Integral
    Author(s): R. J. Moore
    Stable URL: http://www.jstor.org/stable/2348014

    Calculate the derivative of the incomplete gamma function as the series expansion
    Notation from Moore 1982
    Args:
        p: actual observations
        x: model prediction
        eps: the minimal accepted accuracy when the series expansion is
        n_max: max number of iteration
    Returns:
        derivative of the incomplete gamma function
    """
    # uses the notation in (Moore, 1982)
    d_igamma = torch.zeros_like(p)
    idx = torch.logical_or(torch.logical_and(p <= x, x <= 1), x <= p)
    idx_series = idx.nonzero(as_tuple=True)[0]
    idx_continued_fraction = (~idx).nonzero(as_tuple=True)[0]
    if idx_series.numel() > 0:
        p_s, x_s = p[idx_series], x[idx_series]
        d_igamma[idx_series] = d_igamma_dp_series_expansion(p_s, x_s, eps, n_max)
    if idx_continued_fraction.numel() > 0:
        p_cf, x_cf = p[idx_continued_fraction], x[idx_continued_fraction]
        d_igamma[idx_continued_fraction] = d_igamma_dp_cf_expansion(p_cf, x_cf, eps, n_max)
    return d_igamma


class CustomIGamma(torch.autograd.Function):
    """
    PyTorch IGamma function implementation
    https://pytorch.org/docs/stable/distributions.html
    """

    def forward(self, a: torch.Tensor, z: torch.Tensor):  # type: ignore # pylint: disable=W0221
        cdf_ = torch.igamma(a, z)
        self.save_for_backward(a, z)
        return cdf_

    def backward(self, grad_output):  # pylint: disable=W0221
        a, z = self.saved_tensors
        d_igamma_a = d_igamma_dp(a, z)
        # The exact formula can be found:
        # https://www.wolframalpha.com/input/?i=d+GammaRegularized%5Ba%2C+0%2C+z%5D+%2F+dz
        d_igamma_z = (-z + (a - 1.0) * torch.log(z) - torch.lgamma(a)).exp()
        return (grad_output * d_igamma_a, grad_output * d_igamma_z)


class GammaCDF_(torch.autograd.Function):
    """
    PyTorch Gamma distribution CDF implementation. This implementation solves the following issue raised on the PyTorch forum:
    https://github.com/pytorch/pytorch/issues/41637
    """

    def forward(self, x: torch.Tensor, k: torch.Tensor, psi: torch.Tensor):  # type: ignore # pylint: disable=W0221
        cdf_ = torch.igamma(k, x / psi)
        self.save_for_backward(k, x, psi)
        return cdf_

    def backward(self, grad_output, eps=1e-5):  # pylint: disable=W0221
        k, x, psi = self.saved_tensors
        a = k

        # Numerical stability
        x[x < eps] = eps    # NOTE: in-place op.
        z = x / psi

        d_igamma_a = d_igamma_dp(a, z, eps)
        d_igamma_z = (-z + a * torch.log(z) - torch.lgamma(a)).exp()
        return (grad_output * d_igamma_z / x, grad_output * d_igamma_a, -grad_output * d_igamma_z / psi)


###############################
#       Custom Autograd       #
###############################

class GammaCDF(torch.autograd.Function):

    @staticmethod
    def forward(ctx, input: torch.Tensor, shape: torch.Tensor, scale: torch.Tensor):
        cdf = torch.igamma(shape, input / scale)
        ctx.save_for_backward(input, shape, scale)
        return cdf

    @staticmethod
    def backward(ctx, grad_out, eps=1e-6):
        input, shape, scale = ctx.saved_tensors
        input = torch.clamp(input, min=eps) # stability
        # chain rule
        z = input / scale
        dz_dscale = - input / scale**2
        dz_dinput = 1 / scale
        digamma_dz = torch.exp(- torch.lgamma(shape) + (shape-1) * torch.log(z) - z)
        digamma_dshape = d_igamma_dp(shape, z, eps)
        digamma_dscale = - torch.exp(- torch.lgamma(shape) + shape * torch.log(z) - z - torch.log(scale))
        digamma_dinput = digamma_dz * dz_dinput
        return grad_out * digamma_dinput, grad_out * digamma_dshape, grad_out * digamma_dscale


class GammaInvCDF(torch.autograd.Function):

    @staticmethod
    def forward(ctx, input: float, shape: torch.Tensor, scale: torch.Tensor):
        # input: float or array
        # shape/scale: tensor of dimension at least 1
        shape_np = shape.detach().cpu().numpy()
        scale_np = scale.detach().cpu().numpy()
        gamma = stats.gamma(shape_np, scale=scale_np)
        q = gamma.ppf(input)
        fq = gamma.pdf(q)
        ctx.constants = (q, fq)
        ctx.save_for_backward(shape, scale)
        return torch.from_numpy(q).to(dtype=shape.dtype, device=shape.device)

    @staticmethod
    def backward(ctx, grad_out: torch.Tensor, eps=1e-6):
        (q_np, fq_np) = ctx.constants
        shape, scale = ctx.saved_tensors
        q = torch.from_numpy(q_np).to(dtype=grad_out.dtype, device=grad_out.device)
        fq = torch.from_numpy(fq_np).to(dtype=grad_out.dtype, device=grad_out.device)
        q = torch.clamp(q, min=eps) # stability
        fq = torch.clamp(fq, min=eps)
        # gradient of gamma cdf (igamma)
        z = q / scale
        digamma_dshape = d_igamma_dp(shape, z, eps)
        digamma_dscale = - torch.exp(- torch.lgamma(shape) + shape * torch.log(z) - z - torch.log(scale))
        # implicit gradients: -1/(f(r)) * ∇F(r)
        dquant_dshape = - (1/fq) * digamma_dshape
        dquant_dscale = - (1/fq) * digamma_dscale
        return None, grad_out * dquant_dshape, grad_out * dquant_dscale

cdf = GammaCDF.apply
icdf = GammaInvCDF.apply



def test_gamma():
    ##################################
    # Testing the GammaCDF function  #
    ##################################
    torch.autograd.set_detect_anomaly(True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("Device type: ", device)

    x = 10 * torch.rand(1, dtype=torch.double, device=device, requires_grad=True)
    shape = 10 * torch.rand(5, dtype=torch.double, device=device, requires_grad=True)
    scale = 1 + torch.rand(5, dtype=torch.double, device=device, requires_grad=True)
    cdf = GammaCDF.apply(x, shape, scale)
    print(f"{cdf=}")
    test = gradcheck(GammaCDF.apply, (x, shape, scale), eps=1e-6, atol=1e-4)
    print("GammaCDF", test)



def test_gamma_inv():
    ##################################
    #       Testing GammaInvCDF      #
    ##################################
    import numpy as np
    torch.autograd.set_detect_anomaly(True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("Device type: ", device)

    n = 100
    # significance = np.random.uniform(size=n)
    # shape = torch.rand(n, dtype=torch.double, device=device, requires_grad=True)
    # scale = torch.rand(n, dtype=torch.double, device=device, requires_grad=True)

    significance = 0.05
    shape = torch.rand(1, dtype=torch.double, device=device, requires_grad=True)
    scale = torch.rand(1, dtype=torch.double, device=device, requires_grad=True)

    shape = torch.tensor(1, dtype=torch.double, device=device, requires_grad=True)
    print(shape.shape)  #torch.Size([1])

    quantile = GammaInvCDF.apply(significance, shape*10, scale+1)
    print(f"{quantile=}")

    # quantile.backward(torch.ones(n, device=device))
    # print(f"{shape.grad=}")
    # print(f"{scale.grad=}")

    test = gradcheck(GammaInvCDF.apply, (significance, shape*10, scale+1), eps=1e-6, atol=1e-4)
    print("GammaCDF", test)


if __name__ == '__main__':
    test_gamma_inv()



