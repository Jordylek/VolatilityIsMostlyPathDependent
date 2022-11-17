# from montecarlo_utils import *
import torch
import numpy as np
from torch.utils.data import DataLoader, Dataset, random_split
from torch import optim, nn
import copy

torch_normal_law = torch.distributions.normal.Normal(0, 1.)
dt = 1/252

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

pi = torch.tensor(np.pi)


def convert_to_tensor(x, array=True):
    """

    :param x: float or array-like
    :param float: bool. If float is False, then x always return a tensor of dimension at least 1.
    :return: tensor
    """
    if not isinstance(x, torch.Tensor):
        x = torch.tensor(x).float()
    if array:
        if x.dim() == 0:
            x.unsqueeze_(0)
    return x


def black_gammaGPU(K, T, F, vol):
    """

    :param K: torch.Tensor of shape k
    :param T: float, maturity
    :param F: torch.Tensor of shape N or float, forward price
    :param vol: float,
    :return: torch.Tensor of shape k x N containing the Black Gamma
    """
    T = convert_to_tensor(T, array=False)
    vol = convert_to_tensor(vol, array=False)
    K = convert_to_tensor(K, array=True)
    F = convert_to_tensor(F, array=True)
    s = vol * T.sqrt()
    d1 = torch.log(torch.true_divide(F, K)) / s + 0.5 * s
    norm_pdf = torch.exp(-d1 ** 2 / 2) / torch.sqrt(2 * pi)
    return norm_pdf / (F * s)


def black_vegaGPU(K, T, F, vol):
    """

    :param K: torch.Tensor of shape k
    :param T: float, maturity
    :param F: torch.Tensor of shape N or float, forward price
    :param vol: float,
    :return: torch.Tensor of shape k x N containing the Black Vega
    """
    T = convert_to_tensor(T, array=False)
    vol = convert_to_tensor(vol, array=False)
    K = convert_to_tensor(K, array=True)
    F = convert_to_tensor(F, array=True)
    s = vol * T.sqrt()
    d1 = torch.log(torch.true_divide(F, K)) / s + 0.5 * s
    norm_pdf = torch.exp(-d1 ** 2 / 2) / torch.sqrt(2 * pi)
    return F * norm_pdf * T.sqrt()


def black_priceGPU(K, T, F, vol):
    """

    :param K: torch.Tensor of shape k
    :param T: float, maturity
    :param F: torch.Tensor of shape N or float, forward prices
    :param vol: float,
    :return: torch.Tensor of shape k x N containing the Black price
    """
    T = convert_to_tensor(T, array=False)
    vol = convert_to_tensor(vol, array=False)
    K = convert_to_tensor(K, array=True)
    F = convert_to_tensor(F, array=True)
    s = vol * T.sqrt()
    d1 = torch.log(torch.true_divide(F, K)) / s + 0.5 * s
    return torch_normal_law.cdf(d1) * F - torch_normal_law.cdf(d1 - s) * K


def black_imp_volGPU_old(K, T, F, price, iters=1000):
    """

    :param K: torch.Tensor of shape k
    :param T: float, maturity
    :param F: torch.Tensor of shape N or float, forward prices
    :param price: torch.Tensor of shape k x N
    :param iters: integer. Number of iterations in Newton's method
    :return:
    """
    T = convert_to_tensor(T, array=False)
    price = convert_to_tensor(price, array=True)
    K = convert_to_tensor(K, array=True)
    F = convert_to_tensor(F, array=True)
    vol = (2 * pi / T).sqrt() * price/F
#     if vol.isnan().any():
#         print(f'F: {F}, price: {price}, vol: {vol}', )
    obj = black_priceGPU(K, T, F, vol) - price
    vega = black_vegaGPU(K, T, F, vol)
    eps = 1e-4
    it = 0
    while (obj.abs() > eps).any():
        it += 1
        vol -= obj / (vega + eps)  # adding the small epsilon avoid division by 0, which can cause the gradient to be NaN
        vol = vol.clamp(min=eps, max=5.)
        obj = black_priceGPU(K, T, F, vol) - price
        vega = black_vegaGPU(K, T, F, vol)
        
        if it >= iters:
            break
    return vol


def black_imp_volGPU(K, T, F, price, tol=1e-6, maxiter=500):
    """
    Parameters
    ----------
    K: tensor of size n_K
        strikes
    T: tensor of size n_T
        maturities
    F: scalar tensor
        spot
    price: tensor of size (n_K, n_T)
        prices
    callput: str
        Must be either 'call' or 'put'
    Returns
    -------
    iv_KT: tensor of size (n_K, n_T)
        implied volatility surface
    """
    device = F.device
    T = convert_to_tensor(T, array=True).to(device)
    assert (K > 0).all() and (T > 0).all() and (F > 0).all()

    n_K = K.shape[0]
    n_T = T.shape[0]
    K = K[:, None]
    T = T[None, :]
    price = price.unsqueeze(1).clone() / F
    K = K / F
    opttype = 1
#     price = price.clone()
    price -= torch.maximum(opttype * (1 - K), torch.tensor(0, device=device))
    
    price[price < 0] = torch.nan
    price[price == 0] = 0
    
    normal = torch.distributions.normal.Normal(0, 1.)
    j = 1
    p = torch.log(K)
    idx_sup = (K >= 1) * 1
    idx_inf = (K < 1) * 1
    x0 = torch.sqrt(torch.absolute(2 * p))
    tmp_0  = (0.5 - K * normal.cdf(-x0) - price) * torch.sqrt(2 * pi) * idx_sup
    tmp_1 = (0.5 * K - normal.cdf(-x0) - price) * torch.sqrt(2 * pi) * idx_inf / K
    x1 = x0 - (tmp_0 + tmp_1)
    x1[x1.isnan()] = 1e-3
    while (abs(x0 - x1) > tol * torch.sqrt(T)).any() and (j < maxiter):
        x0 = x1
        d1 = - p / x1 + 0.5 * x1
#         if d1.isnan().any():
#             print(j, x1.reshape(-1), d1.reshape(-1))
        tmp_1 = (normal.cdf(d1) - K * normal.cdf(d1 - x1) - price) * idx_sup
        tmp_2 = (K * normal.cdf(x1 - d1) - normal.cdf(-d1) - price) * idx_inf
        tmp_3 = (tmp_1 + tmp_2) * torch.sqrt(2 * pi) * torch.exp(0.5 * d1 ** 2)
        x1 = x1 - tmp_3
        x1 = x1.clamp(min=1e-3, max=5) # /!\ Ask Florian if it is a good idea
        x1[x1.isnan()] = 1e-3
        j += 1
    return (x1 / torch.sqrt(T)).reshape(-1)


def torch_payoff_call(S, K):
    K = convert_to_tensor(K, array=True)
    return torch.clamp(S[:, np.newaxis] - K, min=0)


def torch_payoff_put(S, K):
    K = convert_to_tensor(K, array=True)
    return torch.clamp(K - S[:, np.newaxis], min=0)