from calibration.torch_utils import *
import torch
from empirical_study.plot_functions import *


def exp_kernel_GPU(t, lam, c=1):
    return c * lam * torch.exp(-lam * t)


def identity(x):
    return x


def squared(x):
    return x ** 2


def initialize_R(lam, past_prices=None, max_delta=1000, transform=identity):
    """
    Initialize the R_j for the 4FMPDV model.
    :param lam: torch.tensor of size 2. contains \lambda_{j,1} and \lambda_{j,2}
    :param past_prices: pd.Series. Contains the past prices of the asset.
    :param max_delta: int. Number of past returns to use to compute the weighted averages.
    :param transform: should be identity for R_1 and squared for R_2.
    :return: torch.tensor of size 2. containing R_j1 and R_j2
    """
    returns = 1 - past_prices.shift(1) / past_prices
    returns = torch.tensor(returns.values, device=lam.device)[-max_delta:].flip(0)
    timestamps = torch.arange(returns.shape[0], device=lam.device) * dt
    timestamps.unsqueeze_(0)
    lam = lam.unsqueeze(1)
    weights = exp_kernel_GPU(timestamps, lam)
    x = transform(returns)
    return torch.sum(x * weights, dim=1)


class TorchMonteCarloExponentialModel:
    def __init__(self, lam1, lam2, R_init1, R_init2, betas, theta1=0., theta2=0., parabolic=0., parabolic_offset=0.,
                 S0=1,
                 maturity=30 * dt,
                 timestep_per_day=5, N=int(1e5), vix_N=int(1e3), fixed_seed=True, seed_root=3748192728942,
                 vol_cap=1.5, device=device):
        """
        Used to simulate the prices of an asset whose volatility follows the dynamic:
        $$
        \frac{dS_t}{S_t} = \sigma_t dW_t \\
        \sigma_t = \beta_0 + \beta_1 R_{1,t} + \beta_2 \sqrt{R_{2,t}} + \beta_{1,2} R_{1}^2{\bf 1}_{R_1\geq c} \\
        R_{i,t} = (1-\theta_i) R_{i,0,t} + \theta_i R_{i,1,t}, i\in\{1,2\} \\
        dR_{1,j,t} = \lambda_{1,j} (\sigma_t dW_t - R_{1,j,t}), j\in \{0,1\} \\
        dR_{2,j,t} = \lambda_{2,j}  (\sigma_t^2 dt - R_{2,j,t}), j\in \{0,1\}
        $$
        :param lam1: array-like or tensor of size 2
        :param lam2: array-like or tensor of size 2
        :param R_init1: array-like or tensor of size 2
        :param R_init2: array-like or tensor of size 2
        :param betas: array-like or tensor of size 3
        :param S0: float. Default 1
        :param theta1: float or tensor of size 0
        :param theta2: float or tensor of size 0
        :param maturity: float. Maturity (in years) of the simulation
        :param timestep_per_day: int. Number of steps per day for the montecarlo simulation
        :param N: int. number of paths for the MC simulation
        :param vix_N: int. number of sub-paths for the computation of VIX
        :param fixed_seed: bool. If True, uses the seed_root as the initial seed of dW, then inc
        :param seed_root: int, first seed for the MC simulation
        :param vol_cap: float. the instanteneous volatility is capped at this value
        :param device: torch.device or str. Device on which the computation is done
        :param parabolic: float or tensor of size 0. Default 0. value of the parabolic coefficient $\beta_{1,2}$
        :param parabolic_offset: float or tensor of size 0. Default 0, Value of the offset $c$ before the parabolic term
        """
        self.device = torch.device(device)
        self.timestep_per_day = timestep_per_day
        self.timestep = torch.tensor(dt / timestep_per_day, device=self.device)
        self.maturity = maturity
        self.T = self.index_of_timestamp(self.maturity)  # number of integers
        self.N = N
        self.R_init1 = convert_to_tensor(R_init1).to(self.device)
        self.R_init2 = convert_to_tensor(R_init2).to(self.device)
        self.lam1 = convert_to_tensor(lam1).to(self.device)
        self.lam2 = convert_to_tensor(lam2).to(self.device)
        self.theta1 = convert_to_tensor(theta1).to(self.device)
        self.theta2 = convert_to_tensor(theta2).to(self.device)
        self.betas = convert_to_tensor(betas).to(self.device)
        self.vix_N = vix_N
        self.vix_steps = self.index_of_timestamp(30 / 365)
        self.S0 = S0
        self.fixed_seed = fixed_seed
        self.seed_root = seed_root
        self.vol_cap = vol_cap
        self.parabolic = convert_to_tensor(parabolic, array=False).to(self.device)
        self.parabolic_offset = convert_to_tensor(parabolic_offset, array=False).to(self.device)

    def index_of_timestamp(self, t):
        """

        :param t: float or torch.tensor
        :return: int, index of the timestamp t
        """
        return torch.round(t / self.timestep).to(int)

    def compute_vol(self, R_1, R_2):
        """
        computes volatility
        :param R_1: torch.tensor of the same size as R_2
        :param R_2: torch.tensor of the same size as R_1
        :return: volatity of the same size as R_1 or R_2
        """
        vol = self.betas[0] + self.betas[1] * R_1 + self.betas[2] * torch.sqrt(R_2) + self.parabolic * (
                R_1 - self.parabolic_offset).clamp(
            min=0) ** 2
        if self.vol_cap is None:
            return vol
        return vol.clamp(max=self.vol_cap)

    def compute_R(self, i=1):
        """
        computes $R_i = (1-\theta_i) R_{i,0} + \theta_i R_{i,1}, $
        :param i: int 1 or 2
        :return: return tensor of all $R_i$ from the simulation
        """
        if i == 1:
            return self.R1_array[:, 0] * (1 - self.theta1.cpu()) + self.R1_array[:, 1] * self.theta1.cpu()
        elif i == 2:
            return self.R2_array[:, 0] * (1 - self.theta2.cpu()) + self.R2_array[:, 1] * self.theta2.cpu()
        else:
            raise ValueError('i in (1,2) only')

    def _simulate(self, n_timesteps, n_paths, S0, R1_0, R2_0, seed_root=0, save_R=False):
        """
        Simulates n_paths over n_timesteps of the dynamics
        :param n_timesteps: int, number of timestepss
        :param n_paths: int, number of paths
        :param S0: float or tensor of size n_paths, initial value(s) of S
        :param R1_0: float or tensor of size (n_paths, 2), initial value(s) of R_{1,j}
        :param R2_0: float or tensor of size (n_paths, 2) initial value(s) of R_{2,j}
        :param seed_root: in
        :param save_vol_only: bool. If True, only keeps the tensor of volatility(to save memory). Otherwise, saves also S, R1 and R2.
        :return: tensor of volatility of shape (n_timesteps+1, n_paths) if save_vol_only. Otherwise, returns also S, R1 and R2 tensors.
        """
        r1 = R1_0.to(self.device)
        r2 = R2_0.to(self.device)
        vol_array = torch.zeros((n_timesteps + 1, n_paths), device=self.device)
        if save_R:
            R1_array = torch.zeros((n_timesteps + 1, 2, n_paths), device='cpu')  # (self.device)
            R2_array = torch.zeros((n_timesteps + 1, 2, n_paths), device='cpu')  # (self.device)
        S_array = torch.zeros((n_timesteps + 1, n_paths), device=self.device)
        S_array[0] = S0

        for t in range(n_timesteps):
            R1 = (1 - self.theta1) * r1[0] + self.theta1 * r1[1]
            R2 = (1 - self.theta2) * r2[0] + self.theta2 * r2[1]
            vol = self.compute_vol(R1, R2)
            vol_array[t] = vol
            if save_R:
                R1_array[t] = r1.cpu()
                R2_array[t] = r2.cpu()
            if self.fixed_seed:
                torch.manual_seed(seed_root + t)
            brownian_increment = self.timestep.sqrt() * torch.randn(n_paths, device=self.device)
            increment = vol * brownian_increment
            for j in range(2):
                # R1_array[t + 1, j] = (torch.exp(-self.lam1[j] * self.timestep) * (
                #         r1[j] + self.lam1[j] * increment)).to('cpu')
                # R2_array[t + 1, j] = (torch.exp(-self.lam2[j] * self.timestep) * (
                #         r2[j] + self.lam2[j] * vol ** 2 * self.timestep)).to('cpu')
                r1[j] = (torch.exp(-self.lam1[j] * self.timestep) * (r1[j] + self.lam1[j] * increment)).clone()
                r2[j] = (torch.exp(-self.lam2[j] * self.timestep) * (r2[j] + self.lam2[j] * vol ** 2 * self.timestep)).clone()
            S_array[t + 1] = S_array[t].clone() * torch.exp(increment - 0.5 * vol ** 2 * self.timestep)
        # r1 = R1_array[n_timesteps].clone().to(self.device)
        # r2 = R2_array[n_timesteps].clone().to(self.device)
        R1 = (1 - self.theta1) * r1[0] + self.theta1 * r1[1]
        R2 = (1 - self.theta2) * r2[0] + self.theta2 * r2[1]
        vol_array[n_timesteps] = self.compute_vol(R1, R2)
        if save_R:
            R1_array[n_timesteps] = r1.cpu()
            R2_array[n_timesteps] = r2.cpu()
            return S_array, vol_array, R1_array, R2_array
        else:
            return S_array, vol_array

    def simulate(self, save_R=True):
        """
        simulates until maturity.
        :return:
        """
        if len(self.R_init1.shape) == 1:
            R1_0 = self.R_init1.unsqueeze(1).repeat_interleave(self.N, dim=1)
            R2_0 = self.R_init2.unsqueeze(1).repeat_interleave(self.N, dim=1)
        else:
            R1_0 = self.R_init1
            R2_0 = self.R_init2
        if save_R:
            self.S_array, self.vol_array, self.R1_array, self.R2_array = \
                self._simulate(self.T, self.N, R1_0=R1_0, R2_0=R2_0, S0=self.S0, seed_root=self.seed_root,
                               save_R=save_R)
        else:
            self.S_array, self.vol_array = \
                self._simulate(self.T, self.N, R1_0=R1_0, R2_0=R2_0, S0=self.S0, seed_root=self.seed_root,
                               save_R=save_R)

    def compute_option_price(self, strikes=None, option_maturity=None, volatility=False, return_future=False,
                             var_reduction=True, sigma0=0.1):
        """
        Computes the call option prices onn the underlying
        :param strikes: float or torch.tensor of size n_K
        :param option_maturity: maturity of the option
        :param volatility: if True, computes the option prices on the instantaneous volality instead
        :param return_future: if True, returns the future/forward
        :param var_reduction: if True (only for S), uses theta-gamma method to reduce variance.
        :param sigma0: float. Default 0.1. Value of $\sigma_0$ for the variance reduction
        :return: tuple of strikes and option prices. both of size n_K
        """
        payoff = torch_payoff_call
        maturity = self.maturity if option_maturity is None else option_maturity
        index = int(torch.ceil(maturity / self.timestep))
        array = self.vol_array if volatility else self.S_array
        S = array[index]

        future = torch.mean(S)
        if strikes is None:
            strikes = future * torch.arange(0.7, 1.5, 0.1)
        strikes = convert_to_tensor(strikes, array=True).to(self.device)
        payoff_values = payoff(S, strikes)
        expected_value = payoff_values.mean(axis=0)
        if var_reduction and not volatility:
            expected_value_classic = expected_value
            black_price_0 = black_priceGPU(strikes, maturity, array[0, 0].clone(), sigma0)
            time_to_maturity = torch.arange(maturity, 0, -self.timestep, device=self.device)
            f_tT = self.S_array[
                   :index].clone()  # self.discount_curve(time_to_maturity) / self.discount_curve(T) = e^{\int_t^Tr_udu}
            f_per_strike = f_tT.unsqueeze(0)
            sigma_t = self.vol_array[:index].clone()
            gammas = black_gammaGPU(strikes[:, np.newaxis, np.newaxis], time_to_maturity[:, np.newaxis], f_per_strike,
                                    sigma0)
            PnL = (sigma_t ** 2 - sigma0 ** 2) * f_per_strike ** 2 * gammas
            pnl_per_simulation = 1 / 2 * PnL.sum(axis=1) * self.timestep
            pnl_expectancy = pnl_per_simulation.mean(axis=1)
            expected_value = pnl_expectancy + black_price_0
            expected_value[expected_value <= 0] = expected_value_classic[expected_value <= 0]
            expected_value = expected_value.clamp(min=(future - strikes).clamp(min=0))
        if return_future:
            return future, strikes, expected_value
        else:
            return strikes, expected_value

    def compute_implied_vol(self, strikes=None, option_maturity=None, volatility=False, var_reduction=True, n_batch=1):
        """
        Computes the implied volatility of the options on the underlying (or instantaneous volatility)
        :param strikes: torch.tensor of size n_K
        :param option_maturity: flaot. Maturity of the option
        :param volatility: float. If True, computes implied volatility on the instantaneous volatility instead
        :param var_reduction: float. If True, uses theta-gamma variance reduction technique for the underlying only
        :param n_batch: int. Divides the strikes per batches for memory saving.
        :return: tuple. forward value (float), strikes (tensor of size n_K) and implied volatilities (tensor of size n)K)
        """
        maturity = self.maturity if option_maturity is None else option_maturity
        if strikes is None:
            future, strikes, option_prices = self.compute_option_price(strikes, maturity, volatility=volatility,
                                                                       return_future=True, var_reduction=var_reduction)
        else:
            strikes = torch.tensor(strikes, device=self.device)
            n_K = strikes.shape[0]
            option_prices = torch.zeros(n_K, device=self.device)
            idxs = torch.linspace(0, n_K, n_batch + 1, dtype=torch.int)
            for i in range(n_batch):
                future, _, option_price = self.compute_option_price(strikes[idxs[i]:idxs[i + 1]], maturity,
                                                                    volatility=volatility,
                                                                    return_future=True, var_reduction=var_reduction)
                option_prices[idxs[i]:idxs[i + 1]] = option_price

        implied_vol = black_imp_volGPU(strikes, maturity, future, option_prices)
        return future, strikes, implied_vol, option_prices

    def _compute_vix_nested(self, vix_index, idxs=None, n_subpaths=None):
        """
        computes VIX via nested MC for the paths given by idxs
        :param vix_index: int. index corresponding to the vix maturity
        :param idxs: int or torch.tensor. paths whose vix is computed
        :param n_subpaths: int. number of subpaths used.
        :return:
        """
        if idxs is None:
            idxs = torch.arange(self.N)
        if n_subpaths is None:
            n_subpaths = self.vix_N
        size = len(idxs)
        # vix_index = int(torch.ceil(vix_maturity / self.timestep))

        S0 = torch.repeat_interleave(self.S_array[vix_index, idxs], n_subpaths)
        R1_0 = torch.repeat_interleave(self.R1_array[vix_index, :, idxs], n_subpaths, dim=1)
        R2_0 = torch.repeat_interleave(self.R2_array[vix_index, :, idxs], n_subpaths, dim=1)

        _, nested_vol_array = \
            self._simulate(self.vix_steps, size * n_subpaths, S0=S0, R1_0=R1_0, R2_0=R2_0, save_R=False)
        nested_vol_array = nested_vol_array.reshape((self.vix_steps + 1, size, -1))
        return (nested_vol_array ** 2).mean(dim=(0, 2))

    def compute_vix(self, vix_maturity, subset=None, n_batch=1):
        """
        compute the VIX via nested MC for each path at timestep vix_maturity
        :param vix_maturity: float,
        :param n_batch: int. Divides the paths in batches to compute VIX. This allows to save memory.
        :return: tensor of size self.N. VIX per path
        """
        if subset is None:
            subset = torch.arange(self.N)
        elif isinstance(subset, int):
            subset = torch.arange(subset)
        subset = convert_to_tensor(subset).sort().values
        n = subset.shape[0]
        vix_index = int(torch.ceil(vix_maturity / self.timestep))
        idxs = torch.linspace(0, n, n_batch + 1, dtype=torch.int)
        vix_squared = torch.zeros(self.N, device=self.device)
        for i in range(n_batch):
            vix_squared[idxs[i]:idxs[i + 1]] = self._compute_vix_nested(vix_index,
                                                                        idxs=subset[idxs[i]: idxs[i + 1]])
        return vix_squared.sqrt()

    def compute_vix_implied_vol(self, vix_maturity, strikes=None, subset=None, **kwargs):
        """
        Computes the implied volatility of VIX options for the given strikes
        :param vix_maturity: float, maturity of the VIX option.
        :param strikes: torch.tensor. Strikes
        :param subset: int or array-like. If int, selects the first subset paths to compute the VIX. If array-like of int, uses the indexes passed.
        :param kwargs: kwargs for the vix computation.
        :return:
        """
        vix = self.compute_vix(vix_maturity, subset=subset, **kwargs)
        vix_future = torch.mean(vix)
        if strikes is None:
            strikes = vix_future * torch.arange(0.7, 1.5, 0.1)
        else:
            strikes = torch.tensor([strikes]).reshape(-1)
        strikes = torch.tensor(strikes, device=self.device)
        payoff_values = torch_payoff_call(vix, strikes)
        option_price = payoff_values.mean(dim=0)
        implied_vol = black_imp_volGPU(strikes, vix_maturity, vix_future, option_price)
        return vix_future, strikes, implied_vol, option_price

    def move_tensors_to_device(self, device):
        for key, val in self.__dict__.items():
            if hasattr(val, 'device'):
                self.__dict__[key] = val.to(device)

    def move_parameters_to_device(self, device=device):
        for key in ['lam1', 'lam2', 'theta1', 'theta2', 'betas', 'timestep', 'parabolic', 'parabolic_offset']:
            self.__dict__[key] = self.__dict__[key].to(device)

    def daily_realized_volatilities(self):
        """
        computes the daily realized volatilities
        :return: torch.tensor of shape (self.maturity*252, self.N)
        """
        s = self.S_array[1:, :]  # include the first timestep of the day (eod=begin next day)
        s_per_day = s.reshape(-1, self.timestep_per_day, self.N)
        intra_day_changes = s_per_day[:, 1:] / s_per_day[:, :-1] - 1
        vol_per_day = torch.sqrt(self.timestep_per_day / dt * torch.var(intra_day_changes, dim=1))
        return vol_per_day

    def drift(self):
        R1 = self.compute_R(1).to(self.device)
        R2 = self.compute_R(2).to(self.device)
        vol = self.compute_vol(R1, R2)
        ans = -self.betas[1] * (
                    (1 - self.theta1) * self.lam1[0] * self.R1_array[:, 0] + self.theta1 * self.lam1[1] * self.R1_array[
                                                                                                          :, 1])
        ans += self.betas[2] / (2 * R2.sqrt()) * (
                    ((1 - self.theta2) * self.lam2[0] + self.theta2 * self.lam2[1]) * vol ** 2 -
                    ((1 - self.theta2) * self.lam2[0] * self.R2_array[:, 0] +
                     self.theta2 * self.lam2[1] * self.R2_array[:, 1]))
        return ans

    def compute_vix_path(self, path_id, n_subpaths=None, n_batch=1):
        if n_subpaths is None:
            n_subpaths = self.vix_N
        S0 = self.S_array[1::self.timestep_per_day, path_id]
        R1_0 = self.R1_array[1::self.timestep_per_day, :, path_id].T
        R2_0 = self.R2_array[1::self.timestep_per_day, :, path_id].T
        idxs = torch.linspace(0, S0.shape[0], steps=n_batch + 1, dtype=torch.int64)
        vix = torch.zeros(S0.shape[0])
        for i in range(len(idxs) - 1):
            size = idxs[i + 1] - idxs[i]
            S = S0[idxs[i]:idxs[i + 1]]
            R1 = R1_0[:, idxs[i]:idxs[i + 1]]
            R2 = R2_0[:, idxs[i]:idxs[i + 1]]

            S = torch.repeat_interleave(S, n_subpaths)
            R1 = torch.repeat_interleave(R1, n_subpaths, dim=1)
            R2 = torch.repeat_interleave(R2, n_subpaths, dim=1)
            _, nested_vol_array = \
                self._simulate(self.vix_steps, size * n_subpaths, S0=S, R1_0=R1, R2_0=R2,
                               save_R=False)
            nested_vol_array = nested_vol_array.reshape((self.vix_steps + 1, size, -1))

            vix[idxs[i]:idxs[i + 1]] = (nested_vol_array ** 2).mean(dim=(0, 2)).sqrt().cpu()

        return vix


if __name__ == '__main__':
    lam1 = torch.tensor([80, 10])
    lam2 = torch.tensor([40, 3.])
    betas = torch.tensor([0.04, -0.1, 0.6])
    R_init1 = torch.tensor([-0.10392006, 0.05214729])
    R_init2 = torch.tensor([0.00655017, 0.01580152])
    theta1 = 0.43
    theta2 = 0.21
    N = 50000
    vix_N = 1000

    maturity = 0.5
    torch_mc = TorchMonteCarloExponentialModel(lam1=lam1, lam2=lam2, betas=betas, R_init1=R_init1, R_init2=R_init2,
                                               theta1=theta1, theta2=theta2, N=N, vix_N=vix_N, maturity=maturity)
    torch_mc.simulate()
    option_maturity = 0.1
    strikes = np.array([0.95, 1., 1.02])
    future, _, implied_vol, option_prices = torch_mc.compute_implied_vol(strikes=strikes, option_maturity=option_maturity)
    vix_strikes = np.array([0.18, 0.2, 0.22, 0.25, 0.3])
    vix_future, _, vix_implied_vol, vix_option_price = torch_mc.compute_vix_implied_vol(vix_maturity=option_maturity, strikes=vix_strikes, subset=10000)
    
    print()
