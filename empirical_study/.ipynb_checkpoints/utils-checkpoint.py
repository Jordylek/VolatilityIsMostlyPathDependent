import os
import warnings
import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
from sklearn.linear_model import LinearRegression, RidgeCV, LassoCV

# np.set_printoptions(precision=3, floatmode='unique')
warnings.simplefilter('ignore')


def load_all_historical_prices(path='data/historical_prices_index_and_volatilities.csv'):
    return pd.read_csv(path, index_col=0, parse_dates=True)


indexes = ['spx', 'stoxx',  'ftse', 'dax', 'nikkei']  # ftse is FTSE100(or UKX)
vols = ['vix', 'vix9d', 'vstoxx',  'iviuk', 'vdax', 'vnky']
implied_vols = vols
realized_vols = [f'rv_{index}' for index in indexes]
shifts = [1, 3, 5]
future_realized_vols_per_shift = {k: [f'rv_{index}_{k}' for index in indexes] for k in shifts}
future_realized_vols = sum(list(future_realized_vols_per_shift.values()), [])
vol_to_index = {'vix': 'spx', 'vix9d': 'spx', 'vstoxx': 'stoxx',
                'iviuk': 'ftse', 'vdax': 'dax', 'vnky': 'nikkei'}
# vol_to_index.update(**{f'rv5_{index}': index for index in indexes})
vol_to_index.update(**{vol: vol.split('_')[1] for vol in future_realized_vols})

tickers = indexes + vols
tickers_to_latex = {s: s.upper() for s in tickers}
tickers_to_latex.update(**{'nikkei': 'Nikkei 225',
                           'vnky': 'Nikkei 225 VI',
                           'vdax': 'VDAX-NEW',
                           'iviuk': 'IVI'})

train_start_date = pd.to_datetime('2000-01-01')  # pd.to_datetime('2008-01-01')
vix9d_train_start_date = pd.to_datetime('2011-01-01')
test_start_date = pd.to_datetime('2019-01-01')
test_end_date = pd.to_datetime('2022-05-15')
# fut_rv21_test_end_date = pd.to_datetime('2022-01-20')
dt = 1 / 252

MODELS = {
    'BDB': {'p': 2, 'setting': [(1, 2), (2, 1)]},
    'arch_tspl': {'p': 2, 'setting': [(2, 1)]},
    'linear_R_sigma': {'p': 1, 'setting': [(1, 1), (2, 0.5)]},
    'CB': {'p': 2, 'setting': [(1, 1), (2, 1)]},
    'GJR': {'p': 2, 'setting': [(1, 1)]},
    # 'old_parent': {'p': 2, 'setting': [(1, 1)]},
    'parent': {'p': 1, 'setting': [(1, 1)]},
    'parabola_R':  {'p': 1, 'setting': [(1, (1, 2))]},
    'linear_R':  {'p': 1, 'setting': [(1, 1)]},
    'linear_sigma': {'p': 1, 'setting': [(2, 0.5)]},
    'parabola_R_linear_sigma': {'p': 1, 'setting': [(1, (1, 2)), (2, 0.5)]},
    'squared_linear_R_sigma': {'p': 2, 'setting': [(1, 1), (2, 0.5)]},
}


def convert_to_latex(feat):
    if feat in tickers:
        ans = tickers_to_latex[feat]
    elif feat.startswith('rv'):
        # Realized vols must be in the form vol_{index} or vol10_{index}/ vol5_{index}
        splitted = feat.split('_')
        if len(splitted) > 1:
            name = splitted[1]
            idx = splitted[2] if len(splitted) > 2 else ''
            idx = '' if idx=='1' else idx
        else:
            name, idx = '', ''
        ans = f'$\\mathrm{{RV}}'
        if len(name) >= 1:
            ans += f'^{{\\mathrm{{{convert_to_latex(name)}}}}}'
        if len(idx) >= 1:
            ans += f'_{{{idx}}}'
        ans += '$'  # f'$\\mathrm{{RV}}^{{\\mathrm{{{convert_to_latex(name)}}}}}_{{{idx}}}$'
    elif feat.startswith('sigma'):
        ans = feat.replace('sigma', '$\\Sigma') + '$'
    else:
        ans = '$' + feat.upper() + '$'
    rep = ans.replace('$$', '')
    if rep == '$\\Sigma_2$':
        rep = '$\\Sigma$'
    return rep


def generate_setting(p, n, sqrt):
    name = f'p={p}, n={n}'
    if sqrt:
        name += ',roots'
        setting = [(i, 1 / i) for i in range(1, n + 1)]
    else:
        setting = [(i, 1) for i in range(1, n + 1)]
    return name, {'p': p, 'setting': setting}


def create_directory(path):
    if os.path.isdir(path):
        print(f'{path} already exists')
    else:
        os.makedirs(path)


def year_dates_bound(year):
    return f'{year}-01-01', f'{int(year) + 1}-01-01'


def data_between_dates(data, start_date, end_date):
    start_date = pd.to_datetime(start_date)
    end_date = pd.to_datetime(end_date)
    return data.loc[start_date:end_date]


def split_data(data, train_start_date=train_start_date, test_start_date=test_start_date, test_end_date=test_end_date):
    train_data = data_between_dates(data, train_start_date, test_start_date)
    test_data = data_between_dates(data, test_start_date, test_end_date)
    return train_data, test_data


def negative_part(x):
    return np.clip(-x, 0, None)


def power_to(p):
    def f(x):
        if p in (-1, -2):
            return negative_part(x) ** np.abs(p)
        else:
            if p == 0:
                return x ** p
            if int(1 / p) % 2 == 1:
                return np.abs(x) ** p * np.sign(x)
            else:
                return x ** p
    return f


squared = power_to(2)
sqrt = power_to(0.5)
identity = power_to(1)


def initialize_R(lam, date=None, index=None, max_delta=500, returns=None, transform=identity, data=None):
    """
    Compute the initial value of one feature R using the price of the givenn index at the given date
    :param lam: float
    :param date: datetime, when to initialize R
    :param index: str. name of the market index (e.g spx)
    :param max_delta: int: number of business days used to initialize R
    :param returns: array. If None, uses the historical data until "date". Otherwise, uses this array to initialize R
    :param transform: function. Apply this function to each return. (e.g identity for R_1 and square for R_2)
    :param data: pd.DataFrame. If not None, uses it instead of the default historical dataset
    :return: float
    """
    if returns is None:
        assert date is not None and index is not None
        if data is None:
            data = load_all_historical_prices()
        s = data.loc[:date][index].dropna()
        returns = 1 - s.shift(1) / s
        returns = returns.values[-max_delta:][::-1]
    timestamps = np.arange(returns.shape[0]) * dt
    weights = exp_law(timestamps, lam)
    x = transform(returns)
    return np.sum(x * weights)


def power_law(t, alpha):
    return t ** (-alpha)


def shifted_power_law(t, alpha, delta):
    return (t + delta) ** (-alpha)


def exp_power_law(t, alpha, k):
    eps = 1e-8
    return np.true_divide(1 - np.exp(-(k * t) ** alpha) + eps, ((k * t) ** alpha + eps))


def exp_law(t, lam, c=1):
    return c * lam * np.exp(-lam * t)


def combined_exp_law(t, lam0, lam1, theta, c=1):
    return c * (exp_law(t, lam0, 1 - theta) + exp_law(t, lam1, theta))


def normalized_TSPL(t, alpha, delta):
    return shifted_power_law(t, alpha, delta) / (delta ** (1-alpha) / (alpha - 1))


def find_best_exp(pl_params, fit_period=126, lam=None, plot=False, nlam=2, func_power_law=normalized_TSPL, lower_lam=None):
    """
    Finds the best exponential(if nlam=1) or convex combination of exponentials(nlam=2) that fits the func_power_law
    :param pl_params:
    :param fit_period:
    :param lam:
    :param plot:
    :param nlam:
    :param func_power_law:
    :param lower_lam:
    :return:
    """
    TT = np.arange(fit_period) * dt
    shifted_pl = func_power_law(TT, **pl_params)
    # alpha, delta = pl_params['alpha'], pl_params['delta']
    # shifted_pl = shifted_pl / (delta ** (1 - alpha) / (alpha - 1))
    if nlam == 1:
        lower = np.array([0, 0])
        upper = np.array([np.inf, np.inf])
        f = exp_law
    else:
        lower = np.array([0, 0, 0, 0])
        if lower_lam is not None:
            lower[:2] = lower_lam
        upper = np.array([np.inf, np.inf, 1, np.inf])
        f = combined_exp_law
    if lam is None:
        opt_params, _ = curve_fit(f, TT, shifted_pl, bounds=(lower, upper), maxfev=4000)
    else:
        exp_l = f(TT, lam, 1)
        c = shifted_pl.sum() / exp_l.sum()
        opt_params = np.array([lam, c])

    if len(opt_params) == 2:
        ans = {'lam0': opt_params[0], 'lam1': 0, 'c': opt_params[1], 'theta': 0}
    else:
        ans = {'lam0': max(opt_params[:2]), 'lam1': min(opt_params[:2]),
               'c': opt_params[3], 'theta': opt_params[2] if opt_params[0] > opt_params[1] else 1 - opt_params[2]}
    if plot:
        import matplotlib.pyplot as plt
        pred = combined_exp_law(TT, **ans)
        plt.plot(TT, shifted_pl, label='TSPL')
        plt.plot(TT, pred, label='best fit exp')
        plt.legend()
        plt.show()
    return ans


def compute_kernel_weighted_sum(x, params, func_power_law, transform=identity, result_transform=identity):
    """

    :param x: np.array of shape (n_elements, n_timestamps). Default: returns ordered from the most recent to the oldest
    :param params: array_like of parameters of func_power_law
    :param func_power_law: callable apply the kernel on the timestamps
    :param transform: callable, applied to the values of x. Default: identity (f(x)=x)
    :param result_transform: callable, applied to the computed average. Default: identity (f(x)=x)
    :return: feature as the weighted averages of the transform(x) with weights kernel(ti)
    """
    timestamps = np.arange(x.shape[1]) * dt

    weights = func_power_law(timestamps, *params)
    x = transform(x)
    return result_transform(np.sum(x * weights, axis=1))


def dataframe_of_returns(index, vol, max_delta=1000, data=None):
    """
    constructs a dataframe where each row contains the past max_delta one-day returns from the timestamp corresponding to the index of the dataframe.
    :param index: str market index
    :param vol: str volatility index or realized vol
    :param max_delta: int number of past returns to use
    :param data: pd.DataFrame
    :return:pd.DataFrame
    """
    if data is None:
        data = load_all_historical_prices()
    df = data.loc[:, [index, vol]]
    df.dropna(subset=[index], inplace=True)  # remove closed days
    df.loc[1:, 'return_1d'] = np.diff(df[index]) / df[index].iloc[1:]
    lags = np.arange(0, max_delta)
    df = df.merge(pd.DataFrame({
        f'r_(t-{lag})': df.return_1d.shift(lag)
        for lag in lags
    }), left_index=True, right_index=True)
    return df