from .torch_montecarlo import *

DELTAS = torch.arange(1, 11, 1)
Q_VALUES = torch.tensor([0, 0.5, 1, 1.5, 2, 3])


def compute_moment(vol_timeseries, q, delta):
    """
    compute the estimate of the moment of order q of the log volatility.
    :param vol_timeseries:
    :param q:
    :param delta:
    :return: float
    """
    measures = torch.zeros(delta)
    for i in range(delta):
        values = torch.log(vol_timeseries[i::delta])
        m = torch.mean(torch.abs(torch.diff(values, dim=0)) ** q)
        measures[i] = m
    return measures.mean()


def compute_hurst_estimate(vol_timeseries, deltas=DELTAS, qs=Q_VALUES, plot=True, plot_title=''):
    """
    Use the methodoly from "The Volatility is Rough"
    :param vol_timeseries: torch.tensor timeseries of volatilites
    :param deltas: different values of deltas
    :param qs: different value of qs
    :param plot: bool, if display the plot of log m vs log delta and zeta vs log q
    :param plot_title: str, tit
    :return: float, Hurst exponent estimate
    """
    moment_matrix = torch.zeros((deltas.shape[0], qs.shape[0]))
    for i, delta in enumerate(deltas):
        for j, q in enumerate(qs):
            moment_matrix[i, j] = compute_moment(vol_timeseries, q, delta)
    slopes = np.zeros(len(qs))
    for j, q in enumerate(qs):
        if q == 0:
            continue
        reg = LinearRegression()
        X = torch.log(deltas).reshape(-1, 1).numpy()
        y = torch.log(moment_matrix)[:, j].numpy()
        reg.fit(X, y)
        slopes[j] = reg.coef_[0]

    if plot:
        fig1 = plt.figure(figsize=figsize)
        for j, q in enumerate(qs):
            if q == 0:
                continue
            sns.regplot(torch.log(deltas).numpy(), torch.log(moment_matrix)[:, j].numpy(), label=f'$q={q}$', ci=None,
                        scatter_kws={'s': 40, 'alpha': 0.6}, line_kws={'linewidth': 1.5})
        plt.legend()
        plt.ylabel('$\\log(m(q, \\Delta))$', fontsize=xbigfontsize)
        plt.xlabel('$\\log(\\Delta)$', fontsize=xbigfontsize)
        plt.title(plot_title, fontsize=fontsize)

        plt.grid(alpha=0.3)

    reg = LinearRegression()
    reg.fit(qs.numpy().reshape(-1, 1), slopes)
    hurst = reg.coef_[0]

    if plot:
        fig2 = plt.figure(figsize=figsize)
        plt.plot(qs, slopes, label='$\\zeta_q$', alpha=0.7)
        plt.plot(qs, reg.coef_[0] * qs, label=f'${reg.coef_[0]:.3f}\\times q$', alpha=0.8, linestyle='--')
        plt.legend(fontsize=fontsize)
        plt.ylabel('$\\zeta_q$', fontsize=bigfontsize)
        plt.xlabel('$q$', fontsize=bigfontsize)
        plt.title(plot_title, fontsize=fontsize)
        plt.grid(alpha=0.4)
    return hurst


