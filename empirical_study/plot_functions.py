import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.patches import Patch
from mpl_toolkits import mplot3d
import seaborn as sns
import itertools
from sklearn.linear_model import LinearRegression
import copy
import numpy as np
import pandas as pd
from empirical_study.utils import dt, shifted_power_law
import statsmodels.api as sm

lowess = sm.nonparametric.lowess


plt.rc('figure', dpi=120)  # was 250
xsmallfontsize = 7
smallfontsize = 10
fontsize = 15
bigfontsize = 20
xbigfontsize = 25
xxbigfontsize = 30
# fontsize = 20
smallfigsize = (8, 6)
figsize = (12, 8)
subplot_figsize = (14, 6)
bigfigsize = (16, 8)  # necessary for plot with histogram
# plt.rcParams["figure.figsize"] = figsize
# mpl.rc('xtick', labelsize=bigfontsize)
# mpl.rc('ytick', labelsize=bigfontsize)
plt.rcParams.update(**{'figure.figsize': figsize, 'lines.linewidth': 1, 'lines.markersize': 3,
                       'legend.fontsize': bigfontsize, 'axes.grid': True, 'grid.alpha': 0.4,
                       'xtick.labelsize': bigfontsize, 'ytick.labelsize': bigfontsize, 'legend.framealpha': 0.5,
                       'figure.dpi': 120, 'hist.bins': 20, 'axes.labelsize': xbigfontsize,
                       'axes.titlesize': xxbigfontsize})


def plot_prediction_timeseries(y_pred, y_target, index=None, color_pred='r', color_target='k',
                               color_index='b', figsize=figsize):
    """
    plot the timeseries of prediction and target
    :param y_pred: pd.Series. Timeseries of the predictions of the vol
    :param y_target: pd.Series. Timeseries of the true vol
    :param index: pd.Series or None. if not None, add a plot of the index prices
    :param color_pred: color of the prediction
    :param color_target: color of the target
    :param color_index: color of the index if not None
    :param figsize: tuple of int. Size of the figure
    :return: a matplotlib.Figure object
    """
    fig = plt.figure(figsize=figsize)
    ax = plt.gca()
    if index is not None:
        ax2 = ax.twinx()
        label_index = '$S_t$'
        ax2.set_ylabel(label_index, color=color_index, fontsize=xbigfontsize)
        ax2.tick_params(axis='y', colors=color_index)
        index.plot(ax=ax2, color=color_index, alpha=0.5, linewidth=1, linestyle='-.')
        ax2.grid(False)
    ax.set_ylabel('Volatility', fontsize=xbigfontsize)
    y_target.plot(ax=ax, color=color_target, label='Target')
    y_pred.plot(linestyle='-', alpha=0.7, ax=ax, color=color_pred, label='Prediction')
    # plot the legend
    ax.legend(loc='upper left', fontsize=bigfontsize, framealpha=0.4)
    ax.set_xlabel('')
    # handles is a list, so append manual patch
    # ax.grid(alpha=0.6)
    return fig


def plot_prediction_vs_true(y_pred, y_target, color_pred='r',
                            plot_residuals=False, ratio_residuals=True,
                            add_kernel_plot=False, lowess_frac=0.25, ylim=None,  figsize=figsize):
    """
    Plot a scatterplot of predicted value against true values. If plot_residuals  is True, plots the residuals instead.
    :param y_pred: pd.Series. Timeseries of the predictions of the vol
    :param y_target: pd.Series. Timeseries of the true vol
    :param color_pred: color of the points
    :param plot_residuals: bool. if True plot the residuals instead
    :param ratio_residuals: bool. if True, plot the residuals True/prediction. Otherwise, plot True - Predicted
    :param add_kernel_plot: bool. If True, add E[Y|X] using lowess
    :param lowess_frac: float. Fraction of points used for lowess regression( if add_kernel_plot is True)
    :param ylim: None or tuple of floats. Limits on the float for the y-axis
    :param figsize: Tuple of int. Size of the figure
    :return:
    """
    orig_line_kws = {'linewidth': 1, 'color': 'C0', 'alpha': 1}
    if plot_residuals:
        if ratio_residuals:
            residuals = y_target / y_pred
            ylabel = f'True / Predicted'
            hline = 1
        else:
            residuals = y_target - y_pred
            ylabel = f'True - Predicted'
            hline = 0
        g = sns.jointplot(x=y_target, y=residuals, alpha=0.3, s=50,
                          color=color_pred)
        y = residuals
        g.ax_marg_x.remove()
        g.ax_marg_x.grid(False)
        g.ax_marg_y.grid(False)
        fig = plt.gcf()
        fig.set_size_inches((12, 8))
        ax = g.ax_joint
        ax.spines["top"].set_visible(True)  # putting back the borders
        ax.spines["right"].set_visible(True)
        ax.axhline(hline, linestyle='--', color='k', alpha=0.5)
        ax.set_ylabel(ylabel, fontsize=xbigfontsize)
    else:
        fig = plt.figure(figsize=figsize)
        ax = plt.gca()
        ax.scatter(y_target, y_pred, alpha=0.4, s=25, color=color_pred)
        x = min(y_target.min(), y_pred.min()), max(y_target.max(), y_pred.max())
        ax.axline((x[0], x[0]), (x[1], x[1]), color='k', alpha=0.7, linestyle='--')
        ax.set_ylabel(f'Predicted', fontsize=xbigfontsize)
        y = y_pred
    ax.set_xlabel(f'True', fontsize=xbigfontsize)

    if add_kernel_plot:
        x, y = lowess(y, y_target, frac=lowess_frac, return_sorted=True).T
        ax.plot(x, y, **orig_line_kws)
    if ylim is not None:
        ax.set_ylim(*ylim)
    return fig


def plot_vol_against_feature(feature, target, feature_name=None, color='r', figsize=figsize, add_kernel_plot=True,
                             scatter_kws=None, line_kws=None, color_variable=None, cmap='YlOrRd', lowess_frac=0.25,
                             ax=None, add_vertical_kernel_plot=False, color_label=None):
    """
    Plot a scatterplot of the "target" against the feature. The target is usually the predicted volatility
    :param feature: pd.Series of same size as target. Timeseries of features
    :param target: pd.Series of same size as feature. Timeseries of targets
    :param feature_name: None or str. Used as x_label.
    :param color: color
    :param figsize: figsize
    :param add_kernel_plot: bool.If True, adds E[Y|X] using lowess
    :param scatter_kws: None or dict. keywords for the scatterplot (see plt.scatter)
    :param line_kws:  None or dict. keywords for the eventual kernel regression plot. (see plt.plot)
    :param color_variable: None or pd.Series of same size as target. if not None, uses this variable as the colormap
    :param cmap: colormap if color_variable is specified
    :param lowess_frac: float. Fraction used for lowess E[Y|X] or E[X|Y]
    :param ax: matplotlib.Axes object or None.
    :param add_vertical_kernel_plot: bool. If True, adds E[X|Y] using lowess
    :param color_label: None or str. Used as colorbar label if color_variable is specified
    :return: matplotlib Figure

    """
    if line_kws is None:
        line_kws = {}
    if scatter_kws is None:
        scatter_kws = {}
    if feature_name is None:
        feature_name = 'Feature'
    orig_scatter_kws = {'alpha': 0.5, 'color': color, 's': 20}
    orig_line_kws = {'linewidth': 1, 'color': 'C0', 'alpha': 1}
    orig_scatter_kws.update(**scatter_kws)
    orig_line_kws.update(**line_kws)

    if ax is None:
        fig = plt.figure(figsize=figsize)
        ax = plt.gca()

    else:
        fig = plt.gcf()
    if color_variable is not None:
        orig_scatter_kws.pop('color')
        plot = ax.scatter(feature, target, c=color_variable, cmap=cmap, **orig_scatter_kws)
        cbar = plt.colorbar(plot)
        color_label = 'color_feature' if color_label is None else color_label
        cbar.ax.set_ylabel(color_label, fontsize=xbigfontsize)
    else:
        ax.scatter(feature, target, **orig_scatter_kws)
    if add_kernel_plot:
        x, y = lowess(target, feature, frac=lowess_frac, return_sorted=True).T
        ax.plot(x, y, **orig_line_kws)
    if add_vertical_kernel_plot:
        x, y = lowess(feature, target, frac=lowess_frac, return_sorted=True).T
        ax.plot(y, x, linewidth=orig_line_kws['linewidth'], color='k', linestyle='--', alpha=orig_line_kws['alpha'])
    ax.set_ylabel('Volatility', fontsize=xbigfontsize)
    ax.set_xlabel(feature_name, fontsize=xbigfontsize)
    # ax.grid(alpha=0.4)
    return fig


def plot_timeseries(y, label, score=0, display_score=True, secondary=None, secondary_label=None, figsize=figsize,
                    color='r', color_secondary='k', hline=1, add_hline=True, ylim=None):
    """
    plot the timeseries of y
    :param y: pd.Series. Timeseries
    :param label: str. label of y
    :param score: float. r2 score to display if display_score is True
    :param display_score: bool. If True, display score
    :param secondary: None or pd.Series. if not None, also plot the timeseries of secondary with an axis on the right size
    :param secondary_label: str. name of the secondary timeseries
    :param figsize: tuple of float figure size
    :param color: color of y
    :param color_secondary: color of secondary
    :param hline: float. coordinate of horizontal line if any
    :param add_hline: bool. Adds a horizontal line at hline
    :param ylim: tuple. limit of y-axis
    :return: matplotlib.Figure
    """
    fig = plt.figure(figsize=figsize)
    ax = plt.gca()
    ax.set_ylabel(label, fontsize=bigfontsize)
    y.plot(ax=ax, color=color, label=label)
    handles, labels = ax.get_legend_handles_labels()
    if add_hline:
        ax.axhline(hline, linestyle='--', color='C0', alpha=0.5)
    if display_score:
        patch = Patch(facecolor='w', edgecolor='w', label=f'Test $r^2 = {score:.3f}$')
        handles.append(patch)
    if secondary is not None:
        ax2 = ax.twinx()
        ax2.set_ylabel(secondary_label, color=color_secondary, fontsize=xbigfontsize)
        ax2.tick_params(axis='y', colors=color_secondary)
        secondary.plot(ax=ax2, color=color_secondary, alpha=0.5, linewidth=1, linestyle='-.', label=secondary_label)
        patch2, _ = ax2.get_legend_handles_labels()
        handles.extend(patch2)
        ax2.grid(False)
    ax.legend(handles=handles, loc='upper left', fontsize=xbigfontsize, framealpha=0.4)
    if ylim is not None:
        ax.set_ylim(*ylim)
    ax.set_xlabel('')
    return fig


def hex_to_rgb(value):
    '''
    Converts hex to rgb colours
    value: string of 6 characters representing a hex colour.
    Returns: list length 3 of RGB values'''
    value = value.strip("#")  # removes hash symbol if present
    lv = len(value)
    return tuple(int(value[i:i + lv // 3], 16) for i in range(0, lv, lv // 3))


def rgb_to_dec(value):
    '''
    Converts rgb to decimal colours (i.e. divides each value by 256)
    value: list (length 3) of RGB values
    Returns: list (length 3) of decimal values'''
    return [v / 256 for v in value]


def get_continuous_cmap(hex_list, float_list=None):
    ''' creates and returns a color map that can be used in heat map figures.
        If float_list is not provided, colour map graduates linearly between each color in hex_list.
        If float_list is provided, each color in hex_list is mapped to the respective location in float_list.

        Parameters
        ----------
        hex_list: list of hex code strings
        float_list: list of floats between 0 and 1, same length as hex_list. Must start with 0 and end with 1.

        Returns
        ----------
        colour map'''
    rgb_list = [rgb_to_dec(hex_to_rgb(i)) for i in hex_list]
    if float_list:
        pass
    else:
        float_list = list(np.linspace(0, 1, len(rgb_list)))

    cdict = dict()
    for num, col in enumerate(['red', 'green', 'blue']):
        col_list = [[float_list[i], rgb_list[i][num], rgb_list[i][num]] for i in range(len(float_list))]
        cdict[col] = col_list
    cmp = mpl.colors.LinearSegmentedColormap('my_cmp', segmentdata=cdict, N=256)
    return cmp


def save_figure(fig, filepath):
    if not filepath.endswith('.pdf'):
        filepath = filepath + '.pdf'
    fig.savefig(filepath, bbox_inches='tight', format='pdf')


hexlist = ['7678ed', '70e000', 'e01e37']
cmap = get_continuous_cmap(hexlist)


def plot_3d(X, Y, Z, N_points_per_ax=20, view=(10, 35), tickpad=-5, labelpad=15, coeffs=None,
            xlabel=None, ylabel=None, zlabel=None, cbar_location='right', cbar_label=None):
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(projection='3d')
    N = N_points_per_ax
    X_plan = np.linspace(min(X), max(X), N)
    Y_plan = np.linspace(min(Y), max(Y), N)
    X_1, Y_1 = np.meshgrid(X_plan, Y_plan)
    X_1 = X_1.reshape(-1)
    Y_1 = Y_1.reshape(-1)
    if coeffs is None:
        reg = LinearRegression()
        reg.fit(np.stack([X, Y]).T, Z)
        coeffs = {'beta_0': reg.intercept_, 'beta_1': reg.coef_[0], 'beta_2': reg.coef_[1]}
    Z_plan = coeffs['beta_0'] + coeffs['beta_2'] * Y_1 + coeffs['beta_1'] * X_1
    Z_pred = coeffs['beta_0'] + coeffs['beta_2'] * Y + coeffs['beta_1'] * X

    color_z = Z - Z_pred  # shift range to between -2 and 5

    center = 0
    divnorm = mpl.colors.TwoSlopeNorm(vmin=color_z.min(), vcenter=center, vmax=color_z.max())

    g = ax.scatter(X, Y, Z, marker='o', c=color_z, s=10, cmap=cmap, alpha=0.7, norm=divnorm)
    ax.plot_trisurf(X_1, Y_1, Z_plan, alpha=0.4, color='grey')
    ax.set_xlabel(xlabel, fontsize=xbigfontsize, labelpad=labelpad)
    ax.set_ylabel(ylabel, fontsize=xbigfontsize, labelpad=labelpad)
    ax.zaxis.set_rotate_label(False)  # disable automatic rotation
    ax.set_zlabel(zlabel, fontsize=xbigfontsize, labelpad=labelpad, rotation=90)
    cbar = plt.colorbar(g, shrink=0.7) # , location=cbar_location)
    cbar_label = f'True - Predicted' if cbar_label is None else cbar_label
    cbar.ax.set_ylabel(cbar_label, fontsize=xbigfontsize)
    cbar.ax.tick_params(axis='both', labelsize=fontsize)
    ax.view_init(*view)
    ax.tick_params(axis='both', labelsize=fontsize)
    ax.set_xticklabels(np.round(ax.get_xticks(), 1), rotation=-30,
                       va='top',
                       ha='left'
                       )
    ax.set_yticklabels(np.rint(ax.get_yticks()).astype(int), rotation=20,
                       va='top',
                       ha='right'
                       )
    ax.tick_params(axis='x', pad=tickpad, labelleft=True)
    ax.tick_params(axis='y', pad=tickpad)
    return fig


default_t_array = np.linspace(0, 1, 1000)
def plot_kernel(alpha, delta, label=None, t_array=None, ax=None, color='C0', quantile=0.9,
                func_power_law=shifted_power_law, plot_kwargs=None):
    if t_array is None:
        global default_t_array
        t_array = default_t_array

    if plot_kwargs is None:
        plot_kwargs = {}
    if ax is None:
        ax = plt.gca()
    y_array = func_power_law(t_array, alpha, delta)
    y_array /= (y_array * dt).sum()
    # print(y_array.sum() * dt, y_array.mean())
    ax.plot(t_array, y_array, label=label, alpha=0.75, color=color, **plot_kwargs)
    return t_array[np.where(np.cumsum(y_array) >= quantile)][0]


if __name__ == '__main__':
    train_start_date = pd.to_datetime('2015-01-01')
    test_start_date = pd.to_datetime('2019-01-01')
    test_end_date = pd.to_datetime('2022-05-15')

    spx = pd.read_csv('../data/spx.csv', index_col=0, parse_dates=True)['Close/Last'].sort_index()
    vix = pd.read_csv('../data/vix.csv', index_col=0, parse_dates=True)['Adj Close'].sort_index() / 100

    max_delta = 500
    fixed_initial = False
    use_jacob = True
    func_power_law = shifted_power_law
    optimize_delta = True
    test_start = test_start_date
    test_end = test_end_date
    train_start = train_start_date
    use_alternate = False

    p = 1
    setting = [(1, (1, 2)), (2, 1 / 2)]
    from empirical_study.empirical_study_tspl import find_optimal_parameters_tspl,data_between_dates
    sol_tspl = find_optimal_parameters_tspl(vol=vix, index=spx, p=p, setting=setting,
                                            optimize_delta=optimize_delta, train_start_date=train_start,
                                            test_start_date=test_start, test_end_date=test_end,
                                            max_delta=max_delta, fixed_initial=fixed_initial, use_jacob=use_jacob)

    vix2 = data_between_dates(vix, start_date=test_start_date, end_date=test_end_date)
    # ploty_pred=sol_tspl['test_pred'], y_target=vix2, ratio_residuals=True, plot_residuals=False)
    plt.show()
    plot_3d(sol_tspl['features']['R_1'], sol_tspl['features']['R_2^0.5'], vix2)
