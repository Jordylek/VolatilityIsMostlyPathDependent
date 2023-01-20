from empirical_study.empirical_study_tspl import find_optimal_parameters_tspl
from empirical_study.empirical_study_2exp import find_optimal_parameters_exp
from empirical_study.utils import dataframe_of_returns, train_start_date, test_start_date, test_end_date
import pandas as pd
import numpy as np


def perform_empirical_study(index, vol, setting=((1, 1), (2, 1 / 2)), tspl=True, p=1,
                            train_start_date=train_start_date, test_start_date=test_start_date,
                            test_end_date=test_end_date, max_delta=1000):
    """
    Find the best parameters for the model defined by setting and p (see `historical_analysis.ipynb` on how it is defined)
    :param index: pd.Series. Timeseries of historical price of the index
    :param vol: pd.Series. Timeseries of historical price of the volatility
    :param setting: tuple of tuples. Defines the model
    :param tspl: bool. If True, the kernel is a timeshifted powerlaw,otherwise it is a convex combination of two exponentials
    :param p: float. the model will fit vol^p
    :param train_start_date: date. First date used for training the model. Note that it must be bigger than the smaller date of index + max_delta business days
    :param test_start_date: date. First date of test set
    :param test_end_date: date. Last date of test set
    :param max_delta: int. number of business days used to computed the weighted averages of past returns.
    :return: a dictionary containing the scores, optimal parameters, weighted averages of past returns and predictions on both the train and test set.
    """
    learner = find_optimal_parameters_tspl if tspl else find_optimal_parameters_exp

    sol = learner(index=index, vol=vol, setting=setting, train_start_date=train_start_date, test_start_date=test_start_date,
                  test_end_date=test_end_date, max_delta=max_delta, fixed_initial=False, use_jacob=True,p=p)

    return {key: sol[key] for key in ['train_r2', 'test_r2', 'train_rmse', 'test_rmse',
                                      'features', 'opt_params', 'train_pred', 'test_pred']}
