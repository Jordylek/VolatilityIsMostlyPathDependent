from empirical_study.empirical_study_tspl import *


def exp_kernel(t, lam, c=1):
    return c * lam * np.exp(-lam * t)


def convex_combi_exp_kernel(t, lam0, lam1, theta, c=1):
    return c * (exp_kernel(t, lam0, 1 - theta) + exp_kernel(t, lam1, theta))


def deriv_lam0_exp_kernel(t, lam0, lam1, theta):
    return -t * (1 - theta) * exp_kernel(t, lam0)


def deriv_lam1_exp_kernel(t, lam0, lam1, theta):
    return -t * theta * exp_kernel(t, lam1)


def deriv_theta_exp_kernel(t, lam0, lam1, theta):
    return -exp_kernel(t, lam0) + exp_kernel(t, lam1)


def split_parameters_exp(parameters, setting):
    n_betas = len(setting)
    lambdas = parameters[-2 * n_betas:].reshape(-1, 2).T  # line 0 for lam_1,0 and lam_2,0
    thetas = parameters[-3 * n_betas: -2 * n_betas]
    betas = parameters[1: -3 * n_betas]
    intercept = parameters[0]
    return intercept, betas, thetas, lambdas


def optimal_parameters_from_exponentials_exp(X, y, p, setting, plot=False):
    x = X.iloc[:, 0]
    spans = np.array([1, 5, 10, 20, 120, 250])
    init_betas = []
    init_thetas = []
    init_lambdas = []

    def power_law_with_coef(t, beta, lam0, lam1, theta):
        return beta * convex_combi_exp_kernel(t, beta, lam0, lam1, theta)

    for i, j0 in setting:
        j = min(j0)
        ewms = {span: pd.Series.ewm(power_to(i)(x), span=span).mean() for span in spans}
        X_ewm = pd.DataFrame.from_dict(ewms, orient='columns')
        reg = RidgeCV()
        reg.fit(X_ewm, power_to(p / j)(y))
        coef = reg.coef_
        alphas = 2 / (1 + spans)

        timestamps = np.arange(max(spans))
        exp_kernel = (coef * alphas * (1 - alphas) ** timestamps.reshape(-1, 1)).sum(axis=1)
        exp_kernel /= exp_kernel.sum()
        try:
            opt_coef, _ = curve_fit(power_law_with_coef, timestamps * dt, exp_kernel, maxfev=4000)
        except RuntimeError:
            opt_coef = np.array([1, 60, 60, 1])
        if plot:
            pred_pl = power_law_with_coef(timestamps * dt, *opt_coef)
            plt.plot(timestamps * dt, pred_pl, label='best_fit', linestyle='--')
            plt.plot(timestamps * dt, exp_kernel, label='exp_kernel', alpha=0.5)
            plt.legend()
            plt.show()
        init_betas.extend([1] * len(j0))
        # init_betas.extend([power_to(l)(opt_coef[0]) for l in j0])
        init_thetas.append(opt_coef[3])
        init_lambdas.extend([opt_coef[1], opt_coef[2]])
    parameters = np.concatenate(([0], init_betas, init_thetas, init_lambdas))
    betas = fit_betas_exp(parameters, X_train=X, y_train=y, setting=setting)
    return np.concatenate([betas, init_thetas, init_lambdas])


def fit_betas_exp(parameters, X_train, y_train, setting):
    reg = LinearRegression()
    train_features, _ = linear_of_kernels_exp(returns=X_train, setting=setting, parameters=parameters,
                                              return_features=True)
    X_for_reg = []
    for key in train_features:
        X_for_reg.extend((list(train_features[key].values())))
    X_for_reg = np.array(X_for_reg).T
    reg.fit(X_for_reg, y_train)
    betas = np.concatenate([[reg.intercept_], reg.coef_])
    return betas


def linear_of_kernels_exp(returns, setting, parameters, return_features=False):
    splitted_parameters = split_parameters_exp(parameters=parameters, setting=setting)
    features = compute_features_exp(returns=returns, setting=setting, splitted_parameters=splitted_parameters)
    ans = splitted_parameters[0]  # intercept
    iterator = 0  # iterates over betas
    for j in range(len(setting)):
        i, p = setting[j]
        for k in p:
            ans += splitted_parameters[1][iterator] * features[i][k]
            iterator += 1

    # ans = np.clip(ans, 0, None)  # Clip between 0 and 1
    if return_features:
        return features, ans
    return ans


def compute_features_exp(returns, setting, splitted_parameters):
    intercept, betas, thetas, lambdas = splitted_parameters  # others is either deltas or kappas
    # assert len(splitted_parameters[0]) == len(setting) and len(splitted_parameters[1]) == len(setting)
    features = OrderedDict()
    for j, key in enumerate(setting):
        i, p = key
        features[i] = {k: compute_kernel_weighted_sum(x=returns, params=[lambdas[0, j], lambdas[1, j], thetas[j]],
                                                      func_power_law=convex_combi_exp_kernel, transform=power_to(i),
                                                      result_transform=power_to(k)) for k in p}
    return features


def find_optimal_parameters_exp(vol, index, p=1, setting=((1, 1), (2, 1/2)),
                                train_start_date=train_start_date, test_start_date=test_start_date,
                                test_end_date=test_end_date, max_delta=1000, fixed_initial=False, use_jacob=True,
                                non_negative_beta=False, min_lam=None, max_lam=None, min_theta=None, max_theta=None):

    """
    Computes the optimal parameters to linearly estimate vol^p using the previous returns of index using the convex combination of exponentials
    :param vol: str. Name of the predicted volatility
    :param index: str. Name of the market index
    :param p: int (usually 1 or 2). Target of the prediction of vol^p
    :param setting: list of tuples. Each tuple is either a (i,j) or (i, (j1, dots, jk)).
    This means that each R_i^{j_l} is a feature of the regression, where R_i= \sum_t K(t) r_t^i
    :param train_start_date: datetime. Default May 15 2012. When to start the train dataset
    :param test_start_date: datetime. Default Jan 01 2019. When to start the test dataset
    :param test_end_date: datetime. Default May 15 2022. When to end the test dataset
    :param max_delta: int, default 1000. Number of days used to compute the past returns for each day
    :param fixed_initial: bool. If True, uses the initial parameters given in init_parameters
    :param use_jacob: bool If True, uses the analytical jacobian. Otherwise, it is estimated by the function.
    :param non_negative_beta: bool. Only valid for the parabolic model. Ensures that the beta in front for R_1^2 is positive.
    :return: dictionary containing the solution from the scipy optimization, the optimal parameters, the features on the train and test set,
    the train and test r2 and RMSE, the prediction on the train and test set
    """
    setting = [(i, p if isinstance(p, Iterable) else (p,)) for i, p in setting]  # turn the single values into tuples
    # If it's just one value, transform it to tuple

    df = dataframe_of_returns(index=index, vol=vol, max_delta=max_delta)
    # Remove NaNs on volatility (happens for vstoxx for some reason)

    train_data, test_data = split_data(df, train_start_date=train_start_date, test_start_date=test_start_date,
                                       test_end_date=test_end_date)
    train_data.dropna(inplace=True)
    test_data.dropna(inplace=True)
    cols = [f'r_(t-{lag})' for lag in range(max_delta)]
    X_train = train_data.loc[:, cols]
    X_test = test_data.loc[:, cols]
    vol_train = train_data['vol']
    vol_test = test_data['vol']
    target_transform = lambda x: power_to(p)(x)
    inv_target_transform = lambda x: power_to(1 / p)(x)

    y_train = target_transform(vol_train)
    norm_coeff = 1  # ** p
    y_train = y_train / norm_coeff
    y_test = target_transform(vol_test) / norm_coeff

    def residuals(parameters): # , return_diff=False):
        res = - y_train + linear_of_kernels_exp(returns=X_train, setting=setting, parameters=parameters)
        return res
        # if return_diff:
        #     return res
        # return 1/2 * np.sum(res ** 2)

    def jacobian(parameters):
        train_features, pred_train = linear_of_kernels_exp(returns=X_train, setting=setting, parameters=parameters,
                                                           return_features=True)
        # train_features containts the R_i^j
        splitted_parameters = split_parameters_exp(parameters=parameters, setting=setting)
        intercept, betas, thetas, lambdas = splitted_parameters  # others is either deltas or kappas

        jacob = np.zeros((len(parameters), len(y_train)))
        jacob[0] = 1  # For the intercept

        # df/dbeta
        iter = 1
        for i, p in setting:
            for j in p:
                jacob[iter] = train_features[i][j]
                iter += 1

        theta_jac = np.zeros((n_alphas, len(y_train)))
        lambda_jac = np.zeros((2 * n_alphas, len(y_train)))
        sub_iter = 0  # iterates on the betas
        for iter, (i, p) in enumerate(setting):
            R_i = power_to(1 / p[0])(train_features[i][p[0]])
            params_i = [lambdas[0, iter], lambdas[1, iter], thetas[iter]]
            dR_i_dtheta = compute_kernel_weighted_sum(x=X_train, params=params_i,
                                                      func_power_law=deriv_theta_exp_kernel, transform=power_to(i),
                                                      result_transform=identity)

            dR_i_dlam0 = compute_kernel_weighted_sum(x=X_train, params=params_i,
                                                     func_power_law=deriv_lam0_exp_kernel, transform=power_to(i),
                                                     result_transform=identity)

            dR_i_dlam1 = compute_kernel_weighted_sum(x=X_train, params=params_i,
                                                     func_power_law=deriv_lam1_exp_kernel, transform=power_to(i),
                                                     result_transform=identity)
            coeff = np.full_like(y_train, 0)
            for j in p:
                coeff += j * betas[sub_iter] * power_to(j - 1)(R_i)
                sub_iter += 1
            theta_jac[iter] = coeff * dR_i_dtheta
            lambda_jac[2 * iter] = coeff * dR_i_dlam0
            lambda_jac[2 * iter + 1] = coeff * dR_i_dlam1

        jacob[-2 * n_alphas:] = lambda_jac
        jacob[-3 * n_alphas: -2 * n_alphas] = theta_jac
        return jacob.T
        # diff = residuals(parameters, return_diff=True)

        # return jacob @ diff

    size_parameters = 1 + np.sum([len(p) + 3 for i, p in setting])
    lower_bound = np.full(size_parameters, -np.inf)
    upper_bound = np.full(size_parameters, np.inf)
    n_alphas = len(setting)

    initial_parameters = np.full(size_parameters, 1.)
    if not fixed_initial:
        initial_parameters = optimal_parameters_from_exponentials_exp(X=X_train, y=y_train, p=p, setting=setting, plot=False)

    # # This upper bound is not good enough. If value is
    lower_bound[-2 * n_alphas:] = 0  # force non-negative lambdas

    # theta in 0,1
    lower_bound[-3 * n_alphas:-2 * n_alphas] = 0
    upper_bound[-3 * n_alphas:-2 * n_alphas] = 1
    # Non negative deltas

    if non_negative_beta:
        lower_bound[2] = 0  # Keep the beta of R_1^2 non negative. Only valid for parabole_R_linear_sigma
        initial_parameters[2] = 1e-6

    min_lam_array = []
    max_lam_array = []
    min_theta_array = []
    max_theta_array = []

    min_lam = {} if min_lam is None else min_lam
    max_lam = {} if max_lam is None else max_lam
    min_theta = {} if min_theta is None else min_theta
    max_theta = {} if max_theta is None else max_theta
    for i, _ in setting:
        min_lam_array.extend(min_lam.get(i, [0, 0]))
        max_lam_array.extend(max_lam.get(i, [np.inf] * 2))
        min_theta_array.append(min_theta.get(i, 0))
        max_theta_array.append(max_theta.get(i, 1))

    # lambda bounds$
    lower_bound[-2 * n_alphas:] = np.array(min_lam_array)
    upper_bound[-2 * n_alphas:] = np.array(max_lam_array)

    # theta bounds
    lower_bound[-3 * n_alphas:-2 * n_alphas] = np.array(min_theta_array)
    upper_bound[-3 * n_alphas:-2 * n_alphas] = np.array(max_theta_array)

    initial_parameters = np.clip(initial_parameters, lower_bound, upper_bound)

    initial_pred_train = linear_of_kernels_exp(returns=X_train, setting=setting, parameters=initial_parameters,
                                               return_features=False)
    initial_pred_test = linear_of_kernels_exp(returns=X_test, setting=setting, parameters=initial_parameters,
                                              return_features=False)

    initial_pred_train = np.clip(initial_pred_train, 0, None)
    initial_pred_test = np.clip(initial_pred_test, 0, None)
    initial_vol_pred_train = inv_target_transform(initial_pred_train * norm_coeff)
    initial_vol_pred_test = inv_target_transform(initial_pred_test * norm_coeff)
    # else:
    #     initial_parameters = initial_parameters[:2*len(setting)+1]
    jacob = jacobian if use_jacob else '2-point'

    sol = least_squares(residuals, initial_parameters, method='trf', bounds=(lower_bound, upper_bound), jac=jacob)
    # constraints = {'type': 'ineq', 'fun': constraint_lam_theta, 'args': (min_lam_array, ), 'jac': jac_constrains}
    #                # 'jac': jac_constrains}
    # bounds = [(lower_bound[i], upper_bound[i]) for i in range(len(lower_bound))]
    # sol = minimize(residuals, initial_parameters, method='SLSQP', bounds=bounds, constraints=constraints, jac=jacob)
    opt_params = sol['x']
    split_opt_params = split_parameters_exp(parameters=opt_params, setting=setting)
    split_opt_params = list(split_opt_params)
    train_features, pred_train = linear_of_kernels_exp(returns=X_train, setting=setting, parameters=opt_params,
                                                       return_features=True)

    test_features, pred_test = linear_of_kernels_exp(returns=X_test, setting=setting, parameters=opt_params,
                                                     return_features=True)
    pred_train = np.clip(pred_train, 0, None)
    pred_test = np.clip(pred_test, 0, None)
    vol_pred_train = inv_target_transform(pred_train * norm_coeff)
    vol_pred_test = inv_target_transform(pred_test * norm_coeff)
    split_opt_params[0] = split_opt_params[0] * norm_coeff
    split_opt_params[1] = split_opt_params[1] * norm_coeff  # Add correction term to betas
    train_features = OrderedDict([(key, pd.DataFrame(train_features[key])) for key in train_features])
    test_features = OrderedDict(
        [(key, pd.DataFrame(test_features[key])) for key in test_features])
    iter = 0
    importance = np.full_like(split_opt_params[1], 0)
    for key in train_features:
        size = train_features[key].shape[1]
        importance[iter:iter + size] = split_opt_params[1][iter: iter + size] * train_features[key].std().values
        iter += size

    keys = ['beta_0', 'betas', 'thetas', 'lambdas']
    features = ordered_dict_to_dataframe(train_features, test_features, setting)
    ans = {'sol': sol, 'opt_params': {keys[i]: split_opt_params[i] for i in range(len(keys))},
           'setting': setting, 'p': p,
           'norm_coef': norm_coeff, 'importance': importance,
           'train_pred': pd.Series(vol_pred_train, index=train_data.index),
           'test_pred': pd.Series(vol_pred_test, index=test_data.index),
           'train_rmse': mean_squared_error(y_true=vol_train, y_pred=vol_pred_train, squared=False),
           'test_rmse': mean_squared_error(y_true=vol_test, y_pred=vol_pred_test, squared=False),
           'train_r2': r2_score(y_true=vol_train, y_pred=vol_pred_train),
           'test_r2': r2_score(y_true=vol_test, y_pred=vol_pred_test),
           'features': features,
           'train_features': train_features,
           'test_features': test_features,
           'initial_parameters': {keys[i]: split_parameters_exp(parameters=initial_parameters, setting=setting)[i] for i in
                                  range(len(keys))},
           'initial_train_rmse': mean_squared_error(y_true=vol_train, y_pred=initial_vol_pred_train, squared=False),
           'initial_test_rmse': mean_squared_error(y_true=vol_test, y_pred=initial_vol_pred_test, squared=False),
           'initial_train_r2': r2_score(y_true=vol_train, y_pred=initial_vol_pred_train),
           'initial_test_r2': r2_score(y_true=vol_test, y_pred=initial_vol_pred_test),
           }
    return ans


def main():
    global_data = load_all_historical_prices()
    vol = global_data['vix']
    index = global_data['spx']
    # vol = 'rv5_spx'
    # index = 'spx'  # vol_to_index[vol]
    func_power_law = convex_combi_exp_kernel
    train_start = train_start_date
    test_start = test_start_date
    test_end = test_end_date
    max_delta = 1000
    fixed_initial = False
    use_jacob = True
    name = 'linear_R_sigma'
    model = MODELS[name]
    p = model['p']
    setting = model['setting']

    non_negative_beta = (name == 'parabole_R_linear_sigma')
    min_lam = {1: [0, 0], 2: [0, 0]}
    max_lam = {1: [200, np.inf], 2: [200, np.inf]}
    min_theta = {1: 0, 2: 0}
    max_theta = {1: 1, 2: 1}

    sol = find_optimal_parameters_exp(vol, index, p=1, setting=setting,
                                      train_start_date=train_start, test_start_date=test_start,
                                      test_end_date=test_end, max_delta=max_delta, fixed_initial=fixed_initial,
                                      use_jacob=use_jacob, non_negative_beta=non_negative_beta, min_lam=min_lam,
                                      max_lam=max_lam, min_theta=min_theta, max_theta=max_theta)
    ex = {s: sol[s] for s in ['train_r2', 'test_r2', 'opt_params']}
    print(ex)
    train_data, test_data = split_data(global_data)
    fig = plt.figure(figsize=(8, 6))
    ax = plt.gca()
    linewidth = 1
    sol['test_pred'].plot(label='prediction', ax=ax, alpha=0.8, linewidth=linewidth)
    y_vol = test_data[vol].dropna()
    y_target = (y_vol + y_vol.shift(1) + y_vol.shift(-1)) / 3
    y_target.plot(label='True', ax=ax, alpha=0.8, color='k', linestyle='--', linewidth=linewidth)
    ax.grid(alpha=0.4)
    ax.legend()
    ax.set_ylabel(convert_to_latex(vol))
    plt.show()
    return sol


def main2():
    vol = 'vix'
    index = vol_to_index[vol]

    values_of_p = [10, 30, 60, 120, 250, 1000]
    sol = fit_arch_model(vol, index, values_of_p=values_of_p)
    return sol

if __name__ == '__main__':
    sol = main()