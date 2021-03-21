import numpy as np
import pandas as pd
from scipy.special import chdtri
from scipy.stats import spearmanr
from scipy.stats import kendalltau
from const import gamma


def chi_square():

    sample_1 = np.random.uniform(0, 1, size=n)
    sample_2 = sample_1 + np.random.uniform(-1, 1, size=n)

    df = pd.DataFrame(data=[sample_1, sample_2]).T

    bins_x = np.linspace(sample_1.min(initial=0).round(), sample_1.max(initial=1).round(), num=r)
    bins_y = np.linspace(sample_2.min(initial=-1).round(), sample_2.max(initial=2).round(), num=k)

    df.iloc[:, 0] = bins_x[np.digitize(df.iloc[:, 0], bins=bins_x)]
    df.iloc[:, 1] = bins_y[np.digitize(df.iloc[:, 1], bins=bins_y)]

    v = pd.crosstab(df.iloc[:, 0], df.iloc[:, 1]).to_numpy()

    n_total = r + k
    delta = 0
    for i in range(v.shape[0]):
        for j in range(v.shape[1]):
            delta += ((v[i, j] - (v[i, :].sum() * v[:, j].sum()) / n_total) ** 2) / (v[i, :].sum() * v[:, j].sum())
    delta *= n_total
    criterion = chdtri((r - 1) * (k - 1), gamma)

    if delta > criterion:
        return f'delta = {delta:0.4f}, criterion = {criterion:0.4f} \n' \
               f'The statistical data do CONFLICT with the H0 hypothesis.'
    else:
        return f'delta = {delta:0.4f}, criterion = {criterion:0.4f} \n' \
               f'The statistical data do NOT CONFLICT with the H0 hypothesis.'


def spearman_test():
    sample_1 = np.random.uniform(0, 1, size=n)
    sample_2 = sample_1 + np.random.uniform(-1, 1, size=n)
    corr, critical_value = spearmanr(sample_1, sample_2)

    if critical_value > gamma:
        return f'critical_value={critical_value:.3f} \n' \
               f'The statistical data do NOT CONFLICT with the H0 hypothesis.'

    else:
        return f'critical_value={critical_value:.3f} \n' \
               f'The statistical data do CONFLICT with the H0 hypothesis.'


def kendall_test():
    sample_1 = np.random.uniform(0, 1, size=n)
    sample_2 = sample_1 + np.random.uniform(-1, 1, size=n)
    corr, critical_value = kendalltau(sample_1, sample_2)

    if critical_value > gamma:
        return f'critical_value={critical_value:.3f} \n' \
               f'The statistical data do NOT CONFLICT with the H0 hypothesis.'

    else:
        return f'critical_value={critical_value:.3f} \n' \
               f'The statistical data do CONFLICT with the H0 hypothesis.'


if __name__ == '__main__':
    r, k = 100, 100
    for method in (chi_square, spearman_test, kendall_test):
        for n in map(int, (5e2, 5e3, 5e4)):
            print(f'n = {n}, {method()}')
