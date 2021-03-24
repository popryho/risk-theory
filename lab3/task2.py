import numpy as np
import pandas as pd
from scipy.special import chdtri
from scipy.stats import spearmanr
from scipy.stats import kendalltau
from const import gamma
from scipy.stats import norm


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


def spearman():
    sample_1 = np.random.uniform(0, 1, size=n)
    sample_2 = sample_1 + np.random.uniform(-1, 1, size=n)

    sample_1_sorted = np.sort(sample_1)
    sample_2_sorted = np.sort(sample_2)

    r_value = np.array([np.where(sample_1_sorted == x)[0][0] + 1 for x in sample_1])
    s_value = np.array([np.where(sample_2_sorted == y)[0][0] + 1 for y in sample_2])

    p = np.abs(1 - 6 / (n * (n ** 2 - 1)) * np.sum((r_value - s_value) ** 2))
    criterion = norm.ppf(1 - 0.05 / 2) / np.sqrt(n)

    if p > criterion:
        return f'|p| = {p:.3f}, criterion = {criterion:.3f} \n' \
               f'The statistical data do CONFLICT with the H0 hypothesis.'

    else:
        return f'|p| = {p:.3f}, criterion = {criterion:.3f} \n' \
               f'The statistical data do NOT CONFLICT with the H0 hypothesis.'


def kendall():
    sample_1 = np.random.uniform(0, 1, size=n)
    sample_2 = sample_1 + np.random.uniform(-1, 1, size=n)

    sample_1_sorted = np.sort(sample_1)
    sample_2_sorted = np.sort(sample_2)

    r_value = np.array([np.where(sample_1_sorted == x)[0][0] + 1 for x in sample_1])
    s_value = np.array([np.where(sample_2_sorted == y)[0][0] + 1 for y in sample_2])

    print(r_value, np.argsort())
    pairs = sorted(list(zip(r_value, s_value)), key=lambda x: x[0])
    V = np.array(list(zip(*pairs))[1])
    N = 0
    for i in range(n):
        subV = V[i + 1:]
        N += len(subV[V[i - 1] < subV])

    tau = np.abs((4 * N) / (n * (n - 1)) - 1)
    criterion = 2 * norm.ppf(1 - 0.05 / 2) / (3 * np.sqrt(n))

    if tau > criterion:
        return f'|tau| = {tau:.3f}, criterion = {criterion:.3f} \n' \
               f'The statistical data do CONFLICT with the H0 hypothesis.'

    else:
        return f'|tau| = {tau:.3f}, criterion = {criterion:.3f} \n' \
               f'The statistical data do NOT CONFLICT with the H0 hypothesis.'


if __name__ == '__main__':
    r, k = 10, 20
    for method in (chi_square, spearman, kendall):
        for n in map(int, (5e2, 5e3, 5e4)):
            print(f'n = {n}, {method()}')
