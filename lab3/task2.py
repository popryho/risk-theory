import numpy as np
import pandas as pd
from scipy.special import chdtri
from scipy.stats import norm, rankdata

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
        return f'r = {r}, k = {k}, delta = {delta:0.4f}, criterion = {criterion:0.4f} \n' \
               f'The statistical data do CONFLICT with the H0 hypothesis. \n'
    else:
        return f'delta = {delta:0.4f}, criterion = {criterion:0.4f} \n' \
               f'The statistical data do NOT CONFLICT with the H0 hypothesis. \n'


def spearman():
    sample_1 = np.random.uniform(0, 1, size=n)
    sample_2 = sample_1 + np.random.uniform(-1, 1, size=n)

    p = np.abs(1 - 6 / (n * (n ** 2 - 1)) * np.sum((rankdata(sample_1) - rankdata(sample_2)) ** 2))
    criterion = norm.ppf(1 - 0.05 / 2) / np.sqrt(n)

    if p > criterion:
        return f'|p| = {p:.3f}, criterion = {criterion:.3f} \n' \
               f'The statistical data do CONFLICT with the H0 hypothesis. \n'

    else:
        return f'|p| = {p:.3f}, criterion = {criterion:.3f} \n' \
               f'The statistical data do NOT CONFLICT with the H0 hypothesis. \n'


def kendall():
    sample_1 = np.random.uniform(0, 1, size=n)
    sample_2 = sample_1 + np.random.uniform(-1, 1, size=n)

    ranks = np.concatenate([rankdata(sample_1).reshape(-1, 1),
                            rankdata(sample_2).reshape(-1, 1)], axis=1)
    v = ranks[ranks[:, 0].argsort()][:, 1]
    N = int(np.sum([len(v[i + 1:][v[i - 1] < v[i + 1:]]) for i in range(n)]))

    tau = np.abs((4 * N) / (n * (n - 1)) - 1)
    criterion = 2 * norm.ppf(1 - 0.05 / 2) / (3 * np.sqrt(n))

    if tau > criterion:
        return f'|tau| = {tau:.3f}, criterion = {criterion:.3f} \n' \
               f'The statistical data do CONFLICT with the H0 hypothesis. \n'

    else:
        return f'|tau| = {tau:.3f}, criterion = {criterion:.3f} \n' \
               f'The statistical data do NOT CONFLICT with the H0 hypothesis. \n'


if __name__ == '__main__':

    r, k = 10, 20
    with open('output/output_task2.txt', 'w') as txt:
        txt.write("TASK 2")
        for idx, method in enumerate((chi_square, spearman, kendall)):
            txt.write(f'\n\nMETHOD {idx+1}: {method.__name__}\n')
            for n in map(int, (5e2, 5e3, 5e4)):
                txt.write(f'{"-" * 50}\nn = {n}, {method()}')
