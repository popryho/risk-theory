import numpy as np
from scipy.special import chdtri
from scipy.stats import norm

from const import gamma


def empty_cell(alpha):
    sample_1 = np.sort(np.random.exponential(scale=1, size=n))
    sample_2 = np.random.exponential(scale=1 / alpha, size=m)
    k = 0
    for i in range(1, sample_1.shape[0]):
        mask = np.logical_and(sample_2 > sample_1[i - 1], sample_1[i] > sample_2)
        k += 1 if sample_2[mask].shape[0] == 0 else 0

    ro = m / n
    criterion = n / (1 + ro) + np.sqrt(n) * ro * norm.ppf(q=1 - gamma) / np.power(1 + ro, 3 / 2)

    if k > criterion:
        return f'NOT Accepted, k = {k}, criterion = {criterion}'
    else:
        return f'Accepted, k = {k}, criterion = {criterion}'


def chi_square(alpha, r=3):

    sample_1 = np.random.exponential(scale=1, size=n)
    sample_2 = np.random.exponential(scale=1, size=m)
    sample_3 = np.random.exponential(scale=1 / alpha, size=k)

    distributions = [sample_1, sample_2, sample_3]

    n_total = n + m + k

    split_list = np.linspace(0, 10, r)
    v = np.zeros((r - 1, len(distributions)))

    for idx_split in range(1, r):
        for idx_dist, dist in enumerate(distributions):
            mask = np.logical_and(dist > split_list[idx_split - 1], dist < split_list[idx_split])
            v[idx_split - 1, idx_dist] = dist[mask].shape[0]
    delta = 0
    for i in range(r - 1):
        for j in range(len(distributions)):
            delta += (v[i, j] - (v[i, :].sum() * v[:, j].sum()) / n_total) ** 2 / \
                     (v[i, :].sum() * v[:, j].sum() + 1e-10)
    delta *= n_total

    criterion = chdtri(r - 1, gamma)
    if delta > criterion:
        return f'Not accepted , delta = {delta}, criterion = {criterion}'
    else:
        return f'Accepted , delta = {delta}, criterion = {criterion}'


if __name__ == '__main__':
    a = 1.1
    for m in map(int, (1e3, 1e4, 1e5)):
        n = int(m / 2)
        print(f'{"-" * 50}\nn = {n}, m = {m}, {empty_cell(alpha=a)}')

    for n in map(int, (2e2, 2e3, 2e4)):
        m, k = 3 * n, 2 * n
        print(f'{"-" * 50}\nn = {n}, m = {m}, k = {k}, {chi_square(alpha=a)}')
