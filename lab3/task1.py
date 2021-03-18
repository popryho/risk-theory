import numpy as np
from scipy.special import chdtri
from scipy.stats import norm

from const import gamma


def empty_cell(alpha):
    sample_1 = np.sort(np.random.exponential(scale=1, size=n))
    sample_2 = np.random.exponential(scale=1 / alpha, size=m)
    k_value = 0
    for i in range(1, sample_1.shape[0]):
        mask = np.logical_and(sample_2 > sample_1[i - 1], sample_1[i] > sample_2)
        k_value += 1 if sample_2[mask].shape[0] == 0 else 0

    ro = m / n
    criterion = n / (1 + ro) + np.sqrt(n) * ro * norm.ppf(q=1 - gamma) / np.power(1 + ro, 3 / 2)

    if k_value > criterion:
        return f'k = {k_value}, criterion = {criterion:0.4f}. \n' \
               f'The statistical data do CONFLICT with the H0 hypothesis.'
    else:
        return f'k = {k_value}, criterion = {criterion:0.4f}. \n' \
               f'The statistical data do NOT CONFLICT with the H0 hypothesis.'


def chi_square(alpha):

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
        return f'Delta = {delta:0.4f}, criterion = {criterion:0.4f}. \n' \
               f'The statistical data do CONFLICT with the H0 hypothesis.'
    else:
        return f'Delta = {delta:0.4f}, criterion = {criterion:0.4f}. \n' \
               f'The statistical data do NOT CONFLICT with the H0 hypothesis.'


if __name__ == '__main__':

    a = 1.1
    r = 4
    with open('output/output_task1.txt', 'w') as txt:
        """
        EXAMPLE. EMPTY-BOXES test.
        """
        txt.write(f'TASK A. \n\ngamma = {gamma} => z_gamma = {norm.ppf(q=1 - gamma):0.4f}\n')
        for m in map(int, (1e3, 1e4, 1e5)):
            n = int(m / 2)
            txt.write(f'{"-" * 50}\nn={n}, m={m}, {empty_cell(alpha=a)}\n')

        """
        EXAMPLE. CHI-SQUARE test.
        """
        txt.write(f'\nTASK B.\n\nr = {r}\n')
        for n in map(int, (2e2, 2e3, 2e4)):
            m, k = 3 * n, 2 * n
            txt.write(f'{"-" * 50}\nn={n}, m={m}, k={k}, {chi_square(alpha=a)}\n')
