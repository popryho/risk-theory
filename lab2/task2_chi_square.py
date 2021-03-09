"""
Check hypothesis using the CHI-SQUARE test

scipy.stats.chisquare
scipy.stats.chi2
"""

import numpy as np
from scipy.special import chdtri
from scipy.stats import expon
from const import gamma


def chi_experiment(arr, alpha):
    size = len(arr)
    r = int(20 * size / 1000)

    z_gamma = chdtri(r - 1, gamma)

    nu, bin_edges = np.histogram(arr, bins=r, range=(expon.ppf(0.001), expon.ppf(0.999)))
    p = np.array([expon.sf(x=bin_edges[i - 1], scale=1 / alpha) -
                  expon.sf(x=bin_edges[i], scale=1 / alpha) for i in range(1, r + 1)])
    delta = np.sum(((nu - size * p) ** 2) / (size * p))

    if delta > z_gamma:
        return f'r = {r}, z_gamma = {z_gamma:.3f}, delta = {delta:.3f}. \n' \
               f'The statistical data do CONFLICT with the H0 hypothesis.'

    else:
        return f'z_gamma = {z_gamma:.3f}, delta = {delta:.3f}. \n' \
               f'The statistical data do NOT CONFLICT with the H0 hypothesis.'


if __name__ == '__main__':

    """
    EXAMPLE. CHI-SQUARE test.
    """

    with open('output/output_task2.txt', 'a+') as txt:
        for a in [1, 1.3]:
            txt.write(f'H_0: X_i from F(u, 1) when in fact X_i from F(u, {a})\n')
            for n in map(int, (1e3, 1e4, 1e5)):
                sample = np.random.exponential(scale=1 / a, size=n)
                txt.write(f'{"-" * 50}\nn = {n}, {chi_experiment(sample, 1)}\n')
            txt.write('\n\n')
