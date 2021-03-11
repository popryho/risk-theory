"""
Check hypothesis using the one-sided SMIRNOV test
Smirnov homogeneity test
from scipy.stats import ks_2samp
"""

import numpy as np
from const import gamma

from scipy.special import kolmogi
from scipy.stats import expon


def smirnov(alpha):

    """
    Compute the Kolmogorov-Smirnov statistic on 2 samples.
    :param alpha: the scale parameter
    :return: line about accepting or rejecting a hypothesis
    """
    sample_1 = np.sort(np.random.exponential(scale=1, size=n))
    sample_2 = np.sort(np.random.exponential(scale=1 / alpha, size=int(n / 2)))

    k = np.array(range(1, len(sample_2) + 1))

    D = np.maximum(expon.cdf(x=sample_2, loc=0, scale=1) - (k - 1) / len(sample_2),
                   k / len(sample_2) - expon.cdf(x=sample_2, loc=0, scale=1)).max()

    criteria = kolmogi(gamma) * np.sqrt((1 / n) + (1 / (n / 2)))

    if D < criteria:
        return f'D = {D:0.4f}, criteria = {criteria:0.4f}. \n' \
               f'The statistical data do NOT CONFLICT with the H0 hypothesis.'
    else:
        return f'D = {D:0.4f}, criteria = {criteria:0.4f}. \n' \
               f'The statistical data do CONFLICT with the H0 hypothesis.'


if __name__ == '__main__':

    """
    EXAMPLE. SMIRNOV test.
    """
    with open('output/output_task4.txt', 'a+') as txt:
        txt.write(f'gamma = {gamma} => z_gamma = {kolmogi(gamma):0.4f}\n')
        for a in [1, 1.3]:
            txt.write(f'\n\nH_0: X_1 from F(u, 1), X_2 from F(u, 1) '
                      f'when in fact X_1 from F(u, 1), X_1 from F(u, {a})\n')
            for n in map(int, (1e3, 1e4, 1e5)):
                txt.write(f'{"-"*50}\nn = {n}, {smirnov(alpha=a)}\n')
