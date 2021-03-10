"""
Check hypothesis using the KOLMOGOROV test
"""

import numpy as np
from const import gamma

from scipy.special import kolmogi
from scipy.stats import expon


def kolmogorov(alpha):
    """
    :param alpha: the scale parameter
    :return: line about accepting or rejecting a hypothesis
    """
    arr = np.random.exponential(scale=alpha, size=n)  # sample of size n from an exponential distribution

    arr = np.sort(arr)  # order statistic
    k = np.array(range(1, len(arr) + 1))

    D = np.maximum(expon.cdf(x=arr, loc=0, scale=1) - (k - 1) / len(arr),
                   k / len(arr) - expon.cdf(x=arr, loc=0, scale=1)).max()

    if np.sqrt(len(arr)) * D < kolmogi(gamma):
        return f'D = {D:0.4f}. \nThe statistical data do NOT CONFLICT with the H0 hypothesis.\n'
    else:
        return f'D = {D:0.4f}. \nThe statistical data do CONFLICT with the H0 hypothesis.\n'


if __name__ == '__main__':

    """
    EXAMPLE. KOLMOGOROV test.
    """
    with open('output/output_task1.txt', 'a+') as txt:
        txt.write(f'gamma = {gamma} => z_gamma = {kolmogi(gamma):0.4f}\n')

        for a in [1, 1.3]:
            txt.write(f'\n\nH_0: X_i from F(u, 1) when in fact X_i from F(u, {a})\n')
            for n in map(int, (1e3, 1e4, 1e5)):
                txt.write(f'{"-"*50}\nn = {n} => sqrt(n) = {np.sqrt(n):0.2f}, '
                          f'{kolmogorov(alpha=a)}')
