"""
Check hypothesis using the EMPTY-BOXES test
"""

import numpy as np
from scipy.stats import norm
from scipy.stats import expon
from const import gamma, ro


def t_alpha():
    r = n / ro
    return r * np.exp(-ro) + norm.ppf(q=1 - gamma) * np.sqrt(r * np.exp(-ro) * (1 - (1 + ro) * np.exp(-ro)))


def empty_cells(alpha):
    """
    :param alpha: the scale parameter
    :return: line about accepting or rejecting a hypothesis
    """
    arr = np.random.exponential(scale=alpha, size=n)
    r = int(n / ro)

    uni_sample = expon.cdf(arr, scale=1)
    nu, _ = np.histogram(uni_sample, bins=r, range=(0, 1))

    count = np.count_nonzero(nu == 0)

    if count > t_alpha():
        return f't_alpha = {t_alpha():.3f}, count of empty boxes = {count}\n' \
               f'The statistical data do CONFLICT with the H0 hypothesis.'
    else:
        return f't_alpha = {t_alpha():.3f}, count of empty boxes = {count}\n' \
               f'The statistical data do NOT CONFLICT with the H0 hypothesis.'


if __name__ == '__main__':

    """
    EXAMPLE. EMPTY-BOXES test.
    """

    with open('output/output_task3.txt', 'a+') as txt:
        for a in [1, 1.3]:
            txt.write(f'H_0: X_i from F(u, 1) when in fact X_i from F(u, {a})\n')
            for n in map(int, (1e3, 1e4, 1e5)):
                txt.write(f'{"-" * 50}\nn = {n}, {empty_cells(a)}\n')
            txt.write('\n\n')
