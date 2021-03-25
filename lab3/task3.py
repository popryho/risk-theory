import numpy as np
from scipy.stats import norm

from const import gamma


def randomness():
    ksi = np.random.uniform(low=-1, high=1, size=n)
    sample = np.cumsum(ksi)
    k = int(np.sum([len(sample[i + 1:n][sample[i] > sample[i + 1:n]]) for i in range(n)]))
    criterion = abs((6 / (n ** (3 / 2))) * (k - ((n * (n - 1)) / 4)))
    z = norm.ppf(1 - gamma / 2)

    if criterion > z:
        return f'Number of inversions = {k}, criterion = {criterion:.3f}, z = {z}\n' \
               f'The statistical data do CONFLICT with the H0 hypothesis. \n'
    else:
        return f'Number of inversions = {k}, criterion = {criterion:.3f}, z = {z}\n' \
               f'The statistical data do NOT CONFLICT with the H0 hypothesis. \n'


if __name__ == '__main__':

    with open('output/output_task3.txt', 'w') as txt:
        for n in map(int, (5e2, 5e3, 5e4)):
            txt.write(f'{"-" * 50}\nn={n}, {randomness()}\n')
