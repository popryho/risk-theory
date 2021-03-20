import numpy as np
from scipy.special import chdtri
from scipy.stats import spearmanr
from scipy.stats import kendalltau
from const import gamma


def chi_square(X, Y, n, m):
    n_total = n + m
    v = np.zeros((n, m))
    split_list1 = np.linspace(0, 10, n)
    split_list2 = np.linspace(0, 10, m)
    for i_split in range(1, n):
        for j_split in range(1, m):
            a1 = X[X > split_list1[i_split - 1]]
            b1 = X[X < split_list1[i_split]]
            a2 = Y[Y > split_list2[j_split - 1]]
            b2 = Y[Y < split_list2[j_split]]
            v[i_split - 1, j_split - 1] = np.intersect1d(a1, b1).shape[0] + np.intersect1d(a2, b2).shape[0]
    delta = 0
    for i in range(1, n - 1):
        for j in range(1, m - 1):
            delta += ((v[i, j] - (v[i, :].sum() * v[:, j].sum()) / n_total) ** 2) / (v[i, :].sum() * v[:, j].sum())
    delta *= n_total
    krit = chdtri((n - 1) * (m - 1), gamma)

    if delta > krit:
        return f"Not accepted , delta = {delta}, krit = {krit}"
    else:
        return f"Accepted , delta = {delta}, krit = {krit}"


def spearm_test(data1, data2):
    correl, critical_value = spearmanr(data1, data2)
    # significance level
    alpha = 0.05
    if critical_value > alpha:
        return 'Fail to reject H0 critical_value=%.3f' % critical_value
    else:
        return 'Reject H0 critical_value=%.3f' % critical_value


def kendall_test(data1, data2):
    correl, critical_value = kendalltau(data1, data2)
    # significance level
    alpha = 0.05
    if critical_value > alpha:
        return 'Fail to reject H0 critical_value=%.3f' % critical_value
    else:
        return 'Reject H0 critical_value=%.3f' % critical_value


def task_a():
    r, k = 15, 18
    for n in map(int, (5e2, 5e3, 5e4)):
        X = np.random.uniform(0, 1, size=n)
        Y = np.random.uniform(-1, 1, size=n)
        print(f'n = {n}, {chi_square(X, X + Y, r, k)}')


def task_b():
    for n in map(int, (5e2, 5e3, 5e4)):
        X = np.random.uniform(0, 1, size=n)
        Y = np.random.uniform(-1, 1, size=n)
        print(f'n = {n}, {spearm_test(X, X + Y)}')


def task_c():
    for n in map(int, (5e2, 5e3, 5e4)):
        X = np.random.uniform(0, 1, size=n)
        Y = np.random.uniform(-1, 1, size=n)
        print(f"n = {n}, {kendall_test(X, X + Y)}")


if __name__ == '__main__':
    task_a()
    task_b()
    task_c()
