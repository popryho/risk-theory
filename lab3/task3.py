import scipy.stats
import numpy as np


def number_inversion(X):
    count = 0
    for i in range(1, len(X) - 1):
        for j in range(i):
            if X[j] > X[i]:
                count += 1
    return count


def calculate_confidence_interval(n):
    mean = (n * (n - 1)) / 4
    std = np.sqrt((n * (n - 1) * (2 * n + 5)) / 72)
    h = std * scipy.stats.t.ppf((1 + .95) / 2., n - 1)
    return mean - h, mean + h


if __name__ == '__main__':

    for n in map(int, (5e2, 5e3, 5e4)):

        X = np.random.uniform(-1, 1, size=n)

        n_inversions = number_inversion(X)
        left, right = calculate_confidence_interval(n)

        print(f'Allowed interval for Normal distribution is [{left:.1f}, {right:.1f}]')
        if left < n_inversions < right:
            print(f'Number of inversions = {n_inversions}. '
                  f'The statistical data do NOT CONFLICT with the H0 hypothesis.')
        else:
            print(f'Number of inversions = {n_inversions} '
                  f'The statistical data do CONFLICT with the H0 hypothesis.')



def form_X(n):
    ksi = np.random.uniform(low=-1.0, high=1.0, size=n)
    X = np.zeros(n)
    temp = ksi[0]
    X[0] = temp
    for i in range(1, n):
        temp += ksi[i]
        X[i] = temp
    return X


def calc_eta(X):
    n = len(X)
    eta = np.zeros(n)
    for i in range(n):
        subX = X[i + 1:n]
        eta[i] = len(subX[X[i] > subX])
    return eta


def independence(X):
    n = len(X)
    eta = calc_eta(X)
    S = np.sum(eta)
    z = norm.ppf(1 - 0.05 / 2)
    criteria = abs((6 / (n ** (3 / 2))) * (S - ((n * (n - 1)) / 4)))
    print(f'n = {n}')
    print(f'Приблизні значення вибірки Х: {X[0:4]}')
    print(f'eta: {eta[0:4]}')
    print(f'S(X) = {S}')
    print(f'criteria =  (6/(n^(3/2)))*(S(X)-(n*(n-1)/4)) = {criteria}')
    print(f'z = {z}')
    if criteria < z:
        print(f'criteria < z')
        print(f'Статистичні дані не суперечать гіпотезі H_0\n')
    else:
        print(f'criteria >= z')
        print(f'Слід прийняти альтернативну гіпотезу H_1\n')


if __name__ == '__main__':

    for n in map(int, (5e2, 5e3, 5e4)):
        X = form_X(n)
        independence(X)
