"""
Calculate the integral in four ways and estimate the rates of convergence.
"""
import numpy as np
from scipy import stats
from const import (
    alpha,  # parameters
    eps,    # relative error
    gamma,  # confidence level
)
from time import time


def generate_ksi(n, a):
    """
    generate ksi: ksi = (1/a)*(-ln(w))**(1/4)
    where w_i ~ U([0, 1]) -- sample from uniform distribution
    """
    sample = np.random.uniform(low=0.0, high=1.0, size=int(n))
    return (1 / a) * np.power(-np.log(sample), 0.25)


def generate_eta(n):
    """
    generate eta: eta = (-ln(w))**(1/2)
    where w_i ~ U([0, 1])  -- sample from uniform distribution
    """
    sample = np.random.uniform(low=0.0, high=1.0, size=int(n))
    return np.power(-np.log(sample), 0.5)


def num_of_implementation(array_of_q):
    """
    n*  --  required sample length
    """
    q = array_of_q.mean()
    sigma = array_of_q.std(ddof=1)

    z = stats.norm.ppf(q=1 - (1 - gamma) / 2.)  # 2.575
    return np.power((z * sigma) / (eps * q), 2)


def method_1(a, n_0):
    """
    METHOD 1  The standard Monte Carlo method.
    """
    n = n_0
    start_time = time()
    while True:
        ksi = generate_ksi(n, a)
        eta = generate_eta(n)
        expected_q = ((np.array(eta) - np.array(ksi)) > 0).astype(np.uint8)

        if n > num_of_implementation(expected_q):
            with open('output_task2.txt', 'a') as txt:
                txt.write(f"method_{1}, expected value = {expected_q.mean()}, "
                          f"variance = {expected_q.var(ddof=1)}, "
                          f"sample length = {int(n)}, "
                          f"time = {time() - start_time}\n")
            break
        n += 1e6


def method_2(a, n_0):
    """
    METHOD 2
    """
    n = n_0
    start_time = time()
    while True:
        ksi = generate_ksi(n, a)
        expected_q = np.exp(-np.power(ksi, 2))

        if n > num_of_implementation(expected_q):
            with open('output_task2.txt', 'a') as txt:
                txt.write(f"method_{2}, expected value = {expected_q.mean()}, "
                          f"variance = {expected_q.var(ddof=1)}, "
                          f"sample length = {int(n)}, "
                          f"time = {time() - start_time}\n")
            break
        n += int(1e4)


def method_3(a, n_0):
    """
    METHOD 3
    """
    n = n_0
    start_time = time()
    while True:
        eta = generate_eta(n)
        expected_q = 1 - np.exp(-np.power(a * eta, 4))

        if n > num_of_implementation(expected_q):
            with open('output_task2.txt', 'a') as txt:
                txt.write(f"method_{3}, expected value = {expected_q.mean()}, "
                          f"variance = {expected_q.var(ddof=1)}, "
                          f"sample length = {int(n)}, "
                          f"time = {time() - start_time}\n")
            break
        n += 1e4


def method_4(a, n_0):
    """
    METHOD 4
    """
    n = n_0
    start_time = time()
    while True:
        w = np.random.uniform(0, 1, size=(3, int(n)))
        beta = np.sqrt(-np.log(w).sum(axis=0))

        expected_q = (2 / np.power(beta, 4)) * (1 - np.exp(- np.power(a * beta, 4)))
        if n > num_of_implementation(expected_q):
            with open('output_task2.txt', 'a') as txt_file:
                txt_file.write(f"method_{4}, expected value = {expected_q.mean()}, "
                               f"variance = {expected_q.var(ddof=1)}, "
                               f"sample length = {int(n)}, "
                               f"time = {time() - start_time}\n")
            break
        n += 1e4


if __name__ == '__main__':

    """
    alpha = 1
    """
    with open('output_task2.txt', 'a') as txt_file:
        txt_file.write(f"alpha = {1}:\n")
    [fun(alpha[0], n) for fun, n in zip((method_1, method_2, method_3, method_4),
                                        (1e4, 1e4, 1e4, 1e4))]

    """
    alpha = 0.1
    """
    with open('output_task2.txt', 'a') as txt_file:
        txt_file.write(f"\nalpha = {0.1}:\n")
    [fun(alpha[1], n) for fun, n in zip((method_1, method_2, method_3, method_4),
                                        (4e6, 1e6, 2e5, 3))]

    """
    alpha = 0.01
    """
    with open('output_task2.txt', 'a') as txt_file:
        txt_file.write(f"\nalpha = {0.01}:\n")
    [fun(alpha[2], n) for fun, n in zip((method_3, method_4),
                                        (3e5, 2))]
