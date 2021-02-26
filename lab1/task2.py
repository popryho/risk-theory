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
from multiprocessing import Pool
import functools
from time import time


def generate_ksi(n, a):
    """
    generate ksi: ksi = (1/a)*(-ln(w))**(1/4)
    where w_i ~ U([0, 1]) -- sample from uniform distribution
    """
    sample = np.random.uniform(low=0.0, high=1.0, size=n)
    return (1 / a) * np.power(-np.log(sample), 0.25)


def generate_eta(n):
    """
    generate eta: eta = (-ln(w))**(1/2)
    where w_i ~ U([0, 1])  -- sample from uniform distribution
    """
    sample = np.random.uniform(low=0.0, high=1.0, size=n)
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
    n_step = n_0
    while True:
        n_step += 10000
        ksi = generate_ksi(n_step, a)
        eta = generate_eta(n_step)
        expected_q = ((np.array(eta) - np.array(ksi)) > 0).astype(np.uint8)

        if n_step > num_of_implementation(expected_q):
            print("For alpha = {0}, approximate value = {1} "
                  "var = {2}, n_iterations = {3}".format(a, expected_q.mean(), expected_q.var(ddof=1), n_step))
            break


def method_2(a, n_0):
    """
    METHOD 2
    """
    n_step = n_0
    while True:
        n_step += 100
        ksi = generate_ksi(n_step, a)
        expected_q = np.exp(-np.power(ksi, 2))

        if n_step > num_of_implementation(expected_q):
            print("For alpha = {0}, approximate value = {1} "
                  "var = {2}, n_iterations = {3}".format(a, expected_q.mean(), expected_q.var(ddof=1), n_step))
            break


def method_3(a, n_0):
    """
    METHOD 3
    """
    n_step = n_0
    while True:
        n_step += 1000
        eta = generate_eta(n_step)
        expected_q = 1 - np.exp(-np.power(a * eta, 4))

        if n_step > num_of_implementation(expected_q):
            print("For alpha = {0}, approximate value = {1} "
                  "var = {2}, n_iterations = {3}".format(a, expected_q.mean(), expected_q.var(ddof=1), n_step))
            break


def method_4(a, n_0):
    """
    METHOD 4
    """
    n_step = n_0
    while True:
        n_step += 1
        w = np.random.uniform(0, 1, size=(3, n_step))
        beta = np.sqrt(-np.log(w).sum(axis=0))

        expected_q = (2 / np.power(beta, 4)) * (1 - np.exp(- np.power(a * beta, 4)))
        if n_step > num_of_implementation(expected_q):
            print("For alpha = {0}, approximate value = {1} "
                  "var = {2}, n_iterations = {3}".format(a, expected_q.mean(), expected_q.var(ddof=1), n_step))
            break


if __name__ == '__main__':

    n0 = 1000  # the initial number of implementations required to "stabilize" the variance.

    start_time = time()
    with Pool(processes=6) as p:
        p.map(functools.partial(method_1, n_0=n0), alpha[:2])
    print("--- %s seconds ---" % (time() - start_time))
