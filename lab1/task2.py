import numpy as np
from scipy import stats
from const import (
    alpha,  # parameters
    eps,  # relative error
    gamma,  # confidence level
    n_0,  # the initial number of implementations
)


def generate_ksi(n, a):
    sample = np.random.uniform(low=0, high=1, size=n)
    ksi = (1 / a) * np.power(-np.log(sample), 0.25)
    return ksi


def generate_eta(n):
    sample = np.random.uniform(low=0.0, high=1.0, size=n)
    eta = np.power(-np.log(sample), 0.5)
    return eta


def num_of_implementation(array_of_q):
    Q = array_of_q.mean()
    sigma = array_of_q.std(ddof=1)

    z = stats.norm.ppf(q=1 - (1 - gamma) / 2.)  # 2.575
    return Q, np.power((z * sigma) / (eps * Q), 2)


def method_1(exact_values):
    """
    METHOD 1  The standard Monte Carlo method.
    """

    for a, exact_value in zip(alpha, exact_values):
        n_step = n_0
        while True:
            n_step += 1000
            ksi = generate_ksi(n_step, a)
            theta = generate_eta(n_step)
            evals = ((np.array(theta) - np.array(ksi)) > 0).astype(np.uint8)
            Q, sigma = num_of_implementation(evals)
            if n_step > sigma:
                print("For alpha = {0}, approximate value = {1} "
                      "var = {2}, n_iterations = {3}".format(a, Q, evals.std() ** 2, n_step))
                break


def method_2(exact_values):
    """
    METHOD 2
    """

    for a, exact_value in zip(alpha[:2], exact_values[:2]):
        n_step = n_0
        while True:
            n_step += 100
            psi = generate_ksi(n_step, a)
            evals = np.exp(-np.power(psi, 2))
            Q, sigma = num_of_implementation(evals)
            if n_step > sigma:
                print("For alpha = {0}, approximate value = {1} "
                      "var = {2}, n_iterations = {3}".format(a, Q, evals.std() ** 2, n_step))
                break


def method_3(exact_values):
    """
    METHOD 3
    """

    for a, exact_value in zip(alpha, exact_values):
        n_step = n_0
        while True:
            n_step += 1000
            theta = generate_eta(n_step)
            evals = 1 - np.exp(-np.power(a * theta, 4))
            Q, sigma = num_of_implementation(evals)
            if n_step > sigma:
                print("For alpha = {0}, approximate value = {1} "
                      "var = {2}, n_iterations = {3}".format(alpha, Q, evals.std(), n_step))
                break


def method_4(exact_values):
    """
    METHOD 4
    """

    for a, exact_value in zip(alpha[1:], exact_values[1:]):
        n_step = n_0
        while True:
            n_step += 1
            w1 = np.random.uniform(0, 1, size=n_step)
            w2 = np.random.uniform(0, 1, size=n_step)
            w3 = np.random.uniform(0, 1, size=n_step)
            t1 = -np.log(w1)
            t2 = -np.log(w2)
            t3 = -np.log(w3)
            beta = np.sqrt(t1 + t2 + t3)
            evals = (2 / np.power(beta, 4)) * (1 - np.exp(- np.power(a * beta, 4)))
            Q, sigma = num_of_implementation(evals)
            if n_step > sigma:
                print("For alpha = {0}, approximate value = {1} "
                      "var = {2}, n_iterations = {3}".format(a, Q, evals.std(), n_step))
                break


if __name__ == '__main__':

    # the exact value of the probability Q
    exact_q = (2*a**4 for a in alpha)

    method_1(exact_q)
