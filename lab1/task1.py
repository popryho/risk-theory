# TODO: build a confidence interval
# https://aegis4048.github.io/comprehensive_confidence_intervals_for_python_developers
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats

from const import gamma, N, seed

np.random.seed(seed)
plt.style.use('ggplot')


def A(arr):
    """
    confidence interval for mathematical expectation in the assumption that there are random variables
        that have a normal distribution, but the variance is unknown.
    stats.t.interval(alpha=gamma, df=len(sample)-1, loc=np.mean(sample), scale=stats.sem(sample))

    :param arr: sample
    :return: confidence interval for a population mean
    """
    #  ppf -  probability percentile function, функция, обратная к функции распределения
    n = len(arr)                               # sample sizes
    df = n - 1                                 # degrees of freedom
    s = np.std(arr, ddof=1)                    # sample standard deviation
    t = stats.t.ppf(1 - (1 - gamma) / 2., df)  # t-critical value for 99% CI = 2.576

    lower = np.mean(arr) - (t * s / np.sqrt(n))
    upper = np.mean(arr) + (t * s / np.sqrt(n))

    return lower.astype('float16'), upper.astype('float16')


def B(arr):
    """
    confidence interval for mathematical expectation in the assumption that there are random variables
        whose distribution is unknown.

    stats.norm.interval(alpha=1 - alpha, loc=np.mean(arr), scale=stats.sem(arr))

    :param arr: sample
    :return: confidence interval for a population mean
    """
    z = stats.norm.ppf(q=1 - (1 - gamma) / 2.)
    n = len(arr)
    s = np.std(arr, ddof=1)

    lower = np.mean(arr) - (z * s / np.sqrt(n))
    upper = np.mean(arr) + (z * s / np.sqrt(n))

    return lower.astype('float16'), upper.astype('float16')


def C(arr):
    """
    confidence interval for variance in the assumption that there are random variables
        that have a normal distribution.

    stats.chi2.interval(alpha=gamma, df=len(sample)-1, loc=np.mean(sample), scale=stats.sem(sample)))

    :param arr: sample
    :return: confidence interval for a population variance
    """
    n = len(arr)              # sample sizes
    s2 = np.var(arr, ddof=1)  # sample variance
    df = n - 1                # degrees of freedom

    upper = (n - 1) * s2 / stats.chi2.ppf((1 - gamma) / 2, df)
    lower = (n - 1) * s2 / stats.chi2.ppf(1 - (1 - gamma) / 2, df)

    return lower.astype('float16'), upper.astype('float16')


if __name__ == '__main__':

    population = stats.norm.rvs(loc=0.0, scale=1.0, size=1000000)

    for idx, sample_size in enumerate(N):

        sample = np.random.choice(a=population, size=sample_size)
        sns.histplot(x=sample, kde=True, color='orange')
        plt.show()
        print(idx, "Sample_size = ", sample_size,
              ", a in {}, a in {}, sigma**2 in {}".format(A(sample), B(sample), C(sample)))
