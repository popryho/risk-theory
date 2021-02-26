"""
build a confidence interval for:
    -  expected value in the assumption that there are random variables
            that have a normal distribution, but the variance is unknown.
    -  expected value in the assumption that there are random variables
            whose distribution is unknown.
    -  variance in the assumption that there are random variables
            that have a normal distribution.
"""


# https://aegis4048.github.io/comprehensive_confidence_intervals_for_python_developers
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats

from const import gamma, N, seed

np.random.seed(seed)
plt.style.use('ggplot')


def task_a(arr) -> tuple:
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
    t = stats.t.ppf(1 - (1 - gamma) / 2., df)  # t-critical value for 99% CI = 2.576
    s = np.std(arr, ddof=1)                    # sample standard deviation

    lower = np.mean(arr) - (t * s / np.sqrt(n))
    upper = np.mean(arr) + (t * s / np.sqrt(n))

    return lower.astype('float16'), upper.astype('float16')


def task_b(arr):
    """
    confidence interval for mathematical expectation in the assumption that there are random variables
        whose distribution is unknown.

    stats.norm.interval(alpha=1 - alpha, loc=np.mean(arr), scale=stats.sem(arr))

    :param arr: sample
    :return: confidence interval for a population mean
    """
    n = len(arr)
    z = stats.norm.ppf(q=1 - (1 - gamma) / 2.)
    s = np.std(arr, ddof=1)

    lower = np.mean(arr) - (z * s / np.sqrt(n))
    upper = np.mean(arr) + (z * s / np.sqrt(n))

    return lower.astype('float16'), upper.astype('float16')


def task_c(arr):
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
        plt.savefig(f'../lab1/images/output_task1_{idx}.png', bbox_inches='tight')
        plt.close()

        with open('output_task1.txt', 'a+') as txt:
            txt.write(f"{idx}, Sample_size = {sample_size}, "
                      f"a in {task_a(sample)}, "
                      f"a in {task_b(sample)}, "
                      f"sigma**2 in {task_c(sample)}\n")
