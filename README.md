# Methods of reliability and risk theory

## LABORATORY WORK №1

  ### task 1

  Build a confidence interval for:
  
      -  expected value in the assumption that there are random variables
              that have a normal distribution, but the variance is unknown.
      -  expected value in the assumption that there are random variables
              whose distribution is unknown.
      -  variance in the assumption that there are random variables
              that have a normal distribution.

  ### task 2

  Calculate the integral in four ways and estimate the rates of convergence.


## LABORATORY WORK №2

The statistical hypothesis tests about the type of distribution, and the hypothesis of homogeneity.

Draw samples from an exponential distribution with different scale parameters and check hypothesis with the determined 
significance level for different sample lengths.

### task 1 
    - Check hypothesis using the KOLMOGOROV test
### task 2
    - Check hypothesis using the CHI-SQUARE test
### task 3
    - Check hypothesis using the EMPTY-BOXES test
### task 4
    - Check hypothesis using the one-sided SMIRNOV homogeneity test.

## LABORATORY WORK №3

Statistical hypothesis tests:

    - homogeneity hypothesis (EMPTY-BOXES, and CHI-SQUARE test)
    - independence hypothesis (CHI-SQUARE, SPEARMAN and KENDALL tests)
    - randomness hypothesis (test based on the number of inversions)

### task 1
    Check the homogeneity hypothesis using the EMPTY-BOXES test with the following parameters:
        - n = 500, m = 1000;
        - n = 5000, m = 10000;
        - n = 50000, m = 100000.
    
    Check the homogeneity hypothesis using the CHI-SQUARE test with the following parameters:
        - n = 200, m = 600, k = 400
        - n = 2000, m = 6000, k = 4000
        - n = 20000, m = 60000, k = 40000

        Note: choose the number of intervals and the intervals on your own.

### task 2
    Generate sample (X, Y) = {(X1, Y1), (X2, Y2), ... (Xn, Yn)} where {Xi} - implementation of a random variable ksi 
    from uniform distribution on [0, 1], {Yi}  - implementation of a random variable ksi + eta, where eta is a random 
    variable from uniform distribution on [-1, 1].

    Check independence hypothesis using the CHI-SQUARE test with the following parameters:
        - n = 500
        - n = 5000
        - n = 5000
    Check independence hypothesis using the SPEARMAN test with the following parameters:
        - n = 500
        - n = 5000
        - n = 5000
    Check independence hypothesis using the KENDALL test with the following parameters:
        - n = 500
        - n = 5000
        - n = 5000
### task 3
    Generate sample X = (X1, X2, ..., Xn) where Xi = ksi_1 + ... + ksi_n. 
    ksi_i -  is a random variable from uniform distribution on [-1, 1].

    Check randomness hypothesis with the following parameters:
        - n = 500
        - n = 5000
        - n = 5000