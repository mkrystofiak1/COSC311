import pandas as pd
import numpy as np
import math
from matplotlib import pyplot as plt
from collections import Counter 

def mean(x):
    # Calculate and return mean of a numpy array
    sum = 0
    for i in range(len(x)):
        sum += x[i]
    return sum/len(x)

def median(x):
    """Calculate and return median of a numpy array."""
    temp = sorted(x)
    y = len(x)
    
    #use // for intege division
    return temp[y // 2] if y % 2 == 1 else (temp[y // 2] + (temp[(y // 2)-1]))/2

def mode(xs):
    counts = Counter(xs)
    return np.array([x[0] for x in counts.items() if x[1] == max(counts.values())])
    
def quantiles(xs, q):
    temp = sorted(xs)
    return temp[int(len(temp)*q)]
    
def interquartile_range(xs):
    return quantiles(xs,.75) - quantiles(xs,.25)

def center(xs):
    return np.array([x-mean(xs) for x in xs])

def var(xs):
    # Average squared distance from the mean"
    return mean([x**2 for x in center(xs)])
    
def std(xs):
    return math.sqrt(var(xs))
    
def spread(x):
    return max(x)-min(x)

def cov(xs, ys):
    """ Take two lists of observations and compute their covariance """
    assert len(xs) == len(ys)
    cx = center(xs)
    cy = center(ys)
    return mean([cx[i]*cy[i] for i in range(len(cx))])

def correlation(xs, ys):
    return cov(xs, ys)/(std(xs)*(std(ys)))
                        
def uniform_pdf(x,a,b):
    return 1/(b-a) if a <= x and x <= b else 0
                        
def uniform_cdf(t,a,b):
    if t < a:
        return 0
    if t > b:
        return 1
    return (t-a)/(b-a)

def normal_pdf(x,mu,sigma):
    return (1/(sigma*np.sqrt(2*np.pi))) * np.exp(-1*(x-mu)**2 / (2*sigma**2))

def normal_cdf(x,mu,sigma):
    return (1 + math.erf((x-mu)/(sigma*np.sqrt(2))))/2

def binom_draw(n,p):
    return np.sum(np.random.choice([0,1],size=n,p=[1-p,p]))

def inverse_normal_cdf(p: float, mu: float = 1, tolerance: float = 0.00001) -> float:
    """Find approximate inverse using binary search"""
   # if not standard, compute standard and rescale
    if mu != 0 or sigma != 1:
        return mu + sigma * inverse_normal_cdf(p, tolerance=tolerance)
    low_z = -10.0
    hi_z = 10.0
    while hi_z - low_z > tolerance:
        mid_z = (low_z + hi_z)/2
        mid_p = normal_cdf(mid_z)
        if mid_p < p:
            low_z = mid_z
        else:
            hi_z = mid_z
    return mid_z

def normal_probability_above(t, mu, sigma):
    return 1 - normal_cdf(t,mu,sigma)

def normal_probability_between(a, b, mu, sigma):
    return normal_cdf(b,mu,sigma) - normal_cdf(a, mu, sigma)

def normal_probability_outside(a,b,mu,sigma):
    return 1 - normal_probability_between(a,b,mu,sigma)

def two_sided_p_value(x, mu, sigma):
    """ How likely are we to see a value at least as extreme as x (in
        either direction) if our values are from an N(mu, sigma)? """
    if x >= mu:
        # x is greater than the mean, so the tail is everything greater than x
        return 2 * normal_probability_above(x, mu, sigma)
    else:
        # x is less than the mean, so the tail is everything less than x
        return 2 * normal_cdf(x, mu, sigma)


              
                       
#only run print when stand aloe, not on import
#when a python program runs, it sets the duner __name__
#variable to be its context in the greater program
#So the program t hat got ran through python is __main__
#tpyically used if you want to have some tests here but
#dont run tests when you import into other programs

if __name__ == '__main__':
    print('TESTING stats.py..')
    print(f"the mean of [1,2,3,4,5] is {mean(np.array([1,2,3,4,5]))}.")