from scipy.stats import linregress
from scipy.stats import t

def linear_regression(x,y):
    res = linregress(x, y)
    tinv = lambda p, df: abs(t.ppf(p/2, df))
    ts = tinv(0.05, len(x)-2)
    return res.slope, res.intercept, ts*res.stderr