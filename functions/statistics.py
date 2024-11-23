from typing import Union, Sequence
from scipy.stats import linregress, t
import numpy as np

def linear_regression(x: np.ndarray, y: np.ndarray):
    """
    Perform linear regression on two arrays.

    Parameters
    ----------
    x : np.ndarray
        Independent variable values.
    y : np.ndarray
        Dependent variable values.

    Returns
    -------
    slope : float
        Slope of the regression line.
    intercept : float
        Intercept of the regression line.
    margin_of_error : float
        Margin of error for the slope at the 95% confidence interval.
    """
    res = linregress(x, y)
    tinv = lambda p, df: abs(t.ppf(p / 2, df))
    ts = tinv(0.05, len(x) - 2)
    margin_of_error = ts * res.stderr
    return res.slope, res.intercept, margin_of_error

def trimmean2(x: Union[Sequence[float], np.ndarray], p: float) -> float:
    """
    Compute the trimmed mean by ignoring the top and bottom p/2 percent of the data.

    Parameters
    ----------
    x : Union[Sequence[float], np.ndarray]
        The input data, which can be a list or a numpy array.
    p : float
        Trimming percentage (0 <= p <= 100). Represents the percentage of data 
        to remove equally from both ends.

    Returns
    -------
    float
        The trimmed mean of the data.
    """
    if not (0 <= p <= 100):
        raise ValueError("The trimming percentage 'p' must be between 0 and 100.")

    # Convert the input to a numpy array if it isn't already
    x = np.asarray(x)
    
    # Sort the data
    x_sorted = np.sort(x)
    n = len(x_sorted)
    
    # Calculate indices to trim
    lower_idx = int(np.floor(n * (p / 200)))
    upper_idx = int(np.ceil(n * (1 - (p / 200))))
    
    # Select the trimmed data
    trimmed_data = x_sorted[lower_idx:upper_idx]
        
    return np.mean(trimmed_data)

def bootrsp(in_array: Union[Sequence[float], np.ndarray], B: int = 1) -> np.ndarray:
    """
    Bootstrap resampling procedure.

    Parameters
    ----------
    in_array : Union[Sequence[float], np.ndarray]
        Input data. Can be a vector or a 2D matrix.
    B : int, optional
        Number of bootstrap resamples (default is 1).

    Returns
    -------
    np.ndarray
        Bootstrap resamples of the input data.
        
        - For a vector input of size [N,], produces a matrix of size [N, B],
          with columns being resamples of the input vector.
        - For a matrix input of size [N, M], produces a 3D matrix of size [N, M, B],
          where `out[:, :, i]` is a resample of the input matrix for `i = 0,..., B-1`.

    References
    ----------
    
    - Efron, B. and Tibshirani, R. An Introduction to the Bootstrap. 
    Chapman and Hall, 1993.

    - Zoubir, A.M. Bootstrap: Theory and Applications. Proceedings 
    of the SPIE 1993 Conference on Advanced Signal Processing Algorithms, 
    Architectures and Implementations. pp. 216-235, San Diego, July 1993.

    - Zoubir, A.M. and Boashash, B. The Bootstrap and Its Application
    in Signal Processing. IEEE Signal Processing Magazine, Vol. 15, 
    No. 1, pp. 55-76, 1998.
    """
    in_array = np.asarray(in_array)
    
    if B < 1:
        raise ValueError('B must be at least 1.')
    if in_array.size == 0:
        raise ValueError('Input data cannot be empty.')
    if in_array.ndim > 2:
        raise ValueError('Input data must be a vector or a 2D matrix only.')
    
    s = in_array.shape

    if in_array.ndim == 1 or min(s) == 1:
        # Vector case
        N = in_array.size
        indices = np.random.randint(0, N, size=(N, B))
        out = in_array[indices]
    else:
        # Matrix case
        N, M = in_array.shape
        indices = np.random.randint(0, N, size=(N, B))
        out = np.stack([in_array[:, col_idx] for col_idx in indices.T], axis=2)
    
    return out

def remove_outliers(df, column, method='IQR', factor=1.5):
    """
    Remove outliers from a DataFrame column using IQR method.
    """
    if method == 'IQR':
        Q1 = df[column].quantile(0.25)
        Q3 = df[column].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - factor * IQR
        upper_bound = Q3 + factor * IQR
        return df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]
    else:
        raise ValueError("Invalid method. Use 'IQR'.")