import numpy as np
from scipy.stats import linregress, t
from scipy.fft import fft2, ifft2, fftshift
from typing import Tuple, Union

def linear_regression(x: np.ndarray, y: np.ndarray) -> Tuple[float, float, float]:
    """
    Perform linear regression on two arrays.

    Parameters
    ----------
    x : np.ndarray
        1-D array of independent variable values.
    y : np.ndarray
        1-D array of dependent variable values.

    Returns
    -------
    slope : float
        Slope of the regression line.
    intercept : float
        Intercept of the regression line.
    margin_of_error : float
        Margin of error for the slope at the 95% confidence interval.

    Examples
    --------
    >>> x = np.array([1, 2, 3, 4, 5])
    >>> y = np.array([2, 4, 5, 4, 5])
    >>> slope, intercept, margin_of_error = linear_regression(x, y)
    >>> slope
    0.6
    >>> intercept
    2.2
    >>> margin_of_error
    0.7071067811865476
    """
    res = linregress(x, y)
    tinv = lambda p, df: abs(t.ppf(p / 2, df))
    ts = tinv(0.05, len(x) - 2)
    margin_of_error = ts * res.stderr
    
    return res.slope, res.intercept, margin_of_error


def bootrsp(in_data: Union[np.ndarray, list], B: int = 1) -> np.ndarray:
    """
    Bootstrap resampling procedure.

    Parameters
    ----------
    data : array-like
        Input data, can be a vector or 2D matrix.
    B : int, optional
        Number of bootstrap resamples. Default is 1.

    Returns
    -------
    np.ndarray
        B bootstrap resamples of the input data.

        For a vector input data of size [N,], the resampling procedure produces 
        a matrix of size [N, B] with columns being resamples of the input vector.

        For a matrix input data of size [N, M], the resampling procedure produces 
        a 3D matrix of size [N, M, B] with out[:, :, i], i = 0,...,B-1, being a resample 
        of the input matrix.

    References
    ----------
    Efron, B. and Tibshirani, R. An Introduction to the Bootstrap.
        Chapman and Hall, 1993.

    Zoubir, A.M. Bootstrap: Theory and Applications. Proceedings 
        of the SPIE 1993 Conference on Advanced Signal 
        Processing Algorithms, Architectures and Implementations. 
        pp. 216-235, San Diego, July 1993.

    Zoubir, A.M. and Boashash, B. The Bootstrap and Its Application
        in Signal Processing. IEEE Signal Processing Magazine, 
        Vol. 15, No. 1, pp. 55-76, 1998.

    Examples
    --------
    >>> out = bootrsp(np.random.randn(10), 10)
    """  
    # Convert input to numpy array
    in_data = np.array(in_data)

    # Check if B is provided, otherwise default to 1
    if B < 1:
        raise ValueError("Number of bootstrap resamples (B) must be at least 1.")

    # Check if input data is a vector or a 2D matrix
    if in_data.ndim > 2:
        raise ValueError("Input data can be a vector or a 2D matrix only.")

    s = in_data.shape

    if len(s) == 1:
        # Vector input
        out = in_data[np.random.randint(0, s[0], size=(s[0], B))]
    else:
        # Matrix input
        out = in_data[np.random.randint(0, s[0]*s[1], size=(s[0], s[1], B))]

    return out

def trimmean2(x: np.ndarray, p: float) -> float:
    """
    Compute the trimmed mean of a data vector by ignoring the top and bottom percentiles.

    Parameters
    ----------
    x : numpy.ndarray
        Input data vector.
    p : float
        Percentage of data to ignore from the top and bottom. Should be between 0 and 100.

    Returns
    -------
    float
        Trimmed mean of the input data vector after removing the top and bottom p/2 percent
          of the data.

    Notes
    -----
    This function computes the trimmed mean by removing the top and bottom p/2 percent of the 
    data vector `x`.

    Examples
    --------
    >>> import numpy as np
    >>> x = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    >>> trimmean2(x, 10)
    5.5
    """
    # Check if p is within valid range
    if p <= 0 or p >= 100:
        raise ValueError("p should be between 0 and 100.")

    n = len(x)
    perc = 100 * ((np.arange(1, n+1) - 0.5) / n)
    x_sorted = np.sort(x)
    
    # Indices to keep based on p
    lower_index = np.searchsorted(perc, p / 2)
    upper_index = np.searchsorted(perc, 100 - p / 2)

    # Trimmed data
    newx = x_sorted[lower_index:upper_index]

    # Compute trimmed mean
    out = np.mean(newx)
    
    return out

def process_data(ttx: np.ndarray, tty: np.ndarray, x: np.ndarray, y: np.ndarray, 
                 vsini: np.ndarray, B: int, trimpct: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Processes the data to calculate spatial statistics of vsini.

    Parameters:
    -----------
    ttx : np.ndarray
        Coordinates along the X-axis to define the analysis space.
    tty : np.ndarray
        Coordinates along the Y-axis to define the analysis space.
    x : np.ndarray
        X coordinates of the observations.
    y : np.ndarray
        Y coordinates of the observations.
    vsini : np.ndarray
        vsini values (stellar rotation velocity) of the observations.
    B : int
        Number of bootstrap resamplings.
    trimpct : float
        Trimming percentage for trimmed mean.

    Returns:
    --------
    meanoriginal : np.ndarray
        Median of vsini from the original sample per space cell.
    boot_mean : np.ndarray
        Median of vsini from bootstrap resampling per space cell.
    largura : np.ndarray
        Width of the bootstrap confidence interval per space cell.
    shape : np.ndarray
        Shape index per space cell.
    """

    # Determine the number of cells along X and Y coordinates
    Nx = len(ttx) - 1
    Ny = len(tty) - 1

    # Initialize matrices to store results
    meanoriginal = np.zeros((Ny, Nx))   # Median of the original sample
    boot_mean = np.zeros((Ny, Nx))      # Median from bootstrap resampling
    boot_se = np.zeros((Ny, Nx))        # Bootstrap mean standard error
    ci1 = np.zeros((Ny, Nx))            # Lower confidence interval bootstrap
    ci2 = np.zeros((Ny, Nx))            # Upper confidence interval bootstrap
    countfXY = np.zeros((Ny, Nx))       # Observation count per space cell

    # Loop over each space cell
    for j in range(Nx):
        for i in range(Ny):
            # Condition to select observations within the cell
            condition = (ttx[j] <= x) & (x < ttx[j+1]) & (tty[i] <= y) & (y < tty[i+1])
            count = np.sum(condition)
            countfXY[i, j] = count

            # Calculate statistics if there are enough observations (>= 20)
            if count >= 20:
                sample_vsini = vsini[condition]
                meanoriginal[i, j] = np.median(sample_vsini)

                # Bootstrap resampling
                bootout = bootrsp(sample_vsini, B)  # Assuming bootrsp is defined elsewhere
                boot_trim_means = np.array([trimmean2(bootout[:, b], trimpct) for b in range(B)])
                boot_mean[i, j] = np.median(boot_trim_means)
                boot_se[i, j] = np.std(boot_trim_means)

                # Bootstrap confidence interval
                ci = np.percentile(boot_trim_means, [0.5, 99.5])
                ci1[i, j] = ci[0]
                ci2[i, j] = ci[1]

    # Calculate shape index
    largura = ci2 - ci1
    shape = (ci2 - boot_mean) / (boot_mean - ci1)
    shape[np.isnan(shape)] = 0

    return meanoriginal, boot_mean, largura, shape

def calculate_autocorrelation_2d(data: np.ndarray) -> np.ndarray:
    """
    Calculate the 2D autocorrelation function of input data.

    Parameters:
    -----------
    data : np.ndarray
        2D array of data.

    Returns:
    --------
    np.ndarray
        2D array representing the autocorrelation function.
    """
    n, m = data.shape
    B = np.abs(fftshift(ifft2(fft2(data) * np.conj(fft2(data))))) / (n * m)
    return B


def scramble_blocks(data: np.ndarray, block_size: int) -> np.ndarray:
    """
    Scramble the data by shuffling blocks of a specified size.

    Parameters:
    -----------
    data : np.ndarray
        2D array of data to be scrambled.
    block_size : int
        Size of each block for scrambling.

    Returns:
    --------
    np.ndarray
        Scrambled 2D array of data.
    """
    n_rows, n_cols = data.shape[0] // block_size, data.shape[1] // block_size
    scramble_blocks = [data[i:i + block_size, j:j + block_size] 
                       for i in range(0, n_rows, block_size) 
                       for j in range(0, n_cols, block_size)]
    np.random.shuffle(scramble_blocks)
    scramble = np.block([[scramble_blocks[i * n_cols + j] for j in range(n_cols)] 
                         for i in range(n_rows)])
    return scramble

