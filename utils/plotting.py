import numpy as np
from math import atan2, degrees

import matplotlib.pyplot as plt
import matplotlib.lines as mlines
from matplotlib.axes import Axes

from typing import Tuple

def plot_hist(ax: Axes, x: np.ndarray, y: np.ndarray, bins: int, xlabel: str, ylabel: str, fig: plt.Figure) -> Tuple:
    """
    Plots a 2D histogram on the given axes.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        The axes on which to plot the histogram.
    x : numpy.ndarray
        The data for the x-axis.
    y : numpy.ndarray
        The data for the y-axis.
    xlabel : str
        The label for the x-axis.
    ylabel : str
        The label for the y-axis.
    fig : matplotlib.figure.Figure
        The figure to which the colorbar will be added.

    Returns
    -------
    Tuple
        The return value of `ax.hist2d` which includes the histogram.
    """
    h = ax.hist2d(x, y, bins=(bins, bins), cmap="coolwarm", range=np.array([[-150, 150], [-150, 150]]))

    cbar = fig.colorbar(h[3], ax=ax, pad=0)
    cbar.ax.text(0.5, 1.05, "# of stars", transform=cbar.ax.transAxes, ha="center", fontsize=11)
    cbar.ax.tick_params(labelsize=12)
    cbar.ax.minorticks_off()

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)

    ax.set_facecolor(plt.cm.viridis(0))
    return h


def labelLine(line: mlines.Line2D, x: float, label: str = None, align: bool = True, **kwargs) -> None:
    """
    Label a line at a specified x-coordinate with its label data.

    Parameters
    ----------
    line : mlines.Line2D
        The line object to label.
    x : float
        The x-coordinate where the label should be placed.
    label : str, optional
        The label text. If None, the line's label is used. Default is None.
    align : bool, optional
        Whether to align the label with the slope of the line. Default is True.
    **kwargs
        Additional keyword arguments passed to `ax.text`.

    Returns
    -------
    None
    
    Examples
    --------
    >>> import matplotlib.pyplot as plt
    >>> fig, ax = plt.subplots()
    >>> line, = ax.plot([0, 1], [0, 1], label='Sample Line')
    >>> labelLine(line, x=0.5)
    >>> plt.show()
    """
    ax = line.axes
    xdata = line.get_xdata()
    ydata = line.get_ydata()

    if (x < xdata[0]) or (x > xdata[-1]):
        print("x label location is outside data range!")
        return

    # Find corresponding y-coordinate and angle of the line
    ip = 1
    for i in range(len(xdata)):
        if x < xdata[i]:
            ip = i
            break

    y = ydata[ip - 1] + (ydata[ip] - ydata[ip - 1]) * (x - xdata[ip - 1]) / (xdata[ip] - xdata[ip - 1])

    if not label:
        label = line.get_label()

    if align:
        # Compute the slope
        dx = xdata[ip] - xdata[ip - 1]
        dy = ydata[ip] - ydata[ip - 1]
        ang = degrees(atan2(dy, dx))

        # Transform to screen coordinates
        pt = np.array([x, y]).reshape((1, 2))
        trans_angle = ax.transData.transform_angles(np.array([ang]), pt)[0]
    else:
        trans_angle = 0

    # Set a bunch of keyword arguments
    if "color" not in kwargs:
        kwargs["color"] = line.get_color()

    if "horizontalalignment" not in kwargs and "ha" not in kwargs:
        kwargs["ha"] = "center"

    if "verticalalignment" not in kwargs and "va" not in kwargs:
        kwargs["va"] = "center"

    if "backgroundcolor" not in kwargs:
        kwargs["backgroundcolor"] = ax.get_facecolor()

    if "clip_on" not in kwargs:
        kwargs["clip_on"] = True

    if "zorder" not in kwargs:
        kwargs["zorder"] = 2.5

    ax.text(x, y, label, rotation=trans_angle, **kwargs)

def labelLines(lines: list[mlines.Line2D], align: bool = True, xvals: np.ndarray = None, **kwargs) -> None:
    """
    Label multiple lines at specified x-coordinates.

    Parameters
    ----------
    lines : list of mlines.Line2D
        List of line objects to label.
    align : bool, optional
        Whether to align the labels with the slope of the lines. Default is True.
    xvals : np.ndarray, optional
        Array of x-coordinates where the labels should be placed. If None, the labels are
        evenly spaced across the x-axis range. Default is None.
    **kwargs
        Additional keyword arguments passed to `labelLine`.

    Returns
    -------
    None
    
    Examples
    --------
    >>> import matplotlib.pyplot as plt
    >>> fig, ax = plt.subplots()
    >>> line1, = ax.plot([0, 1], [0, 1], label='Line 1')
    >>> line2, = ax.plot([0, 1], [1, 0], label='Line 2')
    >>> labelLines([line1, line2])
    >>> plt.show()
    """
    ax = lines[0].axes
    labLines = []
    labels = []

    # Take only the lines which have labels other than the default ones
    for line in lines:
        label = line.get_label()
        if "_line" not in label:
            labLines.append(line)
            labels.append(label)

    if xvals is None:
        xmin, xmax = ax.get_xlim()
        xvals = np.linspace(xmin, xmax, len(labLines) + 2)[1:-1]

    for line, x, label in zip(labLines, xvals, labels):
        labelLine(line, x, label, align, **kwargs)

def plot_surface(ax: plt.Axes, X: np.ndarray, Y: np.ndarray, Z: np.ndarray, xlabel: str, ylabel: str, title: str):
    """
    Plot a 3D surface plot on the given axes.

    Parameters:
    -----------
    ax : plt.Axes
        Axes object where the plot will be drawn.
    X : np.ndarray
        2D array of X coordinates.
    Y : np.ndarray
        2D array of Y coordinates.
    Z : np.ndarray
        2D array of Z values (amplitudes).
    xlabel : str
        Label for the X-axis.
    ylabel : str
        Label for the Y-axis.
    title : str
        Title of the plot.
    """
    ax.plot_surface(X, Y, Z, cmap='coolwarm', edgecolor='none')
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_zlabel('Amplitude')
    ax.set_title(title)
    ax.set_xticks(np.arange(-150, 151, 50))
    ax.set_yticks(np.arange(-150, 151, 50))
    ax.view_init(azim=20, elev=30)