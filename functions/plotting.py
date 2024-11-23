import numpy as np

from math import atan2, degrees

import seaborn as sns
import matplotlib.lines as mlines
import matplotlib.pyplot as plt

from scipy.interpolate import RectBivariateSpline

def label_line(line: mlines.Line2D, x: float, label: str = None, align: bool = True, **kwargs):
    """
    Label a line at a specified x-coordinate.

    Parameters
    ----------
    line : mlines.Line2D
        The line object to label.
    x : float
        The x-coordinate where the label should be placed.
    label : str, optional
        Label text. If None, the line's label is used.
    align : bool, optional
        Whether to align the label with the slope of the line.
    """
    ax = line.axes
    xdata = line.get_xdata()
    ydata = line.get_ydata()

    if (x < xdata[0]) or (x > xdata[-1]):
        print("x label location is outside data range!")
        return

    ip = 1
    for i in range(len(xdata)):
        if x < xdata[i]:
            ip = i
            break

    y = ydata[ip - 1] + (ydata[ip] - ydata[ip - 1]) * (x - xdata[ip - 1]) / (xdata[ip] - xdata[ip - 1])

    if not label:
        label = line.get_label()

    if align:
        dx = xdata[ip] - xdata[ip - 1]
        dy = ydata[ip] - ydata[ip - 1]
        ang = degrees(atan2(dy, dx))
        pt = np.array([x, y]).reshape((1, 2))
        trans_angle = ax.transData.transform_angles(np.array([ang]), pt)[0]
    else:
        trans_angle = 0

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

def plotfig(data_dict, ul, plane, data_selection, figure_number, OUTDIR):
    """
    data_dict: dictionary of data to plot, keys are the titles, values are 2D arrays
    ul: upper limit
    plane: 'XY', 'XZ', or 'ZY'
    data_selection: 'F', 'G', or 'ALL'
    figure_number: integer to use in the figure filename
    """
    # Create figure and subplots
    fig, axs = plt.subplots(2, 2, figsize=(11, 9))
    method = 'spline36'

    # Original grid dimensions
    Ny, Nx = next(iter(data_dict.values())).shape
    x_grid = np.arange(Nx)
    y_grid = np.arange(Ny)

    # Interpolation factor (increase grid density)
    factor = 1  # Adjust this factor as needed
    xnew = np.linspace(0, Nx - 1, factor * Nx)
    ynew = np.linspace(0, Ny - 1, factor * Ny)

    # Define the colormap to be consistent across plots
    cmap = 'jet'

    # Titles for subplots
    subplot_titles = list(data_dict.keys())
    data_arrays = list(data_dict.values())

    # Labels for axes based on plane
    if plane == 'XY':
        xlabel = 'X [pc]'
        ylabel = 'Y [pc]'
    elif plane == 'XZ':
        xlabel = 'X [pc]'
        ylabel = 'Z [pc]'
    elif plane == 'ZY':
        xlabel = 'Z [pc]'
        ylabel = 'Y [pc]'

    # Iterate over subplots
    for idx, ax in enumerate(axs.flat):
        if idx >= len(data_arrays):
            fig.delaxes(ax)
            continue
        data_array = data_arrays[idx]
        spline = RectBivariateSpline(x_grid, y_grid, data_array.T, kx=3, ky=3)
        ima = spline(xnew, ynew)
        cax = ax.imshow(ima.T, cmap=cmap, aspect='auto', origin='lower',
                        interpolation=method)
        cbar = fig.colorbar(cax, ax=ax, pad=0)
        cbar.set_label(subplot_titles[idx])
        cax.set_clim(0, np.nanmax(ima))
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)

        # Adjust axis tick labels
        xticklabels = np.arange(-ul, ul + 1, 50)
        xticks = np.linspace(0, ima.shape[1] - 1, len(xticklabels))
        yticklabels = np.arange(-ul, ul + 1, 50)
        yticks = np.linspace(0, ima.shape[0] - 1, len(yticklabels))
        ax.set_xticks(xticks)
        ax.set_xticklabels(xticklabels)
        ax.set_yticks(yticks)
        ax.set_yticklabels(yticklabels)

    plt.tight_layout()
    plt.subplots_adjust(wspace=0.4)
    plt.savefig(OUTDIR+f"/fig{figure_number}_{data_selection}_{plane}.png", bbox_inches="tight", dpi=400)
    plt.close(fig)

def plotvsini(data, filters, bins_range, ylabels, PLANE, OUTDIR, data_selection, figure_name):
    """
    Plots vsini subplots based on filters and labels for a given variable (GLON or GLAT).
    """
    # Configure the layout for subplots and boxplot
    fig = plt.figure(figsize=(8, 12))
    gs = fig.add_gridspec(3, 2, height_ratios=[1, 1, 0.7])

    # Plot the subplots
    axes = [fig.add_subplot(gs[i, j]) for i in range(2) for j in range(2)]
    for ax, f, ylabel in zip(axes, filters, ylabels):
        # Filter data
        subset = data[f]

        # Calculate percentiles for each bin
        grouped = subset.groupby("distance_bins")["vsini"].quantile([0.25, 0.5, 0.75]).unstack()

        # Get distances and percentiles
        distances = grouped.index.categories.mid
        p25 = grouped[0.25]
        p50 = grouped[0.5]
        p75 = grouped[0.75]

        # Plot percentiles
        ax.plot(distances, p50, label="Median", color="k", linestyle="--", marker="x")
        ax.set_xlim(8, 150)
        ax.set_ylim(subset["vsini"].min(), subset["vsini"].max())

        # Fill between 25th and 75th
        ax.fill_between(distances, p25, p75, color="gray", alpha=0.3, label="25th & 75th")

        # Subplot settings
        ax.set_ylabel(f"v sini [km/s] after bias \n reduction {ylabel}")
        ax.set_xlabel("")  # Remove X label for subplots
        ax.legend(loc="upper left")

    # Plot boxplot below the subplots
    ax_box = fig.add_subplot(gs[2, :])
    sns.boxplot(x="distance_bins", y="vsini", data=data, showfliers=False, ax=ax_box, color="gray")
    ax_box.set_xlabel(f"Distance [pc] in Galactic plane ({PLANE})")
    ax_box.tick_params(axis='x', which='major', labelsize=12)
    ax_box.set_ylabel("$v \\sin i$ [km/s]")

    # Adjust layout
    plt.tight_layout()
    plt.savefig(OUTDIR + f"/{figure_name}_{data_selection}_{PLANE}.png", bbox_inches="tight", dpi=400)
    plt.close(fig)

