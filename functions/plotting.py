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

def plotfig_combined(original_dict, bootstrap_dict, ul, plane, data_selection, figure_number, OUTDIR):
    """
    original_dict: dictionary of original data to plot
    bootstrap_dict: dictionary of bootstrap data to plot
    ul: upper limit
    plane: 'XY', 'XZ', or 'ZY'
    data_selection: 'F', 'G', or 'ALL'
    figure_number: integer to use in the figure filename
    """
    fig, axs = plt.subplots(2, 4, figsize=(20, 8))
    method = 'spline36'

    Ny, Nx = next(iter(original_dict.values())).shape
    x_grid = np.arange(Nx)
    y_grid = np.arange(Ny)

    factor = 1
    xnew = np.linspace(0, Nx - 1, factor * Nx)
    ynew = np.linspace(0, Ny - 1, factor * Ny)

    cmap = 'jet'

    if plane == 'XY':
        xlabel = 'X [pc]'
        ylabel = 'Y [pc]'
    elif plane == 'XZ':
        xlabel = 'X [pc]'
        ylabel = 'Z [pc]'
    elif plane == 'ZY':
        xlabel = 'Z [pc]'
        ylabel = 'Y [pc]'

    all_titles = list(original_dict.keys()) + list(bootstrap_dict.keys())
    all_data = list(original_dict.values()) + list(bootstrap_dict.values())

    for idx, ax in enumerate(axs.flat):
        if idx >= len(all_data):
            fig.delaxes(ax)
            continue
        data_array = all_data[idx]
        spline = RectBivariateSpline(x_grid, y_grid, data_array.T, kx=3, ky=3)
        ima = spline(xnew, ynew)
        cax = ax.imshow(ima.T, cmap=cmap, aspect='auto', origin='lower', interpolation=method)
        cbar = fig.colorbar(cax, ax=ax, pad=0)
        cbar.set_label(all_titles[idx])
        cax.set_clim(0, np.nanmax(ima))
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)

        xticklabels = np.arange(-ul, ul + 1, 50)
        xticks = np.linspace(0, ima.shape[1] - 1, len(xticklabels))
        yticklabels = np.arange(-ul, ul + 1, 50)
        yticks = np.linspace(0, ima.shape[0] - 1, len(yticklabels))
        ax.set_xticks(xticks)
        ax.set_xticklabels(xticklabels)
        ax.set_yticks(yticks)
        ax.set_yticklabels(yticklabels)

    plt.tight_layout()
    plt.subplots_adjust(wspace=0.35, hspace=0.25)
    plt.savefig(f"{OUTDIR}/fig{figure_number}_{data_selection}_{plane}_combined.png", bbox_inches="tight", dpi=400)
    plt.close(fig)
    print(f">> Image successfully saved as 'fig{figure_number}_{data_selection}_{plane}_combined.png' in {OUTDIR} folder")

def plotfig_index(length_array, shape_array, ul, plane, data_selection, figure_number, OUTDIR):
    """
    length_array: 2D array of length indices
    shape_array: 2D array of shape indices
    ul: upper limit
    plane: 'XY', 'XZ', or 'ZY'
    data_selection: 'F', 'G', or 'ALL'
    figure_number: integer to use in the figure filename
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9, 4))
    method = 'spline36'

    Ny, Nx = length_array.shape
    x_grid = np.arange(Nx)
    y_grid = np.arange(Ny)
    xnew = np.linspace(0, Nx - 1, Nx)
    ynew = np.linspace(0, Ny - 1, Ny)

    cmap = 'jet'

    if plane == 'XY':
        xlabel = 'X [pc]'
        ylabel = 'Y [pc]'
    elif plane == 'XZ':
        xlabel = 'X [pc]'
        ylabel = 'Z [pc]'
    elif plane == 'ZY':
        xlabel = 'Z [pc]'
        ylabel = 'Y [pc]'

    for ax, data, title in zip([ax1, ax2], [length_array, shape_array], ['Length Index', 'Shape Index']):
        spline = RectBivariateSpline(x_grid, y_grid, data.T, kx=3, ky=3)
        ima = spline(xnew, ynew)
        im = ax.imshow(ima.T, cmap=cmap, origin='lower', interpolation=method, aspect='auto')
        cbar = fig.colorbar(im, ax=ax, pad=0)
        cbar.set_label(title)
        im.set_clim(0, np.nanmax(ima))
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)

        xticklabels = np.arange(-ul, ul + 1, 50)
        xticks = np.linspace(0, ima.shape[1] - 1, len(xticklabels))
        yticklabels = np.arange(-ul, ul + 1, 50)
        yticks = np.linspace(0, ima.shape[0] - 1, len(yticklabels))
        ax.set_xticks(xticks)
        ax.set_xticklabels(xticklabels)
        ax.set_yticks(yticks)
        ax.set_yticklabels(yticklabels)

    plt.tight_layout()
    plt.savefig(f"{OUTDIR}/fig{figure_number}_{data_selection}_{plane}_indices.png", bbox_inches="tight", dpi=400)
    plt.close(fig)
    print(f">> Image successfully saved as 'fig{figure_number}_{data_selection}_{plane}_indices.png' in {OUTDIR} folder")


def plotvsini(combined_data, filters, ylabels, PLANE, figure_number, OUTDIR):
    """
    Plots vsini subplots based on filters and labels for a given variable (GLON or GLAT).
    """
    # Create subplots for combined data
    fig = plt.figure(figsize=(9, 13))
    gs = fig.add_gridspec(3, 2, height_ratios=[1, 1, 0.7])
    axes = [fig.add_subplot(gs[i, j]) for i in range(2) for j in range(2)]

    for ax, f, ylabel in zip(axes, filters, ylabels):
        subset = combined_data[f]
        grouped = subset.groupby(["Source", "distance_bins"])["vsini"].quantile([0.25, 0.5, 0.75]).unstack()
        
        for source, color in zip(["F", "G"], ["tab:blue", "tab:red"]):
            if source in grouped.index.get_level_values(0):
                group = grouped.loc[source]
                distances = group.index.categories.mid
                p25, p50, p75 = group[0.25], group[0.5], group[0.75]
                ax.plot(distances, p50, label=f"{source} (Median & 25th & 75th)", color=color, 
                        linestyle="--", marker="s")
                ax.fill_between(distances, p25, p75, color=color, alpha=0.1)
        
        ax.set_ylabel(f"v sini [km/s] {ylabel}")
        ax.legend(loc="upper left")
        ax.set_xlim(10, 170)

    # Boxplot combining F and G datasets
    ax_box = fig.add_subplot(gs[2, :])
    palette = {"F": "tab:blue", "G": "tab:red"}
    sns.boxplot(x="distance_bins", y="vsini", hue="Source", data=combined_data, showfliers=False, 
                ax=ax_box, palette=palette)
    ax_box.tick_params(axis='x', which='major', labelsize=12)
    ax_box.set_xlabel(f"Distance [pc] in {PLANE} plane")
    ax_box.set_ylabel("$v \\sin i$ [km/s]")
    ax_box.legend(loc="upper left", ncols=2)

    # Final layout adjustments
    plt.tight_layout()
    plt.savefig(OUTDIR+f"/fig{figure_number}_{PLANE}.png", bbox_inches="tight", dpi=400)
    plt.close(fig)

def plot_vsini_lines(combined_data, filters_glon, labels_glon, filters_glat, labels_glat, PLANE, figure_number, OUTDIR):
    """
    Plots vsini lines for GLON and GLAT based on filters and labels."""
    fig, axes = plt.subplots(2, 4, figsize=(16, 8), sharex=True, sharey=True)
    plt.subplots_adjust(wspace=0.1)

    all_filters = [filters_glon, filters_glat]
    all_labels = [labels_glon, labels_glat]
    titles = ['GLON', 'GLAT']

    for row in range(2):
        filters = all_filters[row]
        ylabels = all_labels[row]

        for col, (f, ylabel) in enumerate(zip(filters, ylabels)):
            ax = axes[row, col]
            subset = combined_data[f]
            grouped = subset.groupby(["Source", "distance_bins"])["vsini"].quantile([0.25, 0.5, 0.75]).unstack()

            for source, color in zip(["F", "G"], ["tab:blue", "tab:red"]):
                if source in grouped.index.get_level_values(0):
                    group = grouped.loc[source]
                    distances = group.index.categories.mid
                    p25, p50, p75 = group[0.25], group[0.5], group[0.75]
                    ax.plot(distances, p50, label=f"{source} (Median & 25th & 75th)", color=color, linestyle="--", marker="s")
                    ax.fill_between(distances, p25, p75, color=color, alpha=0.1)

            ax.set_title(f"{titles[row]} {ylabel}")
            ax.set_xlim(10, 170)
            if col == 0:
                ax.set_ylabel("v sin i [km/s]")
            if row == 1:
                ax.set_xlabel("Distance [pc]")

    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(
        handles, labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.04), 
        ncol=2,
        fontsize=18  
    )
    plt.savefig(f"{OUTDIR}/fig{figure_number}_{PLANE}_GLON_GLAT.png", bbox_inches="tight", dpi=400)
    plt.close(fig)
    print(f">> Image successfully saved as 'fig{figure_number}_{PLANE}_GLON_GLAT.png' in {OUTDIR} folder")

def plot_vsini_boxplot(combined_data, PLANE, figure_number, OUTDIR):
    """
    Plots a boxplot of vsini values for different sources in the specified plane.
    """
    fig, ax = plt.subplots(figsize=(10, 4))
    palette = {"F": "tab:blue", "G": "tab:red"}
    sns.boxplot(x="distance_bins", y="vsini", hue="Source", data=combined_data,
                showfliers=False, ax=ax, palette=palette)
    ax.set_xlabel(f"Distance [pc] in {PLANE} plane")
    ax.set_ylabel(r"$v \sin i$ [km/s]")
    ax.legend(loc="upper left", ncols=2)
    ax.tick_params(axis='x', which='major', labelsize=12)

    plt.tight_layout()
    plt.savefig(f"{OUTDIR}/fig{figure_number}_{PLANE}_boxplot.png", bbox_inches="tight", dpi=400)
    plt.close(fig)
    print(f">> Image successfully saved as 'fig{figure_number}_{PLANE}_boxplot.png' in {OUTDIR} folder")


