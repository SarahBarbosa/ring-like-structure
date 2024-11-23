import os
import numpy  as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import pearsonr, spearmanr

from functions import *

# User-defined parameters
TC = 20                         # Increment value
UL = 150                        # Upper limit of parameter interval
DATA_SELECTION = 'G'          # Options: 'F', 'G', 'ALL'
PLANE = 'XZ'                    # Options: 'XY', 'XZ', 'ZY'
B = 1000                        # Number of bootstrap samples
INTERVAL = 1                    # Grid interval
CL = 95                         # Confidence level
TRIMPCT = 20                    # Trim percentage
ALPHA = 0.01                    # Alpha for confidence intervals

ALL = pd.read_csv("data/gcs-allstars.csv")
F = pd.read_csv("data/gcs-Fstars.csv")
G = pd.read_csv("data/gcs-Gstars.csv")

OLDSCHOOL = True

################################################################################

if OLDSCHOOL:
    import smplotlib
else:
    plt.rcParams.update({
        "xtick.top": True,
        "ytick.right": True,
        "xtick.direction": "in",
        "font.family": "Lato",
        "ytick.direction": "in",
        "font.size": 12,
        "axes.labelsize": 12,
        "axes.titlesize": 12,
        "legend.fontsize": 10,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "xtick.minor.visible": True,
        "ytick.minor.visible": True
    })

OUTDIR = "figures"
if not os.path.exists(OUTDIR):
    os.makedirs(OUTDIR)

################################### FIGURE 1 ###################################

def sort_data_by_vsini(dataframe):
    """
    Sort a DataFrame by the 'vsini' column in ascending order.
    """
    sorted_data = dataframe.sort_values(by="vsini", ascending=True)
    return sorted_data["GLON"], sorted_data["GLAT"], sorted_data["vsini"]

glon_F, glat_F, vsini_F = sort_data_by_vsini(F)
glon_G, glat_G, vsini_G = sort_data_by_vsini(G)

# Plotting
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(6, 10), sharex=True, sharey=True)

def plot1(glon, glat, vsini, label, ax, scaler=2):
    scatter = ax.scatter(
        glon, glat, s=vsini * scaler, c=vsini,
        cmap="jet", ec="k", lw=0.5
    )
    ax.set(xlim=(0, 361), ylim=(-90, 90))
    ax.text(0.05, 0.95, label, transform=ax.transAxes, fontsize=12,
            verticalalignment='top', bbox=dict(facecolor='white', edgecolor='black'))
    return scatter

scatter_F = plot1(glon_F, glat_F, vsini_F, "F-type", ax1)
scatter_G = plot1(glon_G, glat_G, vsini_G, "G-type", ax2)

ax2.set(xlabel="Galactic Longitude $l$ [degree]")
fig.text(-0.02, 0.5, "Galactic Latitude $b$ [degree]", va='center', rotation='vertical')

# Add colorbars
plt.colorbar(scatter_F, ax=ax1, pad=0).set_label(r"$v\sin i$ [km/s]")
plt.colorbar(scatter_G, ax=ax2, pad=0).set_label(r"$v\sin i$ [km/s]")

plt.tight_layout()
plt.savefig(f"{OUTDIR}/fig1_FG.png", bbox_inches="tight", dpi=400)
print(f">> Image successfully saved as 'fig1_FG.png' in {OUTDIR} folder")

################################### FIGURE 2 ###################################

x = F["X"]
y = F["Y"]
z = F["Z"]

bins = 20           # 20x20 pc² pixels
vmin, vmax = 0, 40  # Colobar limits (40 stars into the pixel)

fig, axs = plt.subplots(1, 3, figsize=(15, 4))

data_pairs = [
    (x, y, 'X [pc]', 'Y [pc]'),
    (x, z, 'X [pc]', 'Z [pc]'),
    (z, y, 'Z [pc]', 'Y [pc]')
]

for ax, (data1, data2, xlabel, ylabel) in zip(axs, data_pairs):
    hist, xedges, yedges = np.histogram2d(data1, data2, bins=bins, range=[[-150, 150], [-150, 150]])
    cax = ax.pcolormesh(xedges, yedges, hist.T, cmap='jet', vmin=vmin, vmax=vmax)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_xlim(-150, 150)
    ax.set_ylim(-150, 150)
    ax.set_facecolor(plt.cm.jet(0))
    fig.colorbar(cax, ax=ax, label='Number of Stars', pad=0)

plt.tight_layout()
plt.subplots_adjust(wspace=0.4)
plt.savefig(f"{OUTDIR}/fig2_F.png", bbox_inches="tight", dpi=400)
print(f">> Image successfully saved as 'fig2_FG.png' in {OUTDIR} folder")

################################### FIGURE 3 ###################################

vmag_F = F["Vmag"]
vmag_G = G["Vmag"]

falloff_f = 8.5  # Decline for F-type stars
falloff_g = 9.0  # Decline for G-type stars

bin_edges = np.arange(2, 12, 0.5)

_, ax = plt.subplots()

# Plot histogram for F-type stars
ax.hist(vmag_F, bins=bin_edges, ec="tab:blue", histtype="step", label="F-type")
ax.set_xlim(2, 12)

# Plot histogram for G-type stars
ax.hist(vmag_G, bins=bin_edges, ec="tab:red", histtype="step", label="G-type")

# Add vertical lines for fall-off points
ax.axvline(falloff_f, color='tab:gray', linestyle='--')
ax.axvline(falloff_g, color='tab:gray', linestyle='--')

# Labeling and legend
ax.set(xlabel="Apparent magnitude [mag]", ylabel="Number of stars")
ax.legend(loc="upper left")

# Adjust layout and save the figure
plt.tight_layout()
plt.savefig(f"{OUTDIR}/fig3_FG.png", bbox_inches="tight", dpi=400)
print(f">> Image successfully saved as 'fig3_FG.png' in {OUTDIR} folder")

######################################################################

# Sample 0-80 pc for F
x1 = F[F["Dist"] < 80]["Dist"].values
y1 = F[F["Dist"] < 80]["vsini"].values
s1, i1, e1 = linear_regression(x1,y1)

# Sample 80-350 pc  for F
x2 = F[F["Dist"] > 80]["Dist"].values
y2 = F[F["Dist"] > 80]["vsini"].values
s2, i2, e2 = linear_regression(x2,y2)

print(f"\n0-80 slope F-stars slope (95%): {s1:.3f} +/- {e1:.3f}")
print(f"80-350 slope F-stars slope (95%): {s2:.3f} +/- {e2:.3f}")

# Sample 0-80 pc for G
x1g = G[G["Dist"] < 80]["Dist"].values
y1g = G[G["Dist"] < 80]["vsini"].values
s1g, i1g, e1g = linear_regression(x1g,y1g)

# Sample 80-350 pc  for G
x2g = G[G["Dist"] > 80]["Dist"].values
y2g = G[G["Dist"] > 80]["vsini"].values
s2g, i2g, e2g = linear_regression(x2g,y2g)

print(f"\n0-80 slope F-stars slope (95%): {s1g:.3f} +/- {e1g:.3f}")
print(f"80-350 slope F-stars slope (95%): {s2g:.3f} +/- {e2g:.3f}")

################################### FIGURE 4 ###################################

X = np.linspace(0, 400, 100)

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(6, 10), sharex=True, sharey=True)

# Scatter plots for F
ax1.scatter(F["Dist"], F["vsini"], ec = "white", fc = "tab:gray", lw = 0.2, s = 20)
ax1.set_ylim(0, 140)
ax1.set_xlim(0, 350)

line1, = ax1.plot(X, i1 + s1 * X, "tab:blue", lw=1.5, 
                label="0-80 pc slope", ls="--")
line2, = ax1.plot(X, i2 + s2 * X, "tab:red", 
                lw=1.5, label="80-350 pc slope", ls="--")

label_line(line1, x = 280, fontsize=13, color = "tab:blue", 
                backgroundcolor = "white")
label_line(line2, x = 280, fontsize=13, color = "tab:red", 
                backgroundcolor = "white")

ax1.text(0.05, 0.95, "F-type", transform=ax1.transAxes, fontsize=12,
        verticalalignment='top', bbox=dict(facecolor='white', edgecolor='black'))

# and for G
ax2.scatter(G["Dist"], G["vsini"], ec = "white", fc = "tab:gray", lw = 0.2, s = 20)

line1, = ax2.plot(X, i1g + s1g * X, "tab:blue", lw=1.5, 
                label="0-80 pc slope", ls="--")
line2, = ax2.plot(X, i2g + s2g * X, "tab:red", 
                lw=1.5, label="80-350 pc slope", ls="--")

label_line(line1, x = 280, fontsize=13, color = "tab:blue", 
                backgroundcolor = "white")
label_line(line2, x = 280, fontsize=13, color = "tab:red", 
                backgroundcolor = "white")

ax2.text(0.05, 0.95, "G-type", transform=ax2.transAxes, fontsize=12,
        verticalalignment='top', bbox=dict(facecolor='white', edgecolor='black'))

fig.text(-0.02, 0.5, r"$v \sin i$ [km/s]", va='center', rotation='vertical')
plt.xlabel(r"Distance $R$ [pc]")

plt.tight_layout()
plt.savefig(f"{OUTDIR}/fig4_FG.png", bbox_inches="tight", dpi=400)
print(f"\n>> Image successfully saved as 'fig4_FG.png' in {OUTDIR} folder")

######################################################################

# Pearson correlation coefficient
pearson_corr_all, pearson_p_value_all = pearsonr(ALL['Vmag'], ALL['vsini'])
pearson_corr, pearson_p_value = pearsonr(F['Vmag'], F['vsini'])
pearson_corrg, pearson_p_valueg = pearsonr(G['Vmag'], G['vsini'])

# Spearman rank correlation coefficient
spearman_corr_all, spearman_p_value_all = spearmanr(ALL['Vmag'], ALL['vsini'])
spearman_corr, spearman_p_value = spearmanr(F['Vmag'], F['vsini'])
spearman_corrg, spearman_p_valueg = spearmanr(G['Vmag'], G['vsini'])

print(f"\nPearson correlation coefficient for all stars: {pearson_corr_all}")
print(f"Pearson p-value for all stars: {pearson_p_value_all}\n")

print(f"Pearson correlation coefficient for F stars: {pearson_corr}")
print(f"Pearson p-value for F stars: {pearson_p_value}\n")

print(f"Pearson correlation coefficient for G stars: {pearson_corrg}")
print(f"Pearson p-value for G stars: {pearson_p_valueg}")

print(f"\nSpearman correlation coefficient for all stars: {spearman_corr_all}")
print(f"Spearman p-value for all stars: {spearman_p_value_all}\n")

print(f"Spearman correlation coefficient for F stars: {spearman_corr}")
print(f"Spearman p-value for F stars: {spearman_p_value}\n")

print(f"Spearman correlation coefficient for G stars: {spearman_corrg}")
print(f"Spearman p-value for G stars: {spearman_p_valueg}")

################################### FIGURE 5 ###################################

if DATA_SELECTION == 'F':
    data = F
elif DATA_SELECTION == 'G':
    data = G
elif DATA_SELECTION == 'ALL':
    data = ALL
else:
    raise ValueError("Invalid data selection. Choose 'F', 'G', or 'ALL'.")

# Extract variables based on plane
if PLANE == 'XY':
    x = data["X"]
    y = data["Y"]
elif PLANE == 'XZ':
    x = data["X"]
    y = data["Z"]
elif PLANE == 'ZY':
    x = data["Z"]
    y = data["Y"]
else:
    raise ValueError("Invalid plane selection. Choose 'XY', 'XZ', or 'ZY'.")

vsini = data["vsini"]
DPtotal = np.std(vsini)

ttx = np.arange(-UL, UL + INTERVAL * TC, INTERVAL * TC)
tty = np.arange(-UL, UL + INTERVAL * TC, INTERVAL * TC)
Nx = len(ttx) - 1
Ny = len(tty) - 1

# Initialize arrays
meanbootstrap = np.zeros((Ny, Nx))
meanbootstrapSD = np.zeros((Ny, Nx))
percentile25boot = np.zeros((Ny, Nx))
percentile50boot = np.zeros((Ny, Nx))
percentile75boot = np.zeros((Ny, Nx))

meanoriginal = np.zeros((Ny, Nx))
meanoriginalSD = np.zeros((Ny, Nx))
percentile25 = np.zeros((Ny, Nx))
percentile50 = np.zeros((Ny, Nx))
percentile75 = np.zeros((Ny, Nx))

boot_mean = np.zeros((Ny, Nx))
boot_se = np.zeros((Ny, Nx))
ci1 = np.zeros((Ny, Nx))
ci2 = np.zeros((Ny, Nx))
shape = np.zeros((Ny, Nx))
countfXY = np.zeros((Ny, Nx))

for j in range(Nx):
    for i in range(Ny):
        condition = (
            (ttx[j] <= x) & (x < ttx[j + 1]) &
            (tty[i] <= y) & (y < tty[i + 1])
        )
        vsini_condition = vsini[condition]
        count = len(vsini_condition)
        countfXY[i, j] = count

        if count >= 20:
            # Calculate statistics for original data
            meanoriginal[i, j] = np.mean(vsini_condition)
            meanoriginalSD[i, j] = np.std(vsini_condition)
            percentile25[i, j] = np.percentile(vsini_condition, 25)
            percentile50[i, j] = np.percentile(vsini_condition, 50)
            percentile75[i, j] = np.percentile(vsini_condition, 75)

            # Generate bootstrap samples
            samples = bootrsp(vsini_condition.values, B)  # Shape: [N, B]
            bootout = np.array([trimmean2(samples[:, b], TRIMPCT) for b in range(B)])

            # Compute bootstrap statistics
            #boot_true = trimmean2(vsini_condition, TRIMPCT)
            boot_mean[i, j] = np.mean(bootout)
            boot_se[i, j] = np.std(bootout)
            meanbootstrap[i, j] = boot_mean[i, j]
            meanbootstrapSD[i, j] = boot_se[i, j]

            # Compute confidence intervals
            lower_percentile = (ALPHA / 2) * 100
            upper_percentile = (1 - ALPHA / 2) * 100
            ci1[i, j] = np.percentile(bootout, lower_percentile)
            ci2[i, j] = np.percentile(bootout, upper_percentile)

            # Percentiles for bootstrap samples
            percentile25boot[i, j] = np.percentile(bootout, 25)
            percentile50boot[i, j] = np.percentile(bootout, 50)
            percentile75boot[i, j] = np.percentile(bootout, 75)

# Calculate shape and length indices
length = ci2 - ci1
shape = (ci2 - boot_mean) / (boot_mean - ci1)
shape = np.nan_to_num(shape)  # Replace NaN with zero

# Print completion message
print("\n##### Bootstrap and grid analysis completed. #####")

# First set of plots
data_dict1 = {
    r'$\langle v \sin i \rangle$ [km/s] of original sample': meanoriginal,
    r'$v \sin i$ [km/s] (q = 1/4)': percentile25,
    r'$v \sin i$ [km/s] (q = 1/2)': percentile50,
    r'$v \sin i$ [km/s] (q = 3/4)': percentile75
}
plotfig(data_dict1, UL, PLANE, DATA_SELECTION, figure_number=5, OUTDIR=OUTDIR)
print(f"\n>> Image successfully saved as 'fig5_{DATA_SELECTION}_{PLANE}.png' in {OUTDIR} folder")

################################### FIGURE 6 ###################################

data_dict2 = {
    r'$\langle v \sin i \rangle$ [km/s]' + '\n' + 'of bootstrap resampling': meanbootstrap,
    r'$v \sin i$ [km/s]' + '\n' + 'of bootstrap resampling (q=1/4)': percentile25boot,
    r'$v \sin i$ [km/s]' + '\n' + 'of bootstrap resampling (q=1/2)': percentile50boot,
    r'$v \sin i$ [km/s]' + '\n' + 'of bootstrap resampling (q=3/4)': percentile75boot
}
plotfig(data_dict2, UL, PLANE, DATA_SELECTION, figure_number=6, OUTDIR=OUTDIR)
print(f">> Image successfully saved as 'fig6_{DATA_SELECTION}_{PLANE}.png' in {OUTDIR} folder")

################################### FIGURE 7 ###################################

data_dict3 = {
    r'$\langle v \sin i \rangle$ [km/s]' + '\n' + 'of original sample': meanoriginal,
    r'$\langle v \sin i \rangle$ [km/s]' + '\n' + 'of bootstrap resampling': boot_mean,
    'Length Index': length,
    'Shape Index': shape
}
plotfig(data_dict3, UL, PLANE, DATA_SELECTION, figure_number=7, OUTDIR=OUTDIR)
print(f">> Image successfully saved as 'fig7_{DATA_SELECTION}_{PLANE}.png' in {OUTDIR} folder")

################################### FIGURE 8 ###################################

bins_range = np.arange(0, 300, 30)  # Distance bins

def process_data(data, PLANE):
    data_copy = data.copy()
    data_copy["XY"] = np.sqrt(data_copy["X"] ** 2 + data_copy["Y"] ** 2)
    data_copy["XZ"] = np.sqrt(data_copy["X"] ** 2 + data_copy["Z"] ** 2)
    data_copy["ZY"] = np.sqrt(data_copy["Z"] ** 2 + data_copy["Y"] ** 2)
    data_clean = remove_outliers(data_copy, "vsini")  # Remove outliers using IQR

    # Create distance bins based on the selected plane
    if PLANE == 'XY':
        distance_column = 'XY'
    elif PLANE == 'XZ':
        distance_column = 'XZ'
    elif PLANE == 'ZY':
        distance_column = 'ZY'
    else:
        raise ValueError("Invalid plane selection. Choose 'XY', 'XZ', or 'ZY'.")
    
    data_clean["distance_bins"] = pd.cut(data_clean[distance_column], bins_range)
    
    return data_clean

# Process F and G datasets
F_clean = process_data(F, PLANE)
G_clean = process_data(G, PLANE)

# Add source labels and combine datasets
F_clean["Source"] = "F"
G_clean["Source"] = "G"
combined_data = pd.concat([F_clean, G_clean], ignore_index=True)

# Filters and labels for Galactic Longitude (GLON)
filters_glon = [
    (combined_data["GLON"] < 90),
    (combined_data["GLON"] >= 90) & (combined_data["GLON"] < 180),
    (combined_data["GLON"] >= 180) & (combined_data["GLON"] < 270),
    (combined_data["GLON"] >= 270),
]
ylabels_glon = ["(l < 90°)", "(90° ≤ l < 180°)", "(180° ≤ l < 270°)", "(l ≥ 270°)"]

# Plot vsini subplots for GLON
plotvsini(combined_data, filters_glon, ylabels_glon, PLANE, 8, OUTDIR)
print(f">> Image successfully saved as 'fig8_{PLANE}.png' in {OUTDIR} folder")

################################### FIGURE 9 ###################################

# Filters and labels for Galactic Latitude (GLAT)
filters_glat = [
    (combined_data["GLAT"] < -45),
    (combined_data["GLAT"] >= -45) & (combined_data["GLAT"] < 0),
    (combined_data["GLAT"] >= 0) & (combined_data["GLAT"] < 45),
    (combined_data["GLAT"] >= 45),
]
ylabels_glat = ["(b < -45°)", "(-45° ≤ b < 0°)", "(0° ≤ b < 45°)", "(b ≥ 45°)"]

# Plot vsini subplots for GLAT
plotvsini(combined_data, filters_glat, ylabels_glat, PLANE, 9, OUTDIR)
print(f">> Image successfully saved as 'fig9_{PLANE}.png' in {OUTDIR} folder")

