import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.ticker import AutoMinorLocator

# =====================================================
# LOAD DATA
# =====================================================

df = pd.read_csv("C:\\Users\\surya\\Downloads\\plots\\qtl.csv")
df.columns = df.columns.str.lower()

# =====================================================
# STYLE
# =====================================================

sns.set_theme(style="white")

plt.rcParams.update({
    "font.family": "Times New Roman",
    "font.size": 12,
    "axes.labelsize": 14,
    "axes.titlesize": 16,
    "legend.fontsize": 11,
    "xtick.labelsize": 11,
    "ytick.labelsize": 11
})

# =====================================================
# PROFESSIONAL COLORS
# =====================================================

COLOR_NOZNE = "#4E79A7"   # Muted Blue
COLOR_ZNE   = "#E15759"   # Muted Red
COLOR_GAIN  = "#59A14F"   # Muted Green

# =====================================================
# NOISE TYPES
# =====================================================

noise_order = [
    "pauli",
    "phase",
    "depolarizing"
]

# =====================================================
# COMMON Y RANGE (OPTIONAL)
# Makes all figures directly comparable
# =====================================================

all_vals = pd.concat([
    df["accuracy_no_zne"] * 100,
    df["accuracy_zne"] * 100
])

ymin = all_vals.min() - 3
ymax = all_vals.max() + 3

# =====================================================
# CREATE INDIVIDUAL FIGURES
# =====================================================

for noise in noise_order:

    subset = (
        df[df["noise_type"] == noise]
        .sort_values("noise_strength")
    )

    x = subset["noise_strength"]

    nozne = subset["accuracy_no_zne"] * 100
    zne   = subset["accuracy_zne"] * 100

    improvement = (
        (subset["accuracy_zne"]
         - subset["accuracy_no_zne"])
        /
        subset["accuracy_no_zne"]
        * 100
    )

    # =================================================
    # FIGURE
    # =================================================

    fig, ax = plt.subplots(
        figsize=(7, 5)
    )

    # =================================================
    # RECOVERY INTERVALS
    # =================================================

    cap_width = 0.004

    for xi, y_no, y_zne, imp in zip(
        x,
        nozne,
        zne,
        improvement
    ):

        # Vertical connector

        ax.vlines(
            xi,
            y_no,
            y_zne,
            color=COLOR_GAIN,
            linewidth=1.6,
            alpha=0.75,
            zorder=1
        )

        # Bottom cap

        ax.hlines(
            y_no,
            xi-cap_width,
            xi+cap_width,
            color=COLOR_GAIN,
            linewidth=1.6,
            alpha=0.75,
            zorder=1
        )

        # Top cap

        ax.hlines(
            y_zne,
            xi-cap_width,
            xi+cap_width,
            color=COLOR_GAIN,
            linewidth=1.6,
            alpha=0.75,
            zorder=1
        )

        # Improvement annotation

        ax.annotate(
            f"+{imp:.1f}%",
            xy=(xi, (y_no+y_zne)/2),
            xytext=(8, 0),
            textcoords="offset points",
            fontsize=10,
            fontweight='bold',
            color=COLOR_GAIN,
            ha='left',
            va='center',
            bbox=dict(
                facecolor='white',
                edgecolor='none',
                alpha=0.90,
                pad=0.25
            ),
            zorder=10
        )

    # =================================================
    # WITHOUT ZNE
    # =================================================

    ax.plot(
        x,
        nozne,
        '--o',
        color=COLOR_NOZNE,
        linewidth=2.0,
        markersize=5,
        markerfacecolor='white',
        markeredgewidth=1.2,
        label='Without ZNE',
        zorder=5
    )

    # =================================================
    # WITH ZNE
    # =================================================

    ax.plot(
        x,
        zne,
        '-D',
        color=COLOR_ZNE,
        linewidth=2.2,
        markersize=5,
        markeredgewidth=1.0,
        label='With ZNE',
        zorder=6
    )

    # =================================================
    # AXES
    # =================================================

    ax.set_xlabel(
        "Noise Strength"
    )

    ax.set_ylabel(
        "Accuracy (%)"
    )

    ax.set_xticks(
        [0.02, 0.15, 0.25]
    )

    ax.set_ylim(
        ymin,
        ymax
    )

    # =================================================
    # MAJOR GRID
    # =================================================

    ax.grid(
        True,
        which='major',
        axis='both',
        linestyle='--',
        linewidth=0.6,
        color='lightgray',
        alpha=0.7
    )

    # =================================================
    # MINOR GRID
    # =================================================

    ax.xaxis.set_minor_locator(
        AutoMinorLocator()
    )

    ax.yaxis.set_minor_locator(
        AutoMinorLocator()
    )

    ax.grid(
        which='minor',
        linestyle=':',
        linewidth=0.4,
        color='lightgray',
        alpha=0.35
    )

    # =================================================
    # SPINES
    # =================================================

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # =================================================
    # LEGEND
    # =================================================

    ax.legend(
        frameon=False,
        loc='best'
    )

    plt.tight_layout()

    # =================================================
    # SAVE
    # =================================================

    plt.savefig(
        f"{noise}_accuracy_recovery_journal.png",
        dpi=600,
        bbox_inches='tight'
    )

    plt.show()