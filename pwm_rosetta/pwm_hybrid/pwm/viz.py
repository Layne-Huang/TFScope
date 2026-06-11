"""
Sequence-logo and PWM visualisation utilities.

``logomaker`` is an optional dependency — a clear error is raised at
call-time if it is not installed.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter
from matplotlib import cm
from scipy.stats import entropy


def seqToOneHot(seq):
    """Convert a DNA sequence string to a one-hot array of shape (L, 4)."""
    mapping = dict(zip("ACGT", range(4)))
    seq2 = [mapping[i] for i in seq]
    return np.eye(4)[seq2]


def makeLogo(gt, ax):
    """
    Draw an information-content sequence logo on *ax*.

    Parameters
    ----------
    gt : np.ndarray, shape (L, 4)
        Position probability matrix (ACGT order).
    ax : matplotlib.axes.Axes

    Returns
    -------
    logo : logomaker.Logo
    """
    try:
        import logomaker as lm
    except ImportError:
        raise ImportError(
            "logomaker is required for sequence logo plots. "
            "Install it with: pip install logomaker"
        )
    import pandas as pd

    e = 2 - entropy(gt.T, base=2)
    gt_scaled = gt * e[:, np.newaxis]

    gt_df = pd.DataFrame(
        {'A': gt_scaled[:, 0], 'C': gt_scaled[:, 1],
         'G': gt_scaled[:, 2], 'T': gt_scaled[:, 3]}
    )

    colors = {'A': 'green', 'C': 'blue', 'G': '#FBA922', 'T': 'red'}
    logo = lm.Logo(
        gt_df,
        color_scheme=colors,
        ax=ax,
        show_spines=False,
        fade_probabilities=False,
        font_name='DejaVu Sans'
    )
    return logo


def plotPWM(Z, ax, fontsize=10, xaxis=False):
    """
    Plot a per-base probability heatmap.

    Parameters
    ----------
    Z : np.ndarray, shape (4, L)
        PWM/PPM matrix with bases on axis 0 (ACGT order).
    ax : matplotlib.axes.Axes
    fontsize : int
    xaxis : bool
        Whether to show x-axis tick labels.

    Returns
    -------
    ax, im
    """
    cmaps = [
        cm.get_cmap('Greens'),
        cm.get_cmap('Blues'),
        cm.get_cmap('Oranges'),
        cm.get_cmap('Reds'),
    ]
    C = []
    for i in range(Z.shape[0]):
        row_colors = [cmaps[i](Z[i, j]) for j in range(Z.shape[1])]
        C.append(row_colors)
    C = np.array(C)

    im = ax.imshow(C, aspect='auto')
    if not xaxis:
        ax.set_xticks([])
    ax.set_yticks([0, 1, 2, 3])
    ax.set_yticklabels(['A', 'C', 'G', 'T'], fontsize=fontsize)
    ax.tick_params(axis='both', which='both', length=0)
    return ax, im


def plot(ppm, seq=None, outpath='.', filename_prefix='motif',
         highlight_positions=None, highlight_color='red',
         position_range=None, show_only_positions=None):
    """
    Plot a combined sequence-logo visualisation and save to *outpath*.

    Parameters
    ----------
    ppm : array-like, shape (L, 4)
        Position probability matrix (ACGT order).
    seq : str, optional
        DNA sequence (used only for display).
    outpath : str
        Directory for the saved figure.
    filename_prefix : str
        Filename prefix (without extension).
    highlight_positions : list[int], optional
        1-based positions to highlight.
    highlight_color : str
        Colour string for highlighted positions.
    position_range : tuple[int, int], optional
        (start, end) slice to show (1-based, inclusive).
    show_only_positions : list[int], optional
        Specific 1-based positions to display.
    """
    from matplotlib.patches import Patch

    ppm = np.array(ppm)

    # Subset the matrix if requested
    if position_range is not None:
        start, end = position_range
        ppm = ppm[start - 1:end]
        if seq is not None:
            seq = seq[start - 1:end]
        if highlight_positions is not None:
            highlight_positions = [
                p - start + 1 for p in highlight_positions if start <= p <= end
            ]
    elif show_only_positions is not None:
        idx_to_keep = [p - 1 for p in show_only_positions]
        ppm = ppm[idx_to_keep]
        if seq is not None:
            seq = ''.join([seq[p - 1] for p in show_only_positions])
        highlight_positions = None

    # Convert highlight_positions to 0-based
    if highlight_positions is not None:
        highlight_positions = [p - 1 for p in highlight_positions]

    fig, ax = plt.subplots(figsize=(6, 3.5))
    makeLogo(ppm, ax)
    ax.set_ylabel("bits")
    ax.set_xlabel("Positions")
    ax.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
    ax.set_ylim(0, 2)

    if position_range is not None:
        title = f"Motif Logo (positions {position_range[0]}-{position_range[1]})"
    elif show_only_positions is not None:
        title = f"Motif Logo (positions {show_only_positions})"
    elif highlight_positions is not None:
        title = f"Motif Logo (highlighted: {[p + 1 for p in highlight_positions]})"
    else:
        title = "Motif Information Logo"
    ax.set_title(title)

    ax.set_xticks(range(len(ppm)))
    if position_range is not None:
        ax.set_xticklabels(range(position_range[0], position_range[1] + 1))
    elif show_only_positions is not None:
        ax.set_xticklabels(show_only_positions)
    else:
        ax.set_xticklabels(range(1, len(ppm) + 1))

    if highlight_positions is not None:
        for pos in highlight_positions:
            if 0 <= pos < len(ppm):
                ax.axvline(x=pos + 0.5, color=highlight_color,
                           linestyle='--', alpha=0.5, linewidth=1)
                ax.axvspan(pos, pos + 1, color=highlight_color, alpha=0.15)
        legend_elements = [
            Patch(facecolor=highlight_color, alpha=0.15,
                  edgecolor=highlight_color, label='Highlighted Position')
        ]
        ax.legend(handles=legend_elements, loc='upper right',
                  fontsize='small', framealpha=0.9)

    plt.tight_layout()
    os.makedirs(outpath, exist_ok=True)
    save_path = os.path.join(outpath, f"{filename_prefix}_logo.png")
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()
    print(f"Saved: {save_path}")
