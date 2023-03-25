import sys
import os
import shutil
import numpy as np
import array_to_latex as a2l
import matplotlib
import matplotlib.pyplot as plt


def savefig(
    fig, name, format="png", facecolor="white", transparent=False, pgf_font="sans-serif"
):
    if not os.path.isdir("plots"):
        try:
            os.mkdir("plots")
        except OSError as error:
            print(error)

    if format == "pgf":
        matplotlib.use("pgf")
        plt.rcParams.update(
            {
                "font.family": pgf_font,  # use serif/main font for text elements
                "text.usetex": True,  # use inline math for ticks
                "pgf.rcfonts": False,  # don't setup fonts from rc parameters
            }
        )
        if not os.path.isdir(f"plots/" + name):
            try:
                os.mkdir(f"plots/" + name)
            except OSError as error:
                print(error)
        fig.savefig(
            "plots/" + name + "/" + name + f".{format}",
            facecolor=facecolor,
            transparent=transparent,
            format=format,
        )
        with open("plots/" + name + "/" + name + ".tex", "w") as f:
            f.write(
                f"\\documentclass{{standalone}}\n\\usepackage{{pgf}}\n\\begin{{document}}\n\\input{{{name}.pgf}}\n\\end{{document}}"
            )
    else:
        fig.savefig(
            "plots/" + name + f".{format}",
            facecolor=facecolor,
            transparent=transparent,
            dpi=600,
            format=format,
        )
    pass


def saveFigAndMoveToNotes(fig, name, format="png"):
    savefig(fig, name, format)
    shutil.copy(
        "plots/" + name + f".{format}",
        "/Users/matthias/repos/phd_notes/admm/images/" + name + f".{format}",
    )
    pass


def set_size(
    width_pt, fraction=1, subplots=(1, 1), ratio=(5**0.5 - 1) / 2, dpi=72.27
):
    """Set figure dimensions to sit nicely in our document.

    Parameters
    ----------
    width_pt: float
            Document width in points
    fraction: float, optional
            Fraction of the width which you wish the figure to occupy
    subplots: array-like, optional
            The number of rows and columns of subplots.
    Returns
    -------
    fig_dim: tuple
            Dimensions of figure in inches
    """
    # Width of figure (in pts)
    fig_width_pt = width_pt * fraction
    # Convert from pt to inches
    inches_per_pt = 1 / dpi

    # # Golden ratio to set aesthetic figure height
    # golden_ratio =

    # Figure width in inches
    fig_width_in = fig_width_pt * inches_per_pt
    # Figure height in inches
    fig_height_in = fig_width_in * ratio * (subplots[0] / subplots[1])

    return (fig_width_in, fig_height_in)


def writeMatrix(matrix, name, digits=0):
    frmt = "{:6.%df}" % (digits)
    with open(name + ".tex", "w") as text_file:
        stringu = a2l.to_ltx(matrix, frmt=frmt, arraytype="bmatrix", print_out=False)
        text_file.write(stringu)


def NPM(a, b) -> np.float64:
    """
    Computes the Normalized Projection Misalignment (NPM) for two vectors a and b.

    Parameters
    ----------
    a: np.ndarray
            Column vector
    b: np.ndarray
            Column vector

    Returns
    -------
    NPM: np.float64
            Normalized Projection Misalignment
    """
    e = b - (b.conj().T @ a) / (a.conj().T @ a) * a
    return np.linalg.norm(e) / np.linalg.norm(b)
