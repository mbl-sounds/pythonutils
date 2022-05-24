import sys
import os
import shutil
import numpy as np
import array_to_latex as a2l
import matplotlib.pyplot as plt


def savefig(fig, name, format="png"):
    if not os.path.isdir("plots"):
        try:
            os.mkdir("plots")
        except OSError as error:
            print(error)
    fig.savefig(
        "plots/" + name + f".{format}",
        facecolor="white",
        transparent=False,
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


def writeMatrix(matrix, name, digits=0):
    frmt = "{:6.%df}" % (digits)
    with open(name + ".tex", "w") as text_file:
        stringu = a2l.to_ltx(matrix, frmt=frmt, arraytype="bmatrix", print_out=False)
        text_file.write(stringu)
