import sys
import os
import shutil
import array_to_latex as a2l

# import numpy as np
# import scipy.signal as signal
# import matplotlib.pyplot as plt


def savefig(fig, name):
    if not os.path.isdir("plots"):
        try:
            os.mkdir("plots")
        except OSError as error:
            print(error)
    fig.savefig(
        "plots/" + name + ".png",
        facecolor="white",
        transparent=False,
    )
    pass


def saveFigAndMoveToNotes(fig, name):
    savefig(fig, name)
    shutil.copy(
        "plots/" + name + ".png",
        "/Users/matthias/repos/phd_notes/admm/images/" + name + ".png",
    )
    pass


def writeMatrix(matrix, name, digits=0):
    frmt = "{:6.%df}" % (digits)
    with open(name + ".tex", "w") as text_file:
        stringu = a2l.to_ltx(matrix, frmt=frmt, arraytype="bmatrix", print_out=False)
        text_file.write(stringu)
