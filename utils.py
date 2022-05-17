import sys
import os
import shutil
import numpy as np
import array_to_latex as a2l
import scipy.sparse as sparse
import scipy.stats as stats
import scipy.signal as signal
import matplotlib.pyplot as plt


def savefig(fig, name):
    if not os.path.isdir("plots"):
        try:
            os.mkdir("plots")
        except OSError as error:
            print(error)
    fig.savefig("plots/" + name + ".png", facecolor="white", transparent=False, dpi=300)
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


# %%
def generateConnectionMatrix(M, density, rng=None):
    A = sparse.random(M, M, density=density, random_state=rng)
    A = A.toarray()
    A[A > 0] = 1
    return A
