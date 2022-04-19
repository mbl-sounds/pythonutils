import sys
import os
import shutil
import array_to_latex as a2l
from dataclasses import dataclass
import json

# import sparse
import scipy.sparse as sparse

# import stats
import scipy.stats as stats

import numpy as np

# import scipy.signal as signal
# import matplotlib.pyplot as plt

# @dataclass
# class CondorJob:


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
def generateRandomConnectionMatrix(M, density, rng=None):
    A = sparse.random(M, M, density=density, random_state=rng)
    A = A.toarray()
    A[A > 0] = 1
    return A


# %%
def generateRandomConnectionMatrixWithRing(M, density, rng=None):
    rd_c = sparse.random(M**2 - M * 2, 1, density=density, random_state=rng)
    rd_c = rd_c.toarray().squeeze()
    rd_c[rd_c > 0] = 1

    A = np.zeros((M, M))
    diag = np.diagonal(A, 1)
    diag.setflags(write=True)
    diag.fill(1)
    A[-1, 0] = 1
    p = 0
    for i in range(-M + 2, M):
        if i != 0 and i != 1:
            diag = np.diagonal(A, i)
            diag.setflags(write=True)
            p_ = p + diag.shape[0]
            diag[:] = rd_c[p:p_]
            p = p_

    return A


# %%
def runCondor(func, json_string):
    pass


def runLocal(func, json_string):
    pass
