import sys
import os
import shutil
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from typing import Tuple

DATA_DIRECTORY = "/esat/stadiustempdatasets/mblochbe"


def printProgressBar(
    iteration,
    total,
    prefix="",
    suffix="",
    decimals=1,
    length=100,
    fill="█",
    printEnd="\r",
):
    percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
    filledLength = int(length * iteration // total)
    bar = fill * filledLength + "-" * (length - filledLength)
    print(f"\r{prefix} |{bar}| {percent}% {suffix}", end=printEnd, flush=True)


def progressBar(
    iterable,
    total,
    prefix="",
    suffix="",
    decimals=1,
    length=100,
    fill="█",
    printEnd="\r",
):
    """
    Call in a loop to create terminal progress bar
    @params:
        iterable    - Required  : iterable object (Iterable)
        prefix      - Optional  : prefix string (Str)
        suffix      - Optional  : suffix string (Str)
        decimals    - Optional  : positive number of decimals in percent complete (Int)
        length      - Optional  : character length of bar (Int)
        fill        - Optional  : bar fill character (Str)
        printEnd    - Optional  : end character (e.g. "\r", "\r\n") (Str)
    """
    # total = len(iterable)

    # Progress Bar Printing Function
    def printProgressBar(iteration):
        percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
        filledLength = int(length * iteration // total)
        bar = fill * filledLength + "-" * (length - filledLength)
        print(f"\r{prefix} |{bar}| {percent}% {suffix}", end=printEnd, flush=True)

    # Initial Call
    printProgressBar(0)
    # Update Progress Bar
    for i, item in enumerate(iterable):
        yield item
        printProgressBar(i + 1)
    # Print New Line on Complete
    print()


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


def set_size(width_pt, fraction=1, subplots=(1, 1), ratio=(5**0.5 - 1) / 2):
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
    # Convert from pt to inches (this is not DPI!!)
    latex_inches_per_pt = 1 / 72.27

    # # Golden ratio to set aesthetic figure height
    # golden_ratio =

    # Figure width in inches
    fig_width_in = fig_width_pt * latex_inches_per_pt
    # Figure height in inches
    fig_height_in = fig_width_in * ratio * (subplots[0] / subplots[1])

    return (fig_width_in, fig_height_in)


def NPM(a: np.ndarray, b: np.ndarray) -> np.float64:
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


H_TRUE: np.ndarray = None
F_OPT: np.ndarray = None


def NFPM(h_est: np.ndarray, h_true: np.ndarray, L: int, M: int, Df: int = 199):
    """
    Computes the Normalized Filter-Projection Misalignment (NFPM) for two vectors a and b.

    Parameters
    ----------
    a: np.ndarray
            Column vector
    b: np.ndarray
            Column vector
    L: int
            length of single IR
    M: int
            number of channels
    Df: int
            length of common filter

    Returns
    -------
    NFPM: np.float64
            Normalized Filter-Projection Misalignment
    """
    global H_TRUE
    global F_OPT
    assert h_true.shape == h_est.shape, "estimate and true dont have the same length"
    assert h_est.shape[0] == M * L, "estimate is not correct length M*L"
    assert h_true.shape[0] == M * L, "ground truth is not correct length M*L"

    if np.any(H_TRUE != h_true) or F_OPT is None:
        print("recalc")
        H = []
        for i in range(M):
            H_i = np.zeros((L + 2 * Df, 2 * Df + 1))
            for n in range(2 * Df + 1):
                H_i[n : n + L, n, None] = h_true[i * L : (i + 1) * L]
            H.append(H_i)
        H = np.concatenate(H)
        H_TRUE = h_true
        F_OPT = H @ np.linalg.inv(H.T @ H) @ H.T

    h_tilde = []
    for i in range(M):
        h_i_tilde = np.pad(
            h_est[i * L : (i + 1) * L],
            ((Df, Df), (0, 0)),
        )
        h_tilde.append(h_i_tilde)

    h_tilde = np.concatenate(h_tilde)

    nfpm = np.linalg.norm(h_tilde - F_OPT @ h_tilde) / np.linalg.norm(h_tilde)
    return nfpm


def generateNonStationaryNoise(
    num_samples: int, fs: float, rng=np.random.default_rng()
) -> np.ndarray:
    """
    Generates WGN with randomized segments of variance mimicking the envelope of speech
    (not that well though)

    Parameters
    ----------
    num_samples: int
            number of signal samples to generate
    fs: float
            sampling rate
    rng: Generator
            random number generator

    Returns
    -------
    signal: np.ndarray
            generated signal
    """
    # TODO formant filters
    # TODO glottal pulse train
    # "silence" in the beginning
    length = int(0.1 * fs)
    signal = np.zeros((length,))
    while length < num_samples:
        dur = int(rng.uniform(0.01 * fs, 0.2 * fs))
        # amp = rng.uniform(0.0, 1.0)
        # freq = rng.uniform(200, 6000)
        noise = rng.uniform(0.0001, 0.1)
        # noise = 0
        length += dur
        # t = np.linspace(0, (dur) / fs, dur) * freq * 2 * np.pi
        env_part = rng.normal(size=(dur,), scale=noise)  # + np.sin(t) * amp
        signal = np.concatenate([signal, env_part])

    signal = signal[:num_samples].reshape(num_samples, 1)

    return signal / signal.max()


def generateRandomIRs(
    L, N, a, rng=np.random.default_rng()
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generates random impulse responses WGN

    Parameters
    ----------
    L: int
            length of IRs
    N: int
            number of channels/IRs
    a: Array-like
            target norm values of IRs (must have N elements)
    rng: Generator
            The random generator to be used

    Returns
    -------
    h: np.ndarray
            LxN array containing the generated IRs
    hf: np.ndarray
            LxN array containing the L-sample FFT of generated IRs
    """
    h = np.zeros((L, N))
    hf = np.zeros((L, N), dtype=np.complex128)
    for n in range(N):
        h_ = rng.normal(size=(L, 1))
        h_ = h_ / np.linalg.norm(h_) * a[n]
        h[:, n, None] = h_
        hf[:, n, None] = np.fft.fft(h_, n=L, axis=0)
    return h, hf


# def generateRandomIRs(
#     L, N, a, nz, delta, rng=np.random.default_rng()
# ) -> Tuple[np.ndarray, np.ndarray]:
#     """
#     Generates random impulse responses (WGN) with a defined number of (near-common) zeros

#     Parameters
#     ----------
#     L: int
#             length of IRs
#     N: int
#             number of channels/IRs
#     a: Array-like
#             target norm values of IRs (must have N elements)
#     nz: int
#             number of (near-)common zeros
#     delta: float
#             distance of near common zeros
#     rng: Generator
#             The random generator to be used

#     Returns
#     -------
#     h: np.ndarray
#             LxN array containing the generated IRs
#     hf: np.ndarray
#             LxN array containing the L-sample FFT of generated IRs
#     """
#     h = np.zeros((L, N))
#     hf = np.zeros((L, N), dtype=np.complex128)
#     for n in range(N):
#         h_ = rng.normal(size=(L, 1))
#         h_ = h_ / np.linalg.norm(h_) * a[n]
#         h[:, n, None] = h_
#         hf[:, n, None] = np.fft.fft(h_, n=L, axis=0)
#     return h, hf


def getNoisySignalOld(
    signal: np.ndarray,
    IRs: np.ndarray,
    SNR: np.float64,
    rng=np.random.default_rng(),
    noise_type="white",
) -> np.ndarray:
    """
    Convolves clean signal with IRs and adds noise at SNR.

    Parameters
    ----------
    signal: np.ndarray
            clean signal (single channel)
    IRs: np.ndarray
            LxM array containing IRs
    a: Array-like
            target norm values of IRs (must have N elements)
    rng: Generator
            The random generator to be used

    Returns
    -------
    noisy_signal: np.ndarray
            convolved and noisy signals
    """
    N_s = signal.shape[0]
    L = IRs.shape[0]
    M = IRs.shape[1]

    var_s = np.var(signal)

    noisy_signal = np.zeros(shape=(N_s + L - 1, M))
    n_var = 10 ** (-SNR / 10) * var_s * np.linalg.norm(IRs) ** 2 / M
    for m in range(M):
        noisy_signal[:, m] = np.convolve(signal.squeeze(), IRs[:, m].squeeze()) + np.sqrt(
            n_var
        ) * rng.normal(size=(N_s + L - 1,))
    return noisy_signal


def getNoisySignal(
    signal: np.ndarray,
    IRs: np.ndarray,
    SNR: np.float64,
    rng=np.random.default_rng(),
    noise_cov: np.ndarray = None,
) -> np.ndarray:
    """
    Convolves clean signal with IRs and adds noise at SNR.

    Parameters
    ----------
    signal: np.ndarray
            clean signal (single channel)
    IRs: np.ndarray
            LxM array containing IRs
    a: Array-like
            target norm values of IRs (must have N elements)
    rng: Generator
            The random generator to be used

    Returns
    -------
    noisy_signal: np.ndarray
            convolved and noisy signals
    """
    N_s = signal.shape[0]
    L = IRs.shape[0]
    M = IRs.shape[1]

    noise_cov = np.eye(M * L) if noise_cov is None else noise_cov

    # s_ = np.concatenate([np.zeros(shape=(L - 1, 1)), signal])

    convolved_signal = np.zeros(shape=(N_s + L - 1, M))
    for m in range(M):
        convolved_signal[:, m] = np.convolve(signal.squeeze(), IRs[:, m].squeeze())
    var_s = np.var(convolved_signal)
    n_var = 10 ** (-SNR / 10) * var_s
    noise_cov = noise_cov / np.linalg.norm(noise_cov) * n_var
    noise = None
    if n_var > 0:
        noise = generateMCNoise(N_s + L - 1, L, M, noise_cov=noise_cov, rng=rng)
        noisy_signal = convolved_signal + noise
    else:
        noisy_signal = convolved_signal
    return noisy_signal, noise, noise_cov


def generateMCNoise(
    n: int,
    L: int,
    M: int,
    noise_cov: np.ndarray,
    noise_mean: np.ndarray = None,
    noise_dist: str = "normal",
    rng=np.random.default_rng(),
) -> np.ndarray:
    assert noise_cov.shape == (
        M * L,
        M * L,
    ), f"The noise covariance has to be (M*L, M*L)=\
        ({M*L}, {M*L})"
    noise_mean = np.zeros((noise_cov.shape[0],)) if noise_mean == None else noise_mean
    assert noise_mean.shape == (
        M * L,
    ), f"The noise mean has to be (M*L, )=\
        ({M*L}, )"
    size = noise_cov.shape[0]
    # noise = np.zeros((n, M))
    samples = int(n / L) + 1
    match noise_dist:
        case "normal":
            X = np.random.normal(size=(samples, M * L)).T
            for nn in range(X.shape[0]):
                X[nn] = X[nn] - X[nn].mean()

            L_inv = np.linalg.cholesky(np.cov(X))
            L_inv = np.linalg.inv(L_inv)
            X = np.dot(L_inv, X)

            L_ = np.linalg.cholesky(noise_cov)
            X = np.dot(L_, X)
            for nn in range(X.shape[0]):
                X[nn] = X[nn] + noise_mean[nn]
            noise = []
            for m in range(M):
                noise.append(X[m * L : (m + 1) * L, :].T.reshape((samples * L,)))
            noise = np.asarray(noise)
    return noise[:, :n].T


def conj_grad_EV(
    A: np.ndarray,
    epsilon: float = 1e-10,
    max_iter: int = None,
    precond: np.ndarray = None,
    x_start: np.ndarray = None,
    verbose=False,
    return_error=False,
):
    x = np.random.normal(size=(A.shape[0], 1)) if x_start is None else x_start
    # x = x / np.linalg.norm(x)
    M = precond
    mult = False
    w = A @ x
    l = (w.T @ x).squeeze()
    g = w - l * x
    if M is None:
        p = g
    else:
        mult = M.shape == x.shape
        p = M * g if mult else M @ g
    z = A @ p

    error = None
    if return_error:
        error = []

    i = 0
    while True:
        a = (z.T @ x).squeeze()
        b = (z.T @ p).squeeze()
        c = (x.T @ p).squeeze()
        d = (p.T @ p).squeeze()

        delta = (l * d - b) ** 2 - 4 * (b * c - a * d) * (a - l * c)
        alpha = (l * d - b + np.sqrt(delta)) / (2 * (b * c - a * d))
        l = (l + a * alpha) / (1 + c * alpha)
        gamma = np.sqrt(1 + 2 * c * alpha + d * alpha**2)

        x = (x + alpha * p) / gamma
        w = (w + alpha * z) / gamma
        g = w - l * x

        measure = np.linalg.norm(g) / l
        if verbose:
            print("Error:", measure)
        if return_error:
            error.append(measure)
        if measure < epsilon or (max_iter is not None and i >= max_iter):
            break

        if M is None:
            beta = -g.T @ z / b
            p = g + beta * p
        else:
            beta = -g.T @ (M * z) / b if mult else -g.T @ M @ z / b
            p = M * g + beta * p if mult else M @ g + beta * p

        z = A @ p

        i += 1

    return x, error


def conj_grad_GEV(
    A: np.ndarray,
    B: np.ndarray,
    epsilon: float = 1e-10,
    max_iter: int = None,
    precond: np.ndarray = None,
    x_start: np.ndarray = None,
    verbose=False,
    return_error=False,
):
    assert A.shape == B.shape, "A and B have to have the same size!"
    x = np.random.normal(size=(A.shape[0], 1)) if x_start is None else x_start
    x = x / np.linalg.norm(x)
    M = precond
    mult = False
    w = A @ x
    l = (w.T @ x).squeeze()
    g = w - l * x
    if M is None:
        p = g
    else:
        mult = M.shape == x.shape
        p = M * g if mult else M @ g
    z = A @ p

    error = None
    if return_error:
        error = []

    i = 0
    while True:
        a = (z.T @ x).squeeze()
        b = (z.T @ p).squeeze()
        c = (x.T @ B @ p).squeeze()
        d = (p.T @ B @ p).squeeze()

        delta = (l * d - b) ** 2 - 4 * (b * c - a * d) * (a - l * c)
        alpha = (l * d - b + np.sqrt(delta)) / (2 * (b * c - a * d))
        l = (l + a * alpha) / (1 + c * alpha)
        gamma = np.sqrt(1 + 2 * c * alpha + d * alpha**2)

        x = (x + alpha * p) / gamma
        w = (w + alpha * z) / gamma
        g = A @ x - l * B @ x

        measure = np.linalg.norm(g) / l
        if verbose:
            print("Error:", measure)
        if return_error:
            error.append(measure)
        if measure < epsilon or (max_iter is not None and i >= max_iter):
            break

        if M is None:
            beta = -g.T @ z / b
            p = g + beta * p
        else:
            beta = -g.T @ (M * z) / b if mult else -g.T @ M @ z / b
            p = M * g + beta * p if mult else M @ g + beta * p
        z = A @ p
        i += 1

    return x, error


def discreteEntropy(x: np.ndarray, base: float = 2) -> float:
    """
    Computes the discrete entropy -sum(pn*log2(pn))

    Parameters
    ----------
    x: np.ndarray
            array, of which the entropy should be computed.
            Rows are considered elements to test for uniqueness.

    Returns
    -------
    ent: float
        Discrete entropy
    """
    assert base in [2, np.e, 10], "base must be either 2, e, or 10"
    L = x.shape[0]
    _, counts = np.unique(np.round(x * 1e20) / 1e20, return_counts=True, axis=0)
    pn = counts / L
    if base == 10:
        pnl = np.log10(pn)
    if base == np.e:
        pnl = np.log(pn)
    if base == 2:
        pnl = np.log2(pn)

    ent = -np.sum(pn * pnl)
    return ent


def generateRandomWSNTop(
    num_nodes: int,
    room_dim: list | np.ndarray,
    num_closest: int = 0,
    margin: float = 0.5,
    num_sources: int = 0,
    rng=np.random.default_rng(),
) -> Tuple[np.ndarray, list] | Tuple[np.ndarray, list, np.ndarray]:
    """
    generates a random WASN network topology within a room

    Parameters
    ----------
    num_nodes: int
            Number of nodes in the network
    room_dim: list, ndarray
            room dimensions within which the nodes are placed
    num_closest: int
            Number of minumum connections per node (closest neighbors)
            If no value given, no connections generated
    margin: float
            minimum distance from room walls
    rng: Generator
            random number generator


    Returns
    -------
    node_pos: ndarray
            Node positions in 3d space
    network_edges: list
            list of edges (connections) between nodes
    """
    node_pos = np.asarray(
        [
            rng.uniform(0.0 + margin, room_dim[0] - margin, size=(num_nodes,)),
            rng.uniform(0.0 + margin, room_dim[1] - margin, size=(num_nodes,)),
            rng.uniform(0.0 + margin, room_dim[2] - margin, size=(num_nodes,)),
        ]
    ).T

    network_edges = []
    for i in range(num_nodes):
        d = []
        for j in range(num_nodes):
            d.append(np.linalg.norm(node_pos[i, :] - node_pos[j, :]))
        idx = np.argsort(np.asarray(d))
        for kk in range(num_closest):
            if [idx[kk + 1], i] not in network_edges:
                network_edges.append([i, idx[kk + 1]])

    if num_sources > 0:
        src_pos = np.asarray(
            [
                rng.uniform(0.0 + margin, room_dim[0] - margin, size=(num_sources,)),
                rng.uniform(0.0 + margin, room_dim[1] - margin, size=(num_sources,)),
                rng.uniform(0.0 + margin, room_dim[2] - margin, size=(num_sources,)),
            ]
        ).T

        return (node_pos, network_edges, src_pos)
    return (node_pos, network_edges)


def getPart(index, part_len) -> int:
    return int(np.floor(index / part_len))


def erank(A: np.ndarray):
    """
    Computes the effective rank of the matrix A as proposed in
    Olivier Roy and Martin Vetterli, The effective rank: A measure of effective dimensionality

    Parameters
    ----------
    A: ndarray
            The matrix


    Returns
    -------
    erank: float
            effective rank
    """
    _, s, _ = np.linalg.svd(A)
    p_k = s / np.linalg.norm(s, 1)
    H = -np.sum(p_k * np.log(p_k))
    return np.exp(H)


def nrank(A: np.ndarray, epsilon: float = np.finfo, normalized=False):
    """
    Computes the epsilon-numerical rank of the matrix A
    r_n = min{ r: sigma_r <= epsilon}

    Parameters
    ----------
    A: ndarray
            The matrix
    epsilon: float
            threshold of singular values to be counted
    normalized: bool
            whether the singular values should be normalized before applying threshold


    Returns
    -------
    erank: int
            epsilon-numerical rank
    """
    _, s, _ = np.linalg.svd(A)
    if normalized:
        return np.sum(s / np.max(s) > epsilon)
    return np.sum(s > epsilon)


def generateRandomRootSet(delta, rng=np.random.default_rng()):
    points = Bridson_sampling(np.array([2, 1]), radius=delta, rng=rng)
    points -= np.array([1, 0])
    points = points[np.linalg.norm(points, axis=1) < 1, :]
    return points[:, 0] + 1j * points[:, 1]


def _selectRandomRoots(root_set: np.ndarray, L, M, rng=np.random.default_rng()):
    index_set = np.arange(root_set.shape[0])
    roots = []
    assert (
        len(index_set) >= M * L
    ), "Not enough roots to choose from. Can't place enough roots with minimal distance {delta}!"
    for m in range(M):
        selection = rng.choice(index_set, L, False)
        roots.append(np.concatenate([root_set[selection], root_set[selection].conj()]))
        remainder = [i for i in index_set if i not in selection]
        index_set = remainder

    return np.asarray(roots)


def generateRandomIRsFromRootset(root_set: np.ndarray, L, M, rng=np.random.default_rng()):
    selected_roots = _selectRandomRoots(root_set, int(L / 2), M, rng=rng)
    h = np.zeros((L, M))
    hf = np.zeros((L, M), dtype=np.complex128)
    for n in range(M):
        h_ = np.poly(selected_roots[n, :])
        # h_ = h_ / np.linalg.norm(h_) * a[n]
        h[:, n, None] = h_[1:, None]
        hf[:, n, None] = np.fft.fft(h_[1:, None], n=L, axis=0)
    return h, hf


from scipy.special import gammainc


# Uniform sampling in a hyperspere
# Based on Matlab implementation by Roger Stafford
# Can be optimized for Bridson algorithm by excluding all points within the r/2 sphere
def hypersphere_volume_sample(center, radius, k=1, rng=np.random.default_rng()):
    ndim = center.size
    x = rng.normal(size=(k, ndim))
    ssq = np.sum(x**2, axis=1)
    fr = radius * gammainc(ndim / 2, ssq / 2) ** (1 / ndim) / np.sqrt(ssq)
    frtiled = np.tile(fr.reshape(k, 1), (1, ndim))
    p = center + np.multiply(x, frtiled)
    return p


# Uniform sampling on the sphere's surface
def hypersphere_surface_sample(center, radius, k=1, rng=np.random.default_rng()):
    ndim = center.size
    vec = rng.standard_normal(size=(k, ndim))
    vec /= np.linalg.norm(vec, axis=1)[:, None]
    p = center + np.multiply(vec, radius)
    return p


def squared_distance(p0, p1):
    return np.sum(np.square(p0 - p1))


def Bridson_sampling(
    dims=np.array([1.0, 1.0]),
    radius=0.05,
    k=30,
    hypersphere_sample=hypersphere_volume_sample,
    rng=np.random.default_rng(),
):
    # References: Fast Poisson Disk Sampling in Arbitrary Dimensions
    #             Robert Bridson, SIGGRAPH, 2007

    ndim = dims.size

    # size of the sphere from which the samples are drawn relative to the size of a disc (radius)
    sample_factor = 2
    if hypersphere_sample == hypersphere_volume_sample:
        sample_factor = 2

    # for the surface sampler, all new points are almost exactly 1 radius away from at least one existing sample
    # eps to avoid rejection
    if hypersphere_sample == hypersphere_surface_sample:
        eps = 0.001
        sample_factor = 1 + eps

    def in_limits(p):
        return np.all(np.zeros(ndim) <= p) and np.all(p < dims)

    # Check if there are samples closer than "squared_radius" to the candidate "p"
    def in_neighborhood(p, n=2):
        indices = (p / cellsize).astype(int)
        indmin = np.maximum(indices - n, np.zeros(ndim, dtype=int))
        indmax = np.minimum(indices + n + 1, gridsize)

        # Check if the center cell is empty
        if not np.isnan(P[tuple(indices)][0]):
            return True
        a = []
        for i in range(ndim):
            a.append(slice(indmin[i], indmax[i]))
        if np.any(np.sum(np.square(p - P[tuple(a)]), axis=ndim) < squared_radius):
            return True

    def add_point(p):
        points.append(p)
        indices = (p / cellsize).astype(int)
        P[tuple(indices)] = p

    cellsize = radius / np.sqrt(ndim)
    gridsize = (np.ceil(dims / cellsize)).astype(int)

    # Squared radius because we'll compare squared distance
    squared_radius = radius * radius

    # Positions of cells
    P = np.empty(
        np.append(gridsize, ndim), dtype=np.float32
    )  # n-dim value for each grid cell
    # Initialise empty cells with NaNs
    P.fill(np.nan)

    points = []
    add_point(rng.uniform(np.zeros(ndim), dims))
    while len(points):
        i = rng.integers(len(points))
        p = points[i]
        del points[i]
        Q = hypersphere_sample(np.array(p), radius * sample_factor, k, rng)
        for q in Q:
            if in_limits(q) and not in_neighborhood(q):
                add_point(q)
    return P[~np.isnan(P).any(axis=ndim)]
