# %%
import os
import simulation
import pandas as pd
import matplotlib.pyplot as plt

# %%
# Simulation function for testing
def simulationFunction(algorithm, mean, length, run: int, seed: int):
    import numpy as np

    rng = np.random.default_rng(np.random.PCG64DXSM(seed).jumped(run))
    # print(f"run {run} with algorithm={algorithm}, mean={mean}, length={length}")
    data = rng.normal(loc=mean, size=(length,))
    return data


#%%
if __name__ == "__main__":
    cfg = simulation.SimConfig(
        id="test",
        runs=12,
        seed=1234,
        variables=[
            {"algorithm": ["ADMM"], "mean": [2], "length": [100]},
            {"algorithm": ["LMS"], "mean": [2, 30]},
        ],
    )

    # %%
    sim = simulation.Simulation(cfg, os.path.realpath(__file__), simulationFunction)

    # %%
    sim.clearTmpData()

    # %%
    sim.runLocal(nprocesses=4, showprogress=True)

    # %%
    result = sim.getResult()

    # %%
    plt.plot(result.df.groupby(["algorithm", "mean", "length"]).mean().to_numpy().T)
    plt.show()

    # %%
    sim.clearTmpData()

    # %%
    user_submit = {
        "request_walltime": "100",
        "initialdir": ".",
        "notification": "Error",
        "executable": "/users/sista/mblochbe/python_venvs/admmstuff/bin/python",
        "request_cpus": "1",
        "request_memory": "1GB",
    }
    sim.runCondor(user_submit)

    # %%
    print(sim.isDone())

    # %%
    result = sim.getResult()

    # %%
    plt.plot(result.df.groupby(["algorithm", "mean", "length"]).mean().to_numpy().T)
    plt.show()

# %%
