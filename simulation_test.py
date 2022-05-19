# %%
# ######################################################################
import simulation
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# %%
cfg = simulation.SimConfig(
    id="test",
    runs=5,
    seed=1234,
    variables={"algorithm": ["LMS", "ADMM"], "variance": [2, 30], "length": [100]},
)

# %%
# Simulation function for testing
def simulationFunction(a, b, c, run: int, seed: int):
    import numpy as np

    rng = np.random.default_rng(np.random.PCG64DXSM(seed).jumped(run))
    print(f"run {run} with algorithm={a}, variance={b}, length={c}")
    data = rng.normal(loc=b, size=(c,))
    return data


# %%
sim = simulation.Simulation(cfg, simulationFunction)

# %%
sim.clearTmpData()

# %%
sim.runLocal()

# %%
result = sim.getResult()

# %%
plt.plot(result.df.groupby(["algorithm", "variance", "length"]).mean().to_numpy().T)
plt.show()

# %%
import htcondor
import classad
import os
import pickle
import json

# %%
dir_path = os.path.dirname(os.path.realpath(__file__))

# %%
python_job = htcondor.Submit(
    {
        "universe": "Vanilla",
        "request_walltime": "100",
        "nice_user": "true",
        "initialdir": ".",
        "notification": "Error",
        "executable": "/users/sista/mblochbe/python_venvs/admmstuff/bin/python",
        "arguments": f"sim_execute.py $(ProcId) $(tmppath) $(func) \\'$(args)\\' $(seed)",  # sleep for 10 seconds
        "output": f"{cfg.id}-$(ProcId).out",  # output and error for each job, using the $(ProcId) macro
        "error": f"{cfg.id}-$(ProcId).err",
        "log": f"{cfg.id}.log",  # we still send all of the HTCondor logs for every job to the same file (not split up!)
        "request_cpus": "1",
        "request_memory": "128MB",
        "request_disk": "128MB",
    }
)

print(python_job)

# %%
pickle.dump(simulationFunction, open("func.p", "wb"))

# %%
seed = 1234
index = pd.MultiIndex.from_product(
    [*cfg.variables.values(), [*range(cfg.runs)]], names=[*cfg.variables.keys(), "run"]
)
itemdata = [
    {
        "tmppath": "tmpdata/test",
        "func": "func.p",
        "args": json.dumps(json.dumps(element))[1:-1],
        "seed": str(seed),
    }
    for element in index
]

# %%

schedd = htcondor.Schedd()
submit_result = schedd.submit(python_job, itemdata=iter(itemdata))

print(submit_result.cluster())

# %%
schedd.query(
    constraint=f"ClusterId == {submit_result.cluster()}",
    projection=["ClusterId", "ProcId", "Out", "Args"],
)

# %%
