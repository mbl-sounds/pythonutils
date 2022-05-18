# %%
import numpy as np
import htcondor
import classad

# %%
hostname_job = htcondor.Submit(
    {
        "universe": "Vanilla",
        "request_walltime": "100",
        "nice_user": "true",
        "initialdir": ".",
        "notification": "Error",
        "executable": "/bin/hostname",  # the program to run on the execute node
        "output": "hostname.out",  # anything the job prints to standard output will end up in this file
        "error": "hostname.err",  # anything the job prints to standard error will end up in this file
        "log": "hostname.log",  # this file will contain a record of what happened to the job
        "request_cpus": "1",  # how many CPU cores we want
        "request_memory": "128MB",  # how much memory we want
    }
)

print(hostname_job)
# %%
schedd = htcondor.Schedd()  # get the Python representation of the scheduler
submit_result = schedd.submit(hostname_job)  # submit the job
print(submit_result.cluster())

# %%
sleep_job = htcondor.Submit(
    {
        "universe": "Vanilla",
        "request_walltime": "100",
        "nice_user": "true",
        "initialdir": ".",
        "notification": "Error",
        "executable": "/bin/sleep",
        "arguments": "10s",  # sleep for 10 seconds
        "output": "sleep-$(ProcId).out",  # output and error for each job, using the $(ProcId) macro
        "error": "sleep-$(ProcId).err",
        "log": "sleep.log",  # we still send all of the HTCondor logs for every job to the same file (not split up!)
        "request_cpus": "1",
        "request_memory": "128MB",
        "request_disk": "128MB",
    }
)

print(sleep_job)

# %%
schedd = htcondor.Schedd()
submit_result = schedd.submit(sleep_job, count=10)  # submit 10 jobs

print(submit_result.cluster())

# %%
schedd.query(
    constraint=f"ClusterId == {submit_result.cluster()}",
    projection=["ClusterId", "ProcId", "Out"],
)

# %%
import simulation

# %%
cfg = simulation.SimConfig("test", 10, 1234)
test = simulation.Simulation(cfg)

# %%
test.runLocal()

# %%
test.runCondor()

# %%
# TEST SCENARIO
import os
import numpy as np
import pandas as pd
import simulation
import pickle
import matplotlib.pyplot as plt

# %%
cfg = simulation.SimConfig(
    id="test",
    runs=100,
    seed=1234,
    variables={"algorithm": ["LMS", "ADMM"], "variance": [2, 30], "length": [100]},
)

# %%
# Simulation function for testing
def simulationFunction(a, b, c, run, rng: np.random.RandomState):
    print(f"run {run} lol")
    data = rng.normal(loc=b, size=(c,))
    return data


# %%
index = pd.MultiIndex.from_product(
    [*cfg.variables.values(), [*range(cfg.runs)]], names=[*cfg.variables.keys(), "run"]
)
# %%
rng = np.random.RandomState()
rng.seed(1234)

if os.path.isfile(f"{cfg.id}.p"):
    dl = pickle.load(open(f"{cfg.id}.p", "rb"))
else:
    dl = {}

for element in index:
    id = "-".join(map(str, element))
    if id in dl:
        continue

    assert (
        len(element) == len(cfg.variables) + 1
    ), f"number of generated arguments not matching!"
    dl[id] = simulationFunction(*element, rng)

pickle.dump(dl, open(f"{cfg.id}.p", "wb"))

# %%
rng = np.random.RandomState()
rng.seed(1234)

tmp_data_path = f"tmpdata/{cfg.id}"

if not os.path.isdir(tmp_data_path):
    os.mkdir(tmp_data_path)
proc_id = 0
for element in index:
    id = "-".join(map(str, element))
    if not os.path.isfile(tmp_data_path + f"/{proc_id:010d}.p"):
        assert (
            len(element) == len(cfg.variables) + 1
        ), f"number of generated arguments not matching!"
        runresult = simulationFunction(*element, rng)
        pickle.dump(runresult, open(tmp_data_path + f"/{proc_id:010d}.p", "wb"))
    proc_id += 1

# %%
files = sorted(
    file
    for file in os.listdir(tmp_data_path)
    if os.path.isfile(os.path.join(tmp_data_path, file))
)
dl = []
for filename in files:
    print(filename)
    with open(tmp_data_path + "/" + filename, "rb") as f:
        dl.append(pickle.load(f))
df = pd.DataFrame(dl, index=index)
#%%
df.groupby(["algorithm", "variance", "length"]).mean()
# %%
plt.plot(df.groupby(["algorithm", "variance", "length"]).mean().to_numpy().T)
plt.show()
#%%
df.groupby(["algorithm", "variance", "length"]).var()

# %%
result = simulation.SimResult(cfg)
