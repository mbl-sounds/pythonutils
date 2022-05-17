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
