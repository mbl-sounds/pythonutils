import os
import json
import pickle
import shutil
import pandas as pd
from typing import Callable
import multiprocessing as mp
from dataclasses import dataclass


class SimConfig:
    id: str
    runs: int
    seed: int
    variables: list
    variable_values: list
    cluster: int

    def __init__(self, id: str, runs: int, seed: int, variables: list) -> None:
        self.id = id
        self.runs = runs
        self.seed = seed

        self.variable_values = variables

        variable_set = {}
        for vars in variables:
            assert type(vars) == dict, f"variables must be dictionary"
            for key, vals in vars.items():
                assert (
                    type(vals) == list
                ), f"variable '{key}' is not a list. Make list (e.g., [{vals}])"
            variable_set = variable_set | vars

        self.variables = list(variable_set.keys())

        pass

    @classmethod
    def load(cls, filename: str):
        with open(filename + ".json", "r") as f:
            data = json.load(fp=f)
            return cls(
                id=data["id"],
                runs=data["runs"],
                seed=data["seed"],
                # variables=data["variables"],
                variables=data["variable_values"],
            )
        pass

    def save(self, filename: str):
        with open(filename + ".json", "w") as f:
            json.dump(self, fp=f, default=lambda o: o.__dict__, indent=4)
        pass


@dataclass
class SimResult:
    cfg: SimConfig
    df: pd.DataFrame
    done: bool

    @classmethod
    def load(cls, filename: str):
        pass

    def save(self, filename: str):
        pass


class Simulation:
    cfg: SimConfig
    simpath: str
    simfunc: str
    simfunc_local: Callable
    tmppath: str
    index: pd.MultiIndex
    schedd: None
    cluster: int

    def __init__(self, cfg: SimConfig, simpath: str, simfunc: Callable) -> None:
        assert set(
            simfunc.__code__.co_varnames[: simfunc.__code__.co_argcount - 2]
        ) == set(
            cfg.variables
        ), f"Set of function arguments simulate{simfunc.__code__.co_varnames[:simfunc.__code__.co_argcount-2]} does not match variables in SimConfig {cfg.variables}!"

        self.cfg = cfg
        self.simpath = simpath
        self.simfunc_local = simfunc
        self.simfunc = simfunc.__name__

        self.tmppath = f"tmpdata/{self.cfg.id}"
        self.tmppath_data = f"tmpdata/{self.cfg.id}/data"
        self.tmppath_out = f"tmpdata/{self.cfg.id}/out"
        os.makedirs(self.tmppath_data, exist_ok=True)
        os.makedirs(self.tmppath_out, exist_ok=True)

        names = [*self.cfg.variables, "run"]
        self.index = pd.MultiIndex(
            levels=[[] for x in names], codes=[[] for x in names], names=names
        )
        curr_var_set = {}
        tuples = []
        for var_set in cfg.variable_values:
            curr_var_set = curr_var_set | var_set
            index = pd.MultiIndex.from_product(
                [*curr_var_set.values(), [*range(self.cfg.runs)]],
            )
            tuples = tuples + index.values.tolist()

        self.index = pd.MultiIndex.from_tuples(
            tuples,
            names=names,
        )
        print(f"Generated {len(self.index)} sets.")
        try:
            mp.set_start_method("spawn")
        except RuntimeError:
            pass
        pass

    def _workerProc(self, q: mp.Queue, qp: mp.Queue):
        while True:
            data = q.get()
            if data == "DONE":
                qp.put("DONE")
                break
            task_id = data["task_id"]
            runresult = {
                "data": self.simfunc_local(*data["args"], data["seed"]),
                "args": data["args"],
            }
            pickle.dump(runresult, open(self.tmppath_data + f"/{task_id}.p", "wb"))
            qp.put(1)

    def runLocal(self, nprocesses=1, showprogress=False):
        # self.clearTmpData()
        os.makedirs(self.tmppath_data, exist_ok=True)
        os.makedirs(self.tmppath_out, exist_ok=True)

        q = mp.Queue()
        qp = mp.Queue()
        if showprogress:
            import ipywidgets as widgets
            from IPython.display import display
            import time

        # task_id = 0
        submitted_tasks = 0
        for element in self.index:
            task_id = "-".join([str(el) for el in element])
            if not os.path.isfile(self.tmppath_data + f"/{task_id}.p"):
                assert (
                    len(element) == len(self.cfg.variables) + 1
                ), f"number of generated arguments not matching!"
                arg = {"task_id": task_id, "args": element, "seed": self.cfg.seed}
                q.put(arg)
                submitted_tasks += 1
            # task_id += 1
        print(
            f"Run {submitted_tasks} (of {len(self.index)}) tasks locally in {nprocesses} processes"
        )

        processes = list()
        for i in range(nprocesses):
            q.put("DONE")  # tells proesses to quit
            p = mp.Process(
                target=self._workerProc,
                args=(q, qp),
            )
            p.daemon = True
            p.start()  # Launch p() as another proc
            processes.append(p)

        if showprogress:
            layout = widgets.Layout(width="auto", height="30px")  # set width and height
            f = widgets.IntProgress(
                min=0,
                max=submitted_tasks,
                layout=layout,
            )  # instantiate the bar
            display(f)  # display the bar
            all_done = 0
            while True:
                time.sleep(0.5)
                data = qp.get()
                if data == "DONE":
                    all_done += 1
                else:
                    f.value += data
                if all_done >= nprocesses:
                    break
            f.value = submitted_tasks

        for p in processes:
            p.join()

        return

    def runCondor(self, user_submit_data: dict, n_jobs=10):
        try:
            import htcondor
        except:
            print(
                "You're not on a condor submit node or `htcondor` module is not available, dummy!"
            )
            return None
        os.makedirs(self.tmppath_data, exist_ok=True)
        os.makedirs(self.tmppath_out, exist_ok=True)

        self.schedd = htcondor.Schedd()

        execute_path = os.path.dirname(os.path.realpath(__file__)) + "/sim_execute.py"

        submit_data = {
            "universe": "Vanilla",
            "request_walltime": "100",
            "nice_user": "true",
            "initialdir": ".",
            "notification": "Error",
            "executable": "/users/sista/mblochbe/python_venvs/admmstuff/bin/python",
            "arguments": f"{execute_path} $(ProcId) $(job_id) $(tmppath_data) $(func_data)",
            "output": f"{self.tmppath_out}/{self.cfg.id}.out",
            "error": f"{self.tmppath_out}/{self.cfg.id}.err",
            "log": f"{self.tmppath_out}/{self.cfg.id}.log",
            "request_cpus": "1",
            "request_memory": "128MB",
        }
        submit_data.update(user_submit_data)

        submit = htcondor.Submit(submit_data)
        print(submit)
        submitted_tasks = 0
        tasks_per_job = [[] for i in range(n_jobs)]
        job_id = 0
        n_subitted_jobs = 0
        for element in self.index:
            task_id = "-".join([str(el) for el in element])
            if not os.path.isfile(self.tmppath_data + f"/{task_id}.p"):
                assert (
                    len(element) == len(self.cfg.variables) + 1
                ), f"number of generated arguments not matching!"
                tasks_per_job[job_id].append({"task_id": task_id, "args": element})
                n_subitted_jobs += 1
                job_id = job_id + 1 if job_id < n_jobs - 1 else 0

        itemdata = []
        for elements in tasks_per_job:
            itemdata.append(
                {
                    "tmppath_data": self.tmppath_data,
                    "job_id": str(submitted_tasks),
                    "func_data": "{"
                    + json.dumps(
                        json.dumps(
                            {
                                "simpath": self.simpath,
                                "simfunc": self.simfunc,
                                "args": elements,
                                "seed": self.cfg.seed,
                            },
                            indent=None,
                            separators=(",", ":"),
                        )
                    )[2:-2]
                    + "}",
                }
            )
            submitted_tasks += 1

        print(
            f"Run {submitted_tasks} condor process(es) with {n_subitted_jobs} (of {len(self.index)}) jobs."
        )
        # print(itemdata)

        submit_result = self.schedd.submit(submit, itemdata=iter(itemdata))
        self.cluster = submit_result.cluster()
        self.cfg.cluster = submit_result.cluster()
        print(
            f"Submitted {len(itemdata)} process(es) in cluster {submit_result.cluster()}."
        )
        return self.cluster

    def isDone(self, cluster=None) -> bool:
        try:
            import htcondor
        except:
            print(
                "You're not on a condor submit node or `htcondor` module is not available, dummy!"
            )
            return None
        self.schedd = htcondor.Schedd()
        assert type(self.schedd) != None, "No condor jobs run yet :("
        cluster = self.cluster if cluster is None else cluster
        data = self.schedd.query(
            constraint=f"ClusterId == {cluster}",
            projection=["ClusterId", "ProcId"],
        )
        return len(data) == 0

    def getJobStatus(self, cluster=None, proj=["ClusterId", "ProcId", "JobStatus"]):
        try:
            import htcondor
        except:
            print(
                "You're not on a condor submit node or `htcondor` module is not available, dummy!"
            )
            return None
        self.schedd = htcondor.Schedd()
        cluster = self.cluster if cluster is None else cluster
        return self.schedd.query(
            constraint=f"ClusterId == {cluster}",
            projection=proj,
        )

    def getResult(self) -> SimResult:
        assert os.path.isdir(
            self.tmppath_data
        ), f"No data path {self.tmppath_data}. Re-initialize or rerun simulation."
        files = sorted(
            file
            for file in os.listdir(self.tmppath_data)
            if os.path.isfile(os.path.join(self.tmppath_data, file))
        )
        print(f"Found {len(files)} data files.")
        # assert len(files) == len(
        #     self.index
        # ), f"Requires {len(self.index)} data files! Check if simulation finished sucessfully!"
        dl = {}
        for filename in files:
            with open(self.tmppath_data + "/" + filename, "rb") as f:
                data = pickle.load(f)
                dl[tuple(data["args"])] = data["data"]
        df = pd.DataFrame(dl).T
        df.index.set_names(self.index.names, inplace=True)
        result = SimResult(self.cfg, df, True)
        return result

    def clearTmpData(self):
        if os.path.exists(self.tmppath) and os.path.isdir(self.tmppath):
            shutil.rmtree(self.tmppath)
        return
