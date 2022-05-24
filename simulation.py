import os
import json
import pickle
import shutil
import pandas as pd
from typing import Callable
import multiprocessing as mp
from dataclasses import dataclass
import ipywidgets as widgets
from IPython.display import display
import time


class SimConfig:
    id: str
    runs: int
    seed: int
    variables: dict

    def __init__(self, id: str, runs: int, seed: int, variables: dict) -> None:
        self.id = id
        self.runs = runs
        self.seed = seed
        assert type(variables) == dict, f"variables must be dictionary"
        for key, vals in variables.items():
            assert (
                type(vals) == list
            ), f"variable '{key}' is not a list. Make list (e.g., [{vals}])"
        self.variables = variables
        pass

    @classmethod
    def load(cls, filename: str):
        with open(filename + ".json", "r") as f:
            data = json.load(fp=f)
            return cls(
                id=data["id"],
                runs=data["runs"],
                seed=data["seed"],
                variables=data["variables"],
            )
        pass

    def save(self, filename: str):
        with open(filename + ".json", "w") as f:
            json.dump(
                self, fp=f, default=lambda o: o.__dict__, sort_keys=True, indent=4
            )
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
            cfg.variables.keys()
        ), f"Set of function arguments simulate{simfunc.__code__.co_varnames[:simfunc.__code__.co_argcount-2]} does not match variables in SimConfig {list(cfg.variables.keys())}!"

        self.cfg = cfg
        self.simpath = simpath
        self.simfunc_local = simfunc
        self.simfunc = simfunc.__name__

        self.tmppath = f"tmpdata/{self.cfg.id}"
        self.tmppath_data = f"tmpdata/{self.cfg.id}/data"
        self.tmppath_out = f"tmpdata/{self.cfg.id}/out"
        os.makedirs(self.tmppath_data, exist_ok=True)
        os.makedirs(self.tmppath_out, exist_ok=True)

        self.index = pd.MultiIndex.from_product(
            [*self.cfg.variables.values(), [*range(self.cfg.runs)]],
            names=[*self.cfg.variables.keys(), "run"],
        )
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
            runresult = self.simfunc_local(*data["args"], data["seed"])
            pickle.dump(runresult, open(self.tmppath_data + f"/{task_id:010d}.p", "wb"))
            qp.put(1)

    def runLocal(self, nprocesses=1):
        print(f"Run locally in {nprocesses} processes")
        os.makedirs(self.tmppath_data, exist_ok=True)
        os.makedirs(self.tmppath_out, exist_ok=True)

        self.tasks_done = 0
        lock = mp.Lock()

        q = mp.Queue()
        qp = mp.Queue()

        task_id = 0
        for element in self.index:
            if not os.path.isfile(self.tmppath_data + f"/{task_id:010d}.p"):
                assert (
                    len(element) == len(self.cfg.variables) + 1
                ), f"number of generated arguments not matching!"
                arg = {"task_id": task_id, "args": element, "seed": self.cfg.seed}
                q.put(arg)
            task_id += 1

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

        layout = widgets.Layout(width="auto", height="30px")  # set width and height
        f = widgets.IntProgress(
            min=0,
            max=len(self.index),
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

        for p in processes:
            p.join()
        f.value = len(self.index)

        return

    def runCondor(self, user_submit_data: dict):
        try:
            import htcondor
        except:
            print(
                "You're not on a condor submit node or `htcondor` module is not available, dummy!"
            )
            return None

        print("Run on HTcondor cluster")
        os.makedirs(self.tmppath_data, exist_ok=True)
        os.makedirs(self.tmppath_out, exist_ok=True)

        submit_data = {
            "universe": "Vanilla",
            "request_walltime": "100",
            "nice_user": "true",
            "initialdir": ".",
            "notification": "Error",
            "executable": "/users/sista/mblochbe/python_venvs/admmstuff/bin/python",
            "arguments": f"sim_execute.py $(ProcId) $(tmppath_data) $(func_data)",  # sleep for 10 seconds
            "output": f"{self.tmppath_out}/{self.cfg.id}.out",  # output and error for each job, using the $(ProcId) macro
            "error": f"{self.tmppath_out}/{self.cfg.id}.err",
            "log": f"{self.tmppath_out}/{self.cfg.id}.log",  # we still send all of the HTCondor logs for every job to the same file (not split up!)
            "request_cpus": "1",
            "request_memory": "128MB",
        }
        submit_data.update(user_submit_data)

        submit = htcondor.Submit(submit_data)
        print(submit)
        itemdata = [
            {
                "tmppath_data": self.tmppath_data,
                "func_data": "{"
                + json.dumps(
                    json.dumps(
                        {
                            "simpath": self.simpath,
                            "simfunc": self.simfunc,
                            "args": (*element, self.cfg.seed),
                        },
                        indent=None,
                        separators=(",", ":"),
                    )
                )[2:-2]
                + "}",
            }
            for element in self.index
        ]

        self.schedd = htcondor.Schedd()
        submit_result = self.schedd.submit(submit, itemdata=iter(itemdata))
        self.cluster = submit_result.cluster()
        print(f"Submitted {len(itemdata)} job(s) in cluster {submit_result.cluster()}.")
        return

    def isDone(self) -> bool:
        assert type(self.schedd) != None, "No condor jobs run yet :("
        data = self.schedd.query(
            constraint=f"ClusterId == {self.cluster}",
            projection=["ClusterId", "ProcId"],
        )
        return len(data) == 0

    def getJobStatus(self, proj=["ClusterId", "ProcId", "JobStatus"]):
        return self.schedd.query(
            constraint=f"ClusterId == {self.cluster}",
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
        assert len(files) == len(
            self.index
        ), f"Requires {len(self.index)} data files! Check if simulation finished sucessfully!"
        dl = []
        for filename in files:
            with open(self.tmppath_data + "/" + filename, "rb") as f:
                dl.append(pickle.load(f))
        df = pd.DataFrame(dl, index=self.index)
        result = SimResult(self.cfg, df, True)
        return result

    def clearTmpData(self):
        if os.path.exists(self.tmppath) and os.path.isdir(self.tmppath):
            shutil.rmtree(self.tmppath)
        return
