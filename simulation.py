import os
import json
import pickle
import shutil
from typing import Callable
import numpy as np
import pandas as pd
from enum import Enum
from dataclasses import dataclass


class ResultType(Enum):
    LOCAL = 1
    CONDOR = 2


class SimConfig(dict):
    id: str
    runs: int
    seed: int
    variables: dict

    def __init__(self, id: str, runs: int, seed: int, variables: dict) -> None:
        dict.__init__(self)
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
        pass

    def save(self, filename: str):
        pass


@dataclass
class SimResult:
    # type: ResultType
    cfg: SimConfig
    df: pd.DataFrame
    done: bool

    def createDF(self) -> None:
        index = pd.MultiIndex.from_product(
            [*self.cfg.variables.values(), [*range(self.cfg.runs)]],
            names=[*self.cfg.variables.keys(), "run"],
        )

    @classmethod
    def load(cls, filename: str):
        pass

    def save(self, filename: str):
        pass


class Simulation:
    cfg: SimConfig
    # result: SimResult
    func: Callable
    tmppath: str
    index: pd.MultiIndex

    def __init__(self, cfg: SimConfig, func: Callable) -> None:
        self.cfg = cfg
        self.func = func
        self.tmppath = f"tmpdata/{self.cfg.id}"
        if not os.path.isdir(self.tmppath):
            os.mkdir(self.tmppath)
        self.index = pd.MultiIndex.from_product(
            [*self.cfg.variables.values(), [*range(self.cfg.runs)]],
            names=[*self.cfg.variables.keys(), "run"],
        )
        # self.result = SimResult(ResultType.LOCAL, cfg, None, False)
        pass

    def runLocal(self):
        print("Run locally")
        if not os.path.isdir(self.tmppath):
            os.mkdir(self.tmppath)
        proc_id = 0
        for element in self.index:
            # id = "-".join(map(str, element))
            if not os.path.isfile(self.tmppath + f"/{proc_id:010d}.p"):
                assert (
                    len(element) == len(self.cfg.variables) + 1
                ), f"number of generated arguments not matching!"
                runresult = self.func(*element, self.cfg.seed)
                pickle.dump(runresult, open(self.tmppath + f"/{proc_id:010d}.p", "wb"))
            proc_id += 1
        return

    def runCondor(self):
        try:
            import htcondor
        except:
            print(
                "You're not on a condor submit node or `htcondor` module is not available, dummy!"
            )
            return None

        print("Run on cluster")
        if not os.path.isdir(self.tmppath):
            os.mkdir(self.tmppath)

        return

    def _dispatchFunc(self):
        pass

    def getResult(self) -> SimResult:
        assert os.path.isdir(
            self.tmppath
        ), f"No data path {self.tmppath}. Re-initialize or rerun simulation."
        files = sorted(
            file
            for file in os.listdir(self.tmppath)
            if os.path.isfile(os.path.join(self.tmppath, file))
        )
        print(f"Found {len(files)} data files. Creating dataframe.")
        dl = []
        for filename in files:
            with open(self.tmppath + "/" + filename, "rb") as f:
                dl.append(pickle.load(f))
        df = pd.DataFrame(dl, index=self.index)
        result = SimResult(self.cfg, df, True)
        return result

    def clearTmpData(self):
        shutil.rmtree(self.tmppath)
        return
