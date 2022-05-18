from dataclasses import dataclass
import json
import numpy as np
import pandas as pd


class SimConfig:
    id: str
    runs: int
    seed: int
    variables: dict

    def __init__(self, id, runs, seed, variables) -> None:
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
    result: SimResult

    def __init__(self, cfg: SimConfig) -> None:
        self.cfg = cfg
        self.result = SimResult(cfg)
        pass

    def runLocal(self) -> SimResult:
        print("Run locally")
        return self.result

    def runCondor(self) -> SimResult:
        try:
            import htcondor
        except:
            print("You're not on a condor submit node, dummy!")
            return None

        print("Run on cluster")

        return self.result
