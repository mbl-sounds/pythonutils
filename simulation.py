from dataclasses import dataclass
import json
import numpy as np


@dataclass
class SimConfig:
    id: str
    runs: int
    seed: int
    vars

    @classmethod
    def load(cls, filename: str):
        pass

    def save(self, filename: str):
        pass


@dataclass
class SimResult:
    cfg: SimConfig

    @classmethod
    def load(cls, filename: str):
        pass

    def save(self, filename: str):
        pass


class Simulation:
    name: str
    cfg: SimConfig
    result: SimResult
    variables: dict

    def __init__(self, name: str, cfg: SimConfig) -> None:
        self.name = name
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
