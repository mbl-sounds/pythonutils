from dataclasses import dataclass
import json
import numpy as np


@dataclass
class SimConfig:
    runs: int
    seed: int

    @classmethod
    def load(cls, filename: str):
        pass

    def save(self, filename: str):
        pass


@dataclass
class SimResult:
    runs: int
    seed: int

    @classmethod
    def load(cls, filename: str):
        pass

    def save(self, filename: str):
        pass


class Simulation:
    name: str
    cfg: SimConfig
    variables: dict

    def __init__(self, name: str, cfg: SimConfig) -> None:
        self.name = name
        self.cfg = cfg

        pass

    def runLocal(self) -> SimResult:
        return SimResult()
