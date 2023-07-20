import os
import csv
import numpy as np
from tqdm import tqdm
from typing import Callable, Iterable
import multiprocess as mp


class DockerSim:
    func: Callable
    tasks: Iterable[dict]
    variable_names: list[str]
    seed: int

    def __init__(self, func: Callable, tasks: Iterable[dict], seed: int) -> None:
        self.func = func
        self.tasks = tasks
        self.variable_names = tasks[0].keys()
        print(self.variable_names)
        self.seed = seed
        pass

    def _workerProc(self, data):
        file_name = f"data/results_{mp.current_process().name}.csv"
        result_file = open(file_name, mode="a+")
        result_writer = csv.writer(
            result_file,
            delimiter=",",
            quotechar='"',
            quoting=csv.QUOTE_MINIMAL,
        )
        if os.stat(file_name).st_size == 0:
            result_writer.writerow(["run_nr", "a", "b", "series", "value"])

        run_nr = data["run_nr"]
        rng = np.random.default_rng(np.random.PCG64DXSM(self.seed).jumped(run_nr))
        del data["run_nr"]
        result = self.func(rng=rng, **data)
        result_writer.writerows(
            [
                [run_nr, data["a"], data["b"], series, value]
                for series, value in enumerate(result)
            ]
        )
        result_file.close()

    def run(
        self,
        num_processes: int = 4,
        seed: int = None,
    ):
        self.seed = self.seed if seed is None else seed
        with mp.Pool(processes=num_processes) as pool:
            with tqdm(
                total=len(self.tasks),
                position=0,
                leave=True,
                unit="runs",
            ) as pbar:
                for _ in pool.imap(self._workerProc, self.tasks):
                    pbar.update()
