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

    def __init__(
        self,
        func: Callable,
        tasks: Iterable[dict],
        return_value_names: Iterable[str],
        seed: int,
        datadir: str = ".",
    ) -> None:
        self.func = func
        self.tasks = tasks
        self.variable_names = list(tasks[0].keys())
        self.return_value_names = return_value_names
        self.seed = seed
        self.datadir = datadir
        pass

    def _workerProc(self, data):
        file_name = f"{self.datadir}/results_{mp.current_process().name}.csv"
        with open(file_name, mode="a+") as result_file:
            result_writer = csv.DictWriter(
                result_file,
                fieldnames=[
                    "run_nr",
                    *self.variable_names,
                    "series",
                    *self.return_value_names,
                ],
                delimiter=",",
                quotechar='"',
                quoting=csv.QUOTE_MINIMAL,
            )
            if os.stat(file_name).st_size == 0:
                result_writer.writeheader()

            run_nr = data["run_nr"]
            rng = np.random.default_rng(np.random.PCG64DXSM(self.seed).jumped(run_nr))
            del data["run_nr"]
            # result = self.func(rng=rng, **data)
            result_writer.writerows(
                {"run_nr": run_nr, **data, "series": series, **values}
                for series, values in enumerate(self.func(rng=rng, **data))
            )

    def run(
        self,
        runs: int = 50,
        num_processes: int = 4,
        seed: int = None,
    ):
        self.seed = self.seed if seed is None else seed
        run_tasks = []
        for task in self.tasks:
            run_tasks += [{"run_nr": run, **task} for run in range(runs)]
        print(
            f"Running {runs} realizations of {len(self.tasks)} tasks each (= {len(run_tasks)}) in {num_processes} processes."
        )
        with mp.Pool(processes=num_processes) as pool:
            with tqdm(
                total=len(run_tasks),
                position=0,
                leave=True,
                unit="runs",
            ) as pbar:
                for _ in pool.imap(self._workerProc, run_tasks):
                    pbar.update()
