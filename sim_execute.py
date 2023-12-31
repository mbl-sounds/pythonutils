# %%
import os
import sys
import json
import pickle
import importlib.util


#%%
proc_id: int = int(sys.argv[1])
job_id: str = sys.argv[2]
tmppath: str = sys.argv[3]
func_data = json.loads(sys.argv[4])

sys.path.append(os.path.dirname(func_data["simpath"]))
spec = importlib.util.spec_from_file_location("funcmod", func_data["simpath"])
funcmod = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = funcmod
spec.loader.exec_module(funcmod)
simfunc = getattr(funcmod, func_data["simfunc"])

for args in func_data["args"]:
    task_id = args["task_id"]
    runresult = {
        "data": simfunc(*args["args"], func_data["seed"]),
        "args": args["args"],
    }
    pickle.dump(runresult, open(tmppath + f"/{task_id}.p", "wb"))
