# %%
import os
import sys
import json
import pickle
import importlib.util


#%%
proc_id: int = int(sys.argv[1])
tmppath: str = sys.argv[2]
func_data = json.loads(sys.argv[3])

sys.path.append(os.path.dirname(func_data["simpath"]))
spec = importlib.util.spec_from_file_location("funcmod", func_data["simpath"])
funcmod = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = funcmod
spec.loader.exec_module(funcmod)
simfunc = getattr(funcmod, func_data["simfunc"])

runresult = simfunc(*func_data["args"])
pickle.dump(runresult, open(tmppath + f"/{proc_id:010d}.p", "wb"))
