import sys
import pickle
import json

proc_id = int(sys.argv[1])
func = sys.argv[2]
tmppath = sys.argv[3]
args = sys.argv[4]
seed = int(sys.argv[5])
# sim_func: function = pickle.load(open(func, "rb"))
print(f"{proc_id}, {tmppath}, {args}, {seed}")
# runresult = sim_func(*input["args"])
# pickle.dump(runresult, open(input["tmppath"] + f"/{proc_id:010d}.p", "wb"))
