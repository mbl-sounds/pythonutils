import sys
import pickle
import json

proc_id = int(sys.argv[1])
tmppath = sys.argv[2]
func = sys.argv[3]
args = json.loads(sys.argv[4])
seed = int(sys.argv[5])
# print(f"{proc_id}, {tmppath}, {args}, {seed}")
sim_func = pickle.load(open(func, "rb"))
runresult = sim_func(*args)
pickle.dump(runresult, open(tmppath + f"/{proc_id:010d}.p", "wb"))
