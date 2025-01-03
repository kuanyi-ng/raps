from raps.helpers import check_python_version
check_python_version()

from args import args
import copy
args_dict1 = copy.deepcopy(vars(args))
args_dict2 = copy.deepcopy(vars(args))
print(args_dict1)
print(args_dict2)

from raps.config import ConfigManager
from raps.ui import LayoutManager
from raps.flops import FLOPSManager
from raps.power import PowerManager, compute_node_power
from raps.scheduler import Scheduler
from raps.workload import Workload
from raps.utils import convert_to_seconds

config1 = ConfigManager(system_name='setonix-cpu').get_config()
print(config1['system_name'])
config2 = ConfigManager(system_name='setonix-gpu').get_config()
print(config2['system_name'])

pm1 = PowerManager(compute_node_power, **config1)
pm2 = PowerManager(compute_node_power, **config2)

args_dict1['config'] = config1
args_dict2['config'] = config2

fm1 = FLOPSManager(**args_dict1)
fm2 = FLOPSManager(**args_dict2)

sc1 = Scheduler(power_manager=pm1, flops_manager=fm1, cooling_model=None, **args_dict1)
sc2 = Scheduler(power_manager=pm1, flops_manager=fm2, cooling_model=None, **args_dict2)

layout_manager1 = LayoutManager(args.layout, scheduler=sc1, debug=args.debug, **config1)
layout_manager2 = LayoutManager(args.layout, scheduler=sc2, debug=args.debug, **config2)

wl = Workload(**config1)
jobs = getattr(wl, args.workload)(num_jobs=args.numjobs)
#print(jobs)
#exit()

if args.time:
    timesteps = convert_to_seconds(args.time)
else:
    timesteps = 88200 # 24 hours

if args.verbose:
    print(jobs)

# Create generator objects for both partitions
gen1 = layout_manager1.run_nonblocking(jobs, timesteps=timesteps)
gen2 = layout_manager2.run_nonblocking(jobs, timesteps=timesteps)

# Step through both generators in lockstep
for _ in range(timesteps):
    next(gen1)  # Advance first scheduler
    next(gen2)  # Advance second scheduler
