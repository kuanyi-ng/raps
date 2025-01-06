from raps.helpers import check_python_version
check_python_version()

import random
import sys

from args import args
from raps.config import ConfigManager
from raps.policy import PolicyType
from raps.ui import LayoutManager
from raps.scheduler import Scheduler
from raps.flops import FLOPSManager
from raps.power import PowerManager, compute_node_power
from raps.workload import Workload
from raps.utils import convert_to_seconds

# Load configurations for each partition
partition_names = args.partitions
configs = [ConfigManager(system_name=partition).get_config() for partition in partition_names]
args_dicts = [{**vars(args), 'config': config} for config in configs]

# Initialize Workload with all configurations
if args.replay:

    td = Telemetry(**args_dict)

else:
    wl = Workload(*configs)

# Generate jobs based on workload type
jobs = getattr(wl, args.workload)(num_jobs=args.numjobs)

# Group jobs by partition
jobs_by_partition = {partition: [] for partition in wl.partitions}
for job in jobs:
    jobs_by_partition[job['partition']].append(job)

# Initialize layout managers for each partition
layout_managers = {}
for i, config in enumerate(configs):
    pm = PowerManager(compute_node_power, **configs[i])
    fm = FLOPSManager(**args_dicts[i])
    scheduler = Scheduler(power_manager=pm, flops_manager=fm, cooling_model=None, **args_dicts[i])
    layout_managers[config['system_name']] = LayoutManager(args.layout, scheduler=scheduler, debug=args.debug, **config)

# Set simulation timesteps
if args.time:
    timesteps = convert_to_seconds(args.time)
else:
    timesteps = 88200  # Default to 24 hours

# Create generators for each layout manager
generators = {name: lm.run_stepwise(jobs_by_partition[name], timesteps=timesteps)
              for name, lm in layout_managers.items()}

# Step through all generators in lockstep
for timestep in range(timesteps):
    for name, gen in generators.items():
        next(gen)  # Advance each generator

    # Print debug info every UI_UPDATE_FREQ
    if timestep % configs[0]['UI_UPDATE_FREQ'] == 0:  # Assuming same frequency for all partitions
        for name, lm in layout_managers.items():
            sys_util = lm.scheduler.sys_util_history[-1] if lm.scheduler.sys_util_history else 0.0
            print(f"[DEBUG] {name} - Timestep {timestep} - Jobs in queue: {len(lm.scheduler.queue)} - Utilization: {sys_util[1]:.2f}%")

print("Simulation complete.")

