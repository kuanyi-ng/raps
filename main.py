""" Shortest-job first (SJF) job schedule simulator """

import argparse
import json
import numpy as np
import random
import pandas as pd
import os
import re
import sys
import time

# Check for the required Python version
required_major, required_minor = 3, 9

if sys.version_info < (required_major, required_minor):
    sys.stderr.write(f"Error: RAPS requires Python {required_major}.{required_minor} or greater\n")
    sys.exit(1)

parser = argparse.ArgumentParser(description='Resource Allocator & Power Simulator (RAPS)')
parser.add_argument('--disable_cooling', action='store_true', help='Disable cooling model')
parser.add_argument('-d', '--debug', action='store_true', help='Enable debug mode and disable rich layout')
parser.add_argument('-e', '--encrypt', action='store_true', help='Encrypt any sensitive data in telemetry')
parser.add_argument('-n', '--numjobs', type=int, default=1000, help='Number of jobs to schedule')
parser.add_argument('-t', '--time', type=str, default=None, help='Length of time to simulate, e.g., 123, 123s, 27m, 3h, 7d')
parser.add_argument('-v', '--verbose', action='store_true', help='Enable verbose output')
parser.add_argument('-s', '--seed', action='store_true', help='Set random number seed for deterministic simulation')
parser.add_argument('-f', '--replay', nargs='+', type=str, help='Either: path/to/joblive path/to/jobprofile' + \
                                                                ' -or- filename.npz (overrides --workload option)')
parser.add_argument('--reschedule', action='store_true', help='Reschedule the telemetry workload')
parser.add_argument('-u', '--uncertainties', action='store_true',
                    help='Change from floating point units to floating point units with uncertainties.' + \
                                                                ' Very expensive w.r.t simulation time!')
parser.add_argument('--jid', type=str, default='*', help='Replay job id')
parser.add_argument('--validate', action='store_true', help='Use node power instead of CPU/GPU utilizations')
parser.add_argument('-o', '--output', action='store_true', help='Output power, cooling, and loss models for later analysis')
parser.add_argument('-p', '--plot', nargs='+', choices=['power', 'loss', 'pue', 'temp'],
                    help='Specify one or more types of plots to generate: power, loss, pue, temp')
parser.add_argument('--system', type=str, default='frontier', help='System config to use')
choices = ['random', 'benchmark', 'peak', 'idle']
parser.add_argument('-w', '--workload', type=str, choices=choices, default=choices[0], help='Type of synthetic workload')
choices = ['layout1', 'layout2']
parser.add_argument('--layout', type=str, choices=choices, default=choices[0], help='Layout of UI')
args = parser.parse_args()
args_dict = vars(args)
print(args_dict)

from raps.config import initialize_config, load_config_variables
initialize_config(args.system)

from raps.constants import OUTPUT_PATH
from raps.cooling import ThermoFluidsModel
from raps.ui import LayoutManager
from raps.flops import FLOPSManager
from raps.plotting import Plotter
from raps.power import PowerManager, compute_node_power, compute_node_power_validate
from raps.power import compute_node_power_uncertainties, compute_node_power_validate_uncertainties
from raps.scheduler import Scheduler, Job
from raps.telemetry import Telemetry
from raps.workload import Workload
from raps.utils import create_casename, convert_to_seconds

load_config_variables([
    'SC_SHAPE',
    'TOTAL_NODES',
    'DOWN_NODES',
    'SEED',
    'FMU_PATH',
    'MAX_TIME'
], globals())

if args.seed:
    random.seed(SEED)
    np.random.seed(SEED)

if not args.disable_cooling:
    try:
        cooling_model = ThermoFluidsModel(FMU_PATH)
        cooling_model.initialize()
        args.layout = "layout2"
    except:
        cooling_model = None
else:
    cooling_model = None

if args.validate:
    if args.uncertainties:
        power_manager = PowerManager(SC_SHAPE, DOWN_NODES, power_func=compute_node_power_validate_uncertainties)
    else:
        power_manager = PowerManager(SC_SHAPE, DOWN_NODES, power_func=compute_node_power_validate)
else:
    if args.uncertainties:
        power_manager = PowerManager(SC_SHAPE, DOWN_NODES, power_func=compute_node_power_uncertainties)
    else:
        power_manager = PowerManager(SC_SHAPE, DOWN_NODES, power_func=compute_node_power)

flops_manager = FLOPSManager(SC_SHAPE)
layout_manager = LayoutManager(args.layout, args.debug)
sc = Scheduler(TOTAL_NODES, DOWN_NODES, power_manager, flops_manager, layout_manager,
               cooling_model, **args_dict)
if args.replay:
    print(args.replay)
    td = Telemetry(**args_dict)

    # Try to extract date from given name to use as case directory
    matched_date = re.search(r"\d{4}-\d{2}-\d{2}", args.replay[0])
    if matched_date:
        extracted_date = matched_date.group(0)
        DIR_NAME = "sim=" + extracted_date
    else:
        extracted_date = "Date not found"
        DIR_NAME = create_casename()

    # Read either npz file or telemetry parquet files
    if args.replay[0].endswith(".npz"):
        jobs = td.load_snapshot(args.replay[0])
    else:
        print(args.replay)
        print(*args.replay)
        #jobs = td.load_data(args.replay[0], args.replay[1])
        jobs = td.load_data(args.replay)
        td.save_snapshot(jobs, filename=DIR_NAME)

    # Set number of timesteps based on the last job running which we assume
    # is the maximum value of submit_time + wall_time of all the jobs
    if args.time:
        timesteps = convert_to_seconds(args.time)
    else:
        timesteps = int(max(job[4] + job[7] for job in jobs)) + 1

    print(f'Running simulation for {timesteps} seconds')
    time.sleep(1)

else:
    wl = Workload(sc)
    jobs = getattr(wl, args.workload)(num_jobs=args.numjobs)

    if args.verbose:
        for job_vector in jobs:
            job = Job(job_vector, 0)
            print('jobid:', job.id, '\tlen(gpu_trace):', len(job.gpu_trace), '\twall_time(s):', job.wall_time)
        time.sleep(2)

    if args.time:
        timesteps = convert_to_seconds(args.time)
    else:
        timesteps = MAX_TIME

    DIR_NAME = create_casename()

OPATH = OUTPUT_PATH / DIR_NAME
print("Output directory is: ", OPATH)
sc.opath = OPATH

if args.plot or args.output:
    try:
        os.makedirs(OPATH)
    except OSError as error:
        print(f"Error creating directory: {error}")

if args.verbose:
    print(jobs)

sc.run_simulation_blocking(jobs, timesteps=timesteps)
output_stats = sc.get_stats()
print(json.dumps(output_stats, indent=4))

if args.plot:
    if 'power' in args.plot:
        pl = Plotter('Time (s)', 'Power (kW)', 'Power History', OPATH / 'power.svg', uncertainties=args.uncertainties)
        x, y = zip(*power_manager.history)
        pl.plot_history(x, y)

    if 'loss' in args.plot:
        pl = Plotter('Time (s)', 'Power Losses (kW)', 'Power Loss History', OPATH / 'loss.svg', uncertainties=args.uncertainties)
        x, y = zip(*power_manager.loss_history)
        pl.plot_history(x, y)

        pl = Plotter('Time (s)', 'Power Losses (%)', 'Power Loss History', OPATH / 'loss_pct.svg', uncertainties=args.uncertainties)
        x, y = zip(*power_manager.loss_history_percentage)
        pl.plot_history(x, y)

    if 'pue' in args.plot:
        if cooling_model:
            ylabel = 'PUE_Out[1]'
            title = 'FMU ' + ylabel + 'History'
            pl = Plotter('Time (s)', ylabel, title, OPATH / 'pue.svg', uncertainties=args.uncertainties)
            df = pd.DataFrame(cooling_model.fmu_history)
            df.to_parquet('cooling_model.parquet', engine='pyarrow')
            pl.plot_history(df['time'], df[ylabel])
        else:
            print('Cooling model not enabled... skipping output of plot')

    if 'temp' in args.plot:
        if cooling_model:
            ylabel = 'Tr_pri_Out[1]'
            title = 'FMU ' + ylabel + 'History'
            pl = Plotter('Time (s)', ylabel, title, OPATH / 'temp.svg')
            df = pd.DataFrame(cooling_model.fmu_history)
            df.to_parquet('cooling_model.parquet', engine='pyarrow')
            pl.plot_compare(df['time'], df[ylabel])
        else:
            print('Cooling model not enabled... skipping output of plot')

if args.output:

    with open(OPATH / 'stats.out', 'w') as f:
        json.dump(output_stats, f, indent=4)

    if args.uncertainties:
        print('Data dump not implemented using uncertainties!')  # Parquet cannot handle annotated ufloat format AFAIK
        pass
    else:

        if cooling_model:
            df = pd.DataFrame(cooling_model.fmu_history)
            df.to_parquet(OPATH / 'cooling_model.parquet', engine='pyarrow')

        df = pd.DataFrame(power_manager.history)
        df.to_parquet(OPATH / 'power_history.parquet', engine='pyarrow')

        df = pd.DataFrame(power_manager.loss_history)
        df.to_parquet(OPATH / 'loss_history.parquet', engine='pyarrow')
