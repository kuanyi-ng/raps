""" Shortest-job first (SJF) job schedule simulator """

import json
import numpy as np
import random
import pandas as pd
import os
import re
import time

from tqdm import tqdm

from raps.helpers import check_python_version
check_python_version()

from args import args
args_dict = vars(args)
print(args_dict)

from raps.config import ConfigManager
from raps.constants import OUTPUT_PATH, SEED
from raps.cooling import ThermoFluidsModel
from raps.ui import LayoutManager
from raps.flops import FLOPSManager
from raps.plotting import Plotter
from raps.power import PowerManager, compute_node_power, compute_node_power_validate
from raps.power import compute_node_power_uncertainties, compute_node_power_validate_uncertainties
from raps.scheduler import Scheduler, Job
from raps.telemetry import Telemetry
from raps.workload import Workload
from raps.weather import Weather
from raps.utils import create_casename, convert_to_seconds, write_dict_to_file, next_arrival

config = ConfigManager(system_name=args.system).get_config()

if args.seed:
    random.seed(SEED)
    np.random.seed(SEED)

if args.cooling:
    cooling_model = ThermoFluidsModel(**config)
    cooling_model.initialize()
    args.layout = "layout2"

    if args_dict['start']:
        cooling_model.weather = Weather(args_dict['start'], config = config)
else:
    cooling_model = None

if args.validate:
    if args.uncertainties:
        power_manager = PowerManager(compute_node_power_validate_uncertainties, **config)
    else:
        power_manager = PowerManager(compute_node_power_validate, **config)
else:
    if args.uncertainties:
        power_manager = PowerManager(compute_node_power_uncertainties, **config)
    else:
        power_manager = PowerManager(compute_node_power, **config)
args_dict['config'] = config
flops_manager = FLOPSManager(**args_dict)

sc = Scheduler(
    power_manager = power_manager, flops_manager = flops_manager,
    cooling_model = cooling_model,
    **args_dict,
)
layout_manager = LayoutManager(args.layout, scheduler = sc, debug = args.debug, **config)

if args.replay:

    if args.fastforward: args.fastforward = convert_to_seconds(args.fastforward)

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
        print(f"Loading {args.replay[0]}...")
        jobs = td.load_snapshot(args.replay[0])
        if args.reschedule:
            print("available nodes:", config['AVAILABLE_NODES'])
            for job in tqdm(jobs, desc="Updating requested_nodes"):
                job['requested_nodes'] = None
                job['submit_time'] = next_arrival(1 / config['JOB_ARRIVAL_TIME'])
    else:
        print(*args.replay)
        jobs = td.load_data(args.replay)
        td.save_snapshot(jobs, filename=DIR_NAME)

    # Set number of timesteps based on the last job running which we assume
    # is the maximum value of submit_time + wall_time of all the jobs
    if args.time:
        timesteps = convert_to_seconds(args.time)
    else:
        timesteps = int(max(job['wall_time'] + job['submit_time'] for job in jobs)) + 1

    print(f'Simulating {len(jobs)} jobs for {timesteps} seconds')
    time.sleep(1)

else:
    wl = Workload(config)
    jobs = getattr(wl, args.workload)(num_jobs=args.numjobs)

    if args.verbose:
        for job_vector in jobs:
            job = Job(job_vector, 0)
            print('jobid:', job.id, '\tlen(gpu_trace):', len(job.gpu_trace), '\twall_time(s):', job.wall_time)
        time.sleep(2)

    if args.time:
        timesteps = convert_to_seconds(args.time)
    else:
        timesteps = 88200 # 24 hours

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

layout_manager.run(jobs, timesteps=timesteps)

output_stats = sc.get_stats()
# Following b/c we get the following error when we use PM100 telemetry dataset
# TypeError: Object of type int64 is not JSON serializable
try:
    print(json.dumps(output_stats, indent=4))
except:
    print(output_stats)

if args.plot:
    if 'power' in args.plot:
        pl = Plotter('Time (s)', 'Power (kW)', 'Power History', \
                     OPATH / f'power.{args.imtype}', \
                     uncertainties=args.uncertainties)
        x, y = zip(*power_manager.history)
        pl.plot_history(x, y)

    if 'util' in args.plot:
        pl = Plotter('Time (s)', 'System Utilization (%)', \
                     'System Utilization History', OPATH / f'util.{args.imtype}')
        x, y = zip(*sc.sys_util_history)
        pl.plot_history(x, y)

    if 'loss' in args.plot:
        pl = Plotter('Time (s)', 'Power Losses (kW)', 'Power Loss History', \
                     OPATH / f'loss.{args.imtype}', \
                     uncertainties=args.uncertainties)
        x, y = zip(*power_manager.loss_history)
        pl.plot_history(x, y)

        pl = Plotter('Time (s)', 'Power Losses (%)', 'Power Loss History', \
                     OPATH / f'loss_pct.{args.imtype}', \
                     uncertainties=args.uncertainties)
        x, y = zip(*power_manager.loss_history_percentage)
        pl.plot_history(x, y)

    if 'pue' in args.plot:
        if cooling_model:
            ylabel = 'PUE_Out[1]'
            title = 'FMU ' + ylabel + 'History'
            pl = Plotter('Time (s)', ylabel, title, OPATH / f'pue.{args.imtype}', \
                         uncertainties=args.uncertainties)
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

    if args.uncertainties:
        # Parquet cannot handle annotated ufloat format AFAIK
        print('Data dump not implemented using uncertainties!')  
    else:
        if cooling_model:
            df = pd.DataFrame(cooling_model.fmu_history)
            df.to_parquet(OPATH / 'cooling_model.parquet', engine='pyarrow')

        df = pd.DataFrame(power_manager.history)
        df.to_parquet(OPATH / 'power_history.parquet', engine='pyarrow')

        df = pd.DataFrame(power_manager.loss_history)
        df.to_parquet(OPATH / 'loss_history.parquet', engine='pyarrow')

        df = pd.DataFrame(sc.sys_util_history)
        df.to_parquet(OPATH / 'util.parquet', engine='pyarrow')

        try:
            with open(OPATH / 'stats.out', 'w') as f:
                json.dump(output_stats, f, indent=4)
        except:
            write_dict_to_file(output_stats, OPATH / 'stats.out')
