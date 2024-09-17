import numpy as np
import pandas as pd

from ..config import load_config_variables
from ..job import job_dict
from ..utils import power_to_utilization, next_arrival, encrypt

load_config_variables([
    'CPUS_PER_NODE',
    'GPUS_PER_NODE',
    'BLADES_PER_CHASSIS',
    'SC_SHAPE',
    'TRACE_QUANTA',
    'NODES_PER_BLADE',
    'POWER_GPU_IDLE',
    'POWER_GPU_MAX',
    'POWER_CPU_IDLE',
    'POWER_CPU_MAX'
], globals())


def load_data(files, **kwargs):
    """
    Reads job and job profile data from parquet files and parses them.

    Returns
    -------
    list
        The list of parsed jobs.
    """
    assert(len(files) == 2), "Frontier dataloader requires two files: joblive and jobprofile"

    jobs_path = files[0]
    jobs_df = pd.read_parquet(jobs_path, engine='pyarrow')

    jobprofile_path = files[1]
    jobprofile_df = pd.read_parquet(jobprofile_path, engine='pyarrow')

    return load_data_from_df(jobs_df, jobprofile_df, **kwargs)


def load_data_from_df(jobs_df: pd.DataFrame, jobprofile_df: pd.DataFrame, **kwargs):
    """
    Reads job and job profile data from dataframes files and parses them.

    Returns
    -------
    list
        The list of parsed jobs.
    """
    encrypt_bool = kwargs.get('encrypt')
    fastforward = kwargs.get('fastforward')
    reschedule = kwargs.get('reschedule')
    validate = kwargs.get('validate')
    jid = kwargs.get('jid', '*')

    if fastforward: print(f"fast-forwarding {fastforward} seconds")

    min_time = kwargs.get('min_time', None)

    # Sort jobs dataframe based on values in time_start column, adjust indices after sorting
    jobs_df = jobs_df[jobs_df['time_start'].notna()]
    jobs_df = jobs_df.drop_duplicates(subset='job_id', keep='last').reset_index()
    jobs_df = jobs_df.sort_values(by='time_start')
    jobs_df = jobs_df.reset_index(drop=True)

    # Convert timestamp column to datetime format
    jobprofile_df['timestamp'] = pd.to_datetime(jobprofile_df['timestamp'])

    # Sort allocation dataframe based on timestamp, adjust indices after sorting
    jobprofile_df = jobprofile_df.sort_values(by='timestamp')
    jobprofile_df = jobprofile_df.reset_index(drop=True)

    # Take earliest time as baseline reference
    if min_time:
        time_zero = min_time
    else:
        time_zero = jobs_df['time_snapshot'].min()

    num_jobs = len(jobs_df)
    print("time_zero:", time_zero, "num_jobs", num_jobs)

    jobs = []
    # Map dataframe to job state. Add results to jobs list
    for jidx in range(num_jobs - 1):

        job_id = jobs_df.loc[jidx, 'job_id']
        allocation_id = jobs_df.loc[jidx, 'allocation_id']
        nodes_required = jobs_df.loc[jidx, 'node_count']
        end_state = jobs_df.loc[jidx, 'state_current']
        name = jobs_df.loc[jidx, 'name']
        if encrypt_bool: name = encrypt(name)

        if validate:
            cpu_power = jobprofile_df[jobprofile_df['allocation_id']
                                      == allocation_id]['mean_node_power']
            cpu_trace = cpu_power.values
            gpu_trace = cpu_trace

        else:
            cpu_power = jobprofile_df[jobprofile_df['allocation_id']
                                      == allocation_id]['sum_cpu0_power']
            cpu_power_array = cpu_power.values
            cpu_min_power = nodes_required * POWER_CPU_IDLE * CPUS_PER_NODE
            cpu_max_power = nodes_required * POWER_CPU_MAX * CPUS_PER_NODE
            cpu_util = power_to_utilization(cpu_power_array, cpu_min_power, cpu_max_power)
            cpu_trace = cpu_util * CPUS_PER_NODE

            gpu_power = jobprofile_df[jobprofile_df['allocation_id']
                                      == allocation_id]['sum_gpu_power']
            gpu_power_array = gpu_power.values

            gpu_min_power = nodes_required * POWER_GPU_IDLE * GPUS_PER_NODE
            gpu_max_power = nodes_required * POWER_GPU_MAX * GPUS_PER_NODE
            gpu_util = power_to_utilization(gpu_power_array, gpu_min_power, gpu_max_power)
            gpu_trace = gpu_util * GPUS_PER_NODE

        # Set any NaN values in cpu_trace and/or gpu_trace to zero
        cpu_trace[np.isnan(cpu_trace)] = 0
        gpu_trace[np.isnan(gpu_trace)] = 0

        wall_time = gpu_trace.size * TRACE_QUANTA # seconds

        time_start = jobs_df.loc[jidx+1, 'time_start']
        diff = time_start - time_zero
        time_offset = max(diff.total_seconds(), 0)

        if fastforward: time_offset -= fastforward

        xnames = jobs_df.loc[jidx, 'xnames']
        # Don't replay any job with an empty set of xnames
        if '' in xnames: continue

        if reschedule: # Let the scheduler reschedule the jobs
            scheduled_nodes = None
            time_offset = next_arrival()
        else: # Prescribed replay
            scheduled_nodes = []
            for xname in xnames:
                indices = xname_to_index(xname)
                scheduled_nodes.append(indices)

        if gpu_trace.size > 0 and (jid == job_id or jid == '*') and time_offset > 0:
            job_info = job_dict(nodes_required, name, cpu_trace, gpu_trace, wall_time, 
                                end_state, scheduled_nodes, time_offset, job_id)
            jobs.append(job_info)

    return jobs


def xname_to_index(xname: str):
    """
    Converts an xname string to an index value based on system configuration.

    Parameters
    ----------
    xname : str
        The xname string to convert.

    Returns
    -------
    int
        The index value corresponding to the xname.
    """
    row, col = int(xname[2]), int(xname[3:5])
    chassis, slot, node = int(xname[6]), int(xname[8]), int(xname[10])
    if row == 6:
        col -= 9
    rack_index = row * 12 + col
    node_index = chassis * BLADES_PER_CHASSIS * NODES_PER_BLADE + slot * NODES_PER_BLADE + node
    return rack_index * SC_SHAPE[2] + node_index


def index_to_xname(index: int):
    """
    Converts an index value back to an xname string based on system configuration.

    Parameters
    ----------
    index : int
        The index value to convert.

    Returns
    -------
    str
        The xname string corresponding to the index.
    """
    rack_index = index // SC_SHAPE[2]
    node_index = index % SC_SHAPE[2]

    row = rack_index // 12
    col = rack_index % 12
    if row == 6:
        col += 9

    chassis = node_index // (BLADES_PER_CHASSIS * NODES_PER_BLADE)
    remaining = node_index % (BLADES_PER_CHASSIS * NODES_PER_BLADE)
    slot = remaining // NODES_PER_BLADE
    node = remaining % NODES_PER_BLADE

    return f"x2{row}{col:02}c{chassis}s{slot}b{node}"
