import uuid
import hashlib
import pandas as pd
import numpy as np

from raps.config import load_config_variables
from raps.utils import power_to_utilization, next_arrival

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
    'POWER_CPU_MAX',
    'POWER_NICS',
    'POWER_NVME', 
    'UI_UPDATE_FREQ'
], globals())


def load_data(jobs_path, **kwargs):
    """
    Reads job and job profile data from parquet files and parses them.

    Parameters
    ----------
    jobs_path : str
        The path to the jobs parquet file.

    Returns
    -------
    list
        The list of parsed jobs.
    """
    min_time = None
    encrypt_bool = kwargs.get('encrypt')
    reschedule = kwargs.get('reschedule')
    validate = kwargs.get('validate')
    jid = kwargs.get('jid')

    jobs_df = pd.read_parquet(jobs_path, engine='pyarrow')

    # Sort jobs dataframe based on values in time_start column, adjust indices after sorting
    jobs_df = jobs_df.sort_values(by='start_time')
    jobs_df = jobs_df.reset_index(drop=True)

    # Take earliest time as baseline reference
    # We can use the start time of the first job.
    if min_time:
        time_zero = min_time
    else:
        time_zero = jobs_df['start_time'].min()

    num_jobs = len(jobs_df)
    print("time_zero:", time_zero, "num_jobs", num_jobs)

    jobs = []
    # Map dataframe to job state. Add results to jobs list
    for i in range(num_jobs - 1):
        job_id = jobs_df.loc[i, 'job_id']

        if not jid == '*': 
            if int(jid) == int(job_id): 
                print(f'Extracting {job_id} profile')
            else:
                continue
        nodes_required = jobs_df.loc[i, 'num_nodes_alloc']

        name = str(uuid.uuid4())[:6]
            
        if validate:
            cpu_power = jobs_df.loc[i, 'node_power_consumption']/jobs_df.loc[i, 'num_nodes_alloc']
            cpu_trace = cpu_power
            gpu_trace = cpu_trace

        else:                
            cpu_power = jobs_df.loc[i, 'cpu_power_consumption']
            cpu_power_array = cpu_power.tolist()
            cpu_min_power = nodes_required * POWER_CPU_IDLE * CPUS_PER_NODE
            cpu_max_power = nodes_required * POWER_CPU_MAX * CPUS_PER_NODE
            cpu_util = power_to_utilization(cpu_power_array, cpu_min_power, cpu_max_power)
            cpu_trace = cpu_util * CPUS_PER_NODE
                
            node_power = (jobs_df.loc[i, 'node_power_consumption']).tolist()
            mem_power = (jobs_df.loc[i, 'mem_power_consumption']).tolist()
            # Find the minimum length among the three lists
            min_length = min(len(node_power), len(cpu_power), len(mem_power))
            # Slice each list to the minimum length
            node_power = node_power[:min_length]
            cpu_power = cpu_power[:min_length]
            mem_power = mem_power[:min_length]
                
            gpu_power = (node_power - cpu_power - mem_power
                - ([nodes_required * 2 * POWER_NICS] * len(node_power))
                - ([nodes_required * POWER_NVME] * len(node_power)))
            gpu_power_array = gpu_power.tolist()
            gpu_min_power = nodes_required * POWER_GPU_IDLE * GPUS_PER_NODE
            gpu_max_power = nodes_required * POWER_GPU_MAX * GPUS_PER_NODE
            gpu_util = power_to_utilization(gpu_power_array, gpu_min_power, gpu_max_power)
            gpu_trace = gpu_util * GPUS_PER_NODE
            
            # wall_time = jobs_df.loc[i, 'run_time']
            wall_time = gpu_trace.size * TRACE_QUANTA # seconds
            
            end_state = jobs_df.loc[i, 'job_state']
            
            time_start = jobs_df.loc[i+1, 'start_time']
            diff = time_start - time_zero
            if jid == '*': 
                time_offset = max(diff.total_seconds(), 0)
            else:
                # When extracting out a single job, run one iteration past the end of the job
                time_offset = UI_UPDATE_FREQ
            if reschedule: # Let the scheduler reschedule the jobs
                scheduled_nodes = None
                time_offset = next_arrival()
            else: # Prescribed replay
                scheduled_nodes = (jobs_df.loc[i, 'nodes']).tolist()
            
            jobs.append([
                nodes_required,
                name,
                cpu_trace,
                gpu_trace,
                wall_time,
                end_state,
                scheduled_nodes,
                time_offset,
                job_id
            ])

        return jobs