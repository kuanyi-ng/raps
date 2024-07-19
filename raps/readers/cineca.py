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
    'UI_UPDATE_FREQ'
], globals())


def read_parquets(jobs_path):
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
    jobs_df = pd.read_parquet(jobs_path, engine='pyarrow')

    # Sort jobs dataframe based on values in time_start column, adjust indices after sorting
    '''
    jobs_df = jobs_df[jobs_df['time_start'].notna()]
    jobs_df = jobs_df.drop_duplicates(subset='job_id', keep='last').reset_index()
    jobs_df = jobs_df.sort_values(by='time_start')
    jobs_df = jobs_df.reset_index(drop=True)
    '''
    jobs_df = jobs_df.sort_values(by='start_time')
    jobs_df = jobs_df.reset_index(drop=True)

    # Convert timestamp column to datetime format
    # jobprofile_df['timestamp'] = pd.to_datetime(jobprofile_df['timestamp'])

    # Sort allocation dataframe based on timestamp, adjust indices after sorting
    # jobprofile_df = jobprofile_df.sort_values(by='timestamp')
    # jobprofile_df = jobprofile_df.reset_index(drop=True)

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

        #if not self.jid == '*': 
        #    if int(self.jid) == int(job_id): 
        #        print(f'Extracting {job_id} profile')
        #    else:
        #        continue

        nodes_required = jobs_df.loc[i, 'num_nodes_alloc']
        
        name = str(uuid.uuid4())[:6]
        
        cpu_power = jobs_df.loc[i, 'cpu_power_consumption']
        cpu_power_array = cpu_power.tolist()
        cpu_min_power = nodes_required * POWER_CPU_IDLE * CPUS_PER_NODE
        cpu_power_array = [cpu_min_power if x < cpu_min_power else x for x in cpu_power_array]
        cpu_max_power = nodes_required * POWER_CPU_MAX * CPUS_PER_NODE
        cpu_util = power_to_utilization(cpu_power_array, cpu_min_power, cpu_max_power)
        cpu_trace = cpu_util * CPUS_PER_NODE
        
        # gpus_required = jobs_df.loc[i, 'num_gpus_alloc']
        node_power = (jobs_df.loc[i, 'node_power_consumption']).tolist()
        mem_power = (jobs_df.loc[i, 'mem_power_consumption']).tolist()
        # Find the minimum length among the three lists
        min_length = min(len(node_power), len(cpu_power), len(mem_power))
        # Slice each list to the minimum length
        node_power = node_power[:min_length]
        cpu_power = cpu_power[:min_length]
        mem_power = mem_power[:min_length]
        gpu_power = node_power - cpu_power - mem_power
        gpu_power_array = gpu_power.tolist()
        gpu_min_power = nodes_required * POWER_GPU_IDLE * GPUS_PER_NODE
        gpu_power_array = [gpu_min_power if x < gpu_min_power else x for x in gpu_power_array]
        gpu_max_power = nodes_required * POWER_GPU_MAX * GPUS_PER_NODE
        gpu_util = power_to_utilization(gpu_power_array, gpu_min_power, gpu_max_power)
        gpu_trace = gpu_util * GPUS_PER_NODE
        
        # wall_time = jobs_df.loc[i, 'run_time']
        wall_time = gpu_trace.size * TRACE_QUANTA # seconds
        
        end_state = jobs_df.loc[i, 'job_state']
        
        
        time_start = jobs_df.loc[i+1, 'start_time']
        diff = time_start - time_zero
        #if self.jid == '*': 
        if True:
            time_offset = max(diff.total_seconds(), 0)
        else:
            # When extracting out a single job, run one iteration past the end of the job
            time_offset = UI_UPDATE_FREQ

        #if self.reschedule: # Let the scheduler reschedule the jobs
        if False:
            scheduled_nodes = None
            time_offset = next_arrival()
        else: # Prescribed replay
            scheduled_nodes = (jobs_df.loc[i, 'nodes']).tolist()
            #scheduled_nodes = []
            #for xname in xnames:
            #    indices = xname_to_index(xname)
            #    scheduled_nodes.append(indices)
        
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
