"""
Reference:

    Patki, Tapasya, et al. "Monitoring large scale supercomputers: A case study with the Lassen supercomputer." 
    2021 IEEE International Conference on Cluster Computing (CLUSTER). IEEE, 2021.

Instructions:

    git clone https://github.com/LLNL/LAST/ && cd LAST
    git lfs pull
    
    # to analyze dataset
    python -m raps.telemetry -f /path/to/LAST/Lassen-Supercomputer-Job-Dataset --system lassen -v
    
    # to simulate the dataset    
    python main.py -f /path/to/LAST/Lassen-Supercomputer-Job-Dataset --system lassen

    # to reschedule
    python main.py -f /path/to/LAST/Lassen-Supercomputer-Job-Dataset --system lassen --reschedule

    # to fast-forward 37 days and replay for 1 day
    python main.py -f /path/to/LAST/Lassen-Supercomputer-Job-Dataset --system lassen -ff 37d -t 1d
"""
import math
import numpy as np
import os
import pandas as pd
from ..config import load_config_variables
from ..job import job_dict
from ..utils import power_to_utilization, next_arrival
from tqdm import tqdm
import time

load_config_variables(['TRACE_QUANTA', 'CPUS_PER_NODE', 'GPUS_PER_NODE', 
                       'POWER_GPU_IDLE', 'POWER_GPU_MAX', 'POWER_CPU_IDLE',
                       'POWER_CPU_MAX', 'POWER_MEM', 'POWER_NIC', 'POWER_NVME',
                       'NICS_PER_NODE'], globals())

def load_data(path, **kwargs):
    """
    Loads data from the given file paths and returns job info.
    """
    nrows = 1E4
    alloc_df = pd.read_csv(os.path.join(path[0], 'final_csm_allocation_history_hashed.csv'), nrows=nrows)
    node_df = pd.read_csv(os.path.join(path[0], 'final_csm_allocation_node_history.csv'), nrows=nrows)
    step_df = pd.read_csv(os.path.join(path[0], 'final_csm_step_history.csv'), nrows=nrows)
    return load_data_from_df(alloc_df, node_df, step_df, **kwargs)

def load_data_from_df(allocation_df, node_df, step_df, **kwargs):
    """
    Loads data from pandas DataFrames and returns the extracted job info.
    """
    reschedule = kwargs.get('reschedule')
    fastforward = kwargs.get('fastforward')
    if fastforward: print(f"fast-forwarding {fastforward} seconds")

    allocation_df['begin_time'] = pd.to_datetime(allocation_df['begin_time'], format='mixed', errors='coerce')
    allocation_df['end_time'] = pd.to_datetime(allocation_df['end_time'], format='mixed', errors='coerce')

    earliest_begin_time = pd.to_datetime(allocation_df['begin_time']).min()
    print(earliest_begin_time)
    job_list = []

    for _, row in tqdm(allocation_df.iterrows(), total=len(allocation_df), desc="Processing Jobs"):
        node_data = node_df[node_df['allocation_id'] == row['allocation_id']]

        nodes_required = row['num_nodes']

        wall_time = compute_wall_time(row['begin_time'], row['end_time'])
        samples = math.ceil(wall_time / TRACE_QUANTA)

        # Compute GPU power
        gpu_energy = node_data['gpu_energy'].sum() # Joules
        # divide by nodes_required to get average gpu_usage per node
        gpu_usage = node_data['gpu_usage'].sum() / 1E6 / nodes_required # seconds
        gpu_power = gpu_energy / gpu_usage if gpu_usage > 0 else 0
        #gpu_power = gpu_energy / wall_time
        gpu_power_array = np.array([gpu_power] * samples)

        gpu_min_power = nodes_required * POWER_GPU_IDLE
        gpu_max_power = nodes_required * POWER_GPU_MAX
        gpu_util = power_to_utilization(gpu_power_array, gpu_min_power, gpu_max_power)
        gpu_trace = gpu_util * GPUS_PER_NODE

        # Compute CPU power (assuming total energy minus gpu_energy is cpu_energy)
        total_energy = node_data['energy'].sum() # Joules
        cpu_energy = total_energy - gpu_energy 

        cpu_usage = node_data['cpu_usage'].sum() / 1E9 / nodes_required # seconds
        cpu_power = cpu_energy / cpu_usage if cpu_usage > 0 else 0
        #cpu_power = cpu_energy / wall_time 
        cpu_power -= nodes_required * (POWER_MEM + NICS_PER_NODE * POWER_NIC + POWER_NVME)
        cpu_power_array = np.array([cpu_power] * samples)

        cpu_min_power = nodes_required * POWER_CPU_IDLE
        cpu_max_power = nodes_required * POWER_CPU_MAX
        cpu_util = power_to_utilization(cpu_power_array, cpu_min_power, cpu_max_power)
        cpu_trace = cpu_util * CPUS_PER_NODE

        if reschedule: # Let the scheduler reschedule the jobs
            scheduled_nodes = None
            time_offset = next_arrival()
        else:
            scheduled_nodes = get_scheduled_nodes(row['allocation_id'], node_df)
            time_offset = compute_time_offset(row['begin_time'], earliest_begin_time)
            if fastforward: time_offset -= fastforward

        if time_offset >= 0:

            job_info = job_dict(nodes_required, \
                                row['hashed_user_id'], \
                                cpu_trace, gpu_trace, wall_time, \
                                row['exit_status'], \
                                scheduled_nodes, \
                                time_offset, \
                                row['primary_job_id'], \
                                row.get('priority', 0))

            job_list.append(job_info)

    return job_list

def get_scheduled_nodes(allocation_id, node_df):
    """
    Gets the list of scheduled nodes for a given allocation.
    """
    node_data = node_df[node_df['allocation_id'] == allocation_id]
    if 'node_name' in node_data.columns:
        node_list = [int(node.split('lassen')[-1]) for node in node_data['node_name'].tolist()]
        return node_list
    return []

def compute_wall_time(begin_time, end_time):
    """
    Computes the wall time for the job.
    """
    wall_time = pd.to_datetime(end_time) - pd.to_datetime(begin_time)
    return int(wall_time.total_seconds())

def compute_time_offset(begin_time, reference_time):
    """
    Computes the time offset from a reference time.
    """
    time_offset = pd.to_datetime(begin_time) - reference_time
    return int(time_offset.total_seconds())

# Example usage
if __name__ == "__main__":
    path = "/Users/w1b/data/LAST/Lassen-Supercomputer-Job-Dataset/"
    jobs = load_data(path)
    print(jobs)
