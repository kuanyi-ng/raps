import math
import os
import pandas as pd
from ..config import load_config_variables
from ..job import job_dict
from tqdm import tqdm
import time

load_config_variables(['TRACE_QUANTA'], globals())

def load_data_with_progress(path):
    print(f"reading {path}...")
    chunksize = 10000  # Adjust the chunk size as needed
    total_rows = sum(1 for _ in open(path)) - 1  # Total rows minus header
    
    chunk_list = []  # Store chunks
    for chunk in tqdm(pd.read_csv(path, chunksize=chunksize), total=total_rows//chunksize):
        chunk_list.append(chunk)
    
    # Concatenate all chunks into a single DataFrame
    df = pd.concat(chunk_list)
    return df

def load_data(path, **kwargs):
    """
    Loads data from the given file paths and returns job info.
    """
    nrows = 1E4
    #alloc_df = load_data_with_progress(os.path.join(path[0], 'final_csm_allocation_history_hashed.csv'))
    #node_df = load_data_with_progress(os.path.join(path[0], 'final_csm_allocation_node_history.csv'))
    #step_df = load_data_with_progress(os.path.join(path[0], 'final_csm_step_history.csv'))
    alloc_df = pd.read_csv(os.path.join(path[0], 'final_csm_allocation_history_hashed.csv'), nrows=nrows)
    node_df = pd.read_csv(os.path.join(path[0], 'final_csm_allocation_node_history.csv'), nrows=nrows)
    step_df = pd.read_csv(os.path.join(path[0], 'final_csm_step_history.csv'), nrows=nrows)
    return load_data_from_df(alloc_df, node_df, step_df, **kwargs)

def load_data_from_df(allocation_df, node_df, step_df, **kwargs):
    """
    Loads data from pandas DataFrames and returns the extracted job info.
    """
    allocation_df['begin_time'] = pd.to_datetime(allocation_df['begin_time'], format='mixed', errors='coerce')
    allocation_df['end_time'] = pd.to_datetime(allocation_df['end_time'], format='mixed', errors='coerce')

    earliest_begin_time = pd.to_datetime(allocation_df['begin_time']).min()
    job_list = []

    for _, row in tqdm(allocation_df.iterrows(), total=len(allocation_df), desc="Processing Jobs"):
        start_row = time.time()  # Start timing each row
        node_data = node_df[node_df['allocation_id'] == row['allocation_id']]

        wall_time_start = time.time()  # Timing wall_time calculation
        wall_time = compute_wall_time(row['begin_time'], row['end_time'])
        #print(f"Wall time calculation took: {time.time() - wall_time_start} seconds")
        samples = math.ceil(wall_time / TRACE_QUANTA)

        # Compute GPU power
        gpu_start = time.time()  # Timing GPU power calculation
        gpu_energy = node_data['gpu_energy'].sum()
        gpu_usage = node_data['gpu_usage'].sum()
        gpu_power = gpu_energy / (gpu_usage / 1E6) if gpu_usage > 0 else 0
        gpu_trace = [gpu_power] * samples
        #print(f"GPU power calculation took: {time.time() - gpu_start} seconds")

        # Compute CPU power (assuming total energy minus gpu_energy is cpu_energy)
        cpu_start = time.time()  # Timing CPU power calculation
        total_energy = node_data['energy'].sum()
        cpu_usage = node_data['cpu_usage'].sum()
        cpu_energy = total_energy - gpu_energy
        cpu_power = cpu_energy / (cpu_usage / 1E9) if cpu_usage > 0 else 0
        cpu_trace = [cpu_power] * samples
        #print(f"CPU power calculation took: {time.time() - cpu_start} seconds")

        job_info = job_dict(row['num_nodes'], \
                            row['primary_job_id'], \
                            cpu_trace, gpu_trace, wall_time, \
                            row['exit_status'], \
                            get_scheduled_nodes(row['allocation_id'], node_df), \
                            compute_time_offset(row['begin_time'], earliest_begin_time), \
                            row['primary_job_id'], \
                            row.get('priority', 0))
        job_list.append(job_info)
        #print(f"Processing this row took: {time.time() - start_row} seconds")

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
