import os
import pandas as pd

def load_data(path, **kwargs):
    """
    Loads data from the given file paths and returns job info.
    """
    nrows = 1000
    alloc_df = pd.read_csv(os.path.join(path[0], 'final_csm_allocation_history_hashed.csv'), nrows=nrows)
    node_df = pd.read_csv(os.path.join(path[0], 'final_csm_allocation_node_history.csv'), nrows=nrows)
    step_df = pd.read_csv(os.path.join(path[0], 'final_csm_step_history.csv'), nrows=nrows)
    return load_data_from_df(alloc_df, node_df, step_df, **kwargs)

def load_data_from_df(allocation_df, node_df, step_df, **kwargs):
    """
    Loads data from pandas DataFrames and returns the extracted job info.
    """
    earliest_begin_time = pd.to_datetime(allocation_df['begin_time']).min()
    job_list = []
    for _, row in allocation_df.iterrows():
        #job_info = {
        #    'nodes_required': row['num_nodes'],
        #    'name': row['primary_job_id'],
        #    'cpu_trace': get_cpu_trace(row['allocation_id'], node_df),
        #    'gpu_trace': get_gpu_trace(row['allocation_id'], node_df),
        #    'wall_time': compute_wall_time(row['begin_time'], row['end_time']),
        #    'end_state': row['exit_status'],
        #    'scheduled_nodes': get_scheduled_nodes(row['allocation_id'], node_df),
        #    'time_offset': compute_time_offset(row['begin_time'], earliest_begin_time),
        #    'job_id': row['primary_job_id'],
        #    'priority': row.get('priority', 0)  # Default to 0 if priority is not available
        #}
        job_info = [row['num_nodes'], row['primary_job_id'], get_cpu_trace(row['allocation_id'], node_df), get_gpu_trace(row['allocation_id'], node_df), compute_wall_time(row['begin_time'], row['end_time']), row['exit_status'], get_scheduled_nodes(row['allocation_id'], node_df), compute_time_offset(row['begin_time'], earliest_begin_time), row['primary_job_id'], row.get('priority', 0)]
        job_list.append(job_info)
    return job_list

def get_cpu_trace(allocation_id, node_df):
    """
    Gets the CPU trace for a given allocation from node history.
    """
    node_data = node_df[node_df['allocation_id'] == allocation_id]
    return node_data['cpu_usage'].tolist() if 'cpu_usage' in node_data.columns else []

def get_gpu_trace(allocation_id, node_df):
    """
    Gets the GPU trace for a given allocation from node history.
    """
    node_data = node_df[node_df['allocation_id'] == allocation_id]
    return node_data['gpu_usage'].tolist() if 'gpu_usage' in node_data.columns else []

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
