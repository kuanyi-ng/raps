import pandas as pd
from ..job import job_dict


def load_data(path, **kwargs):
    """
    Loads data from the given Parquet file path and returns job info.

    Parameters:
    path (str): Path to the Parquet file.
    
    Returns:
    list: List of job dictionaries.
    """
    # Load the parquet file
    parquet_file = path[0]  # Assuming path is a list containing the path to the parquet file
    df = pd.read_parquet(parquet_file)

    # Process the DataFrame and pass to load_data_from_df
    return load_data_from_df(df, **kwargs)


def load_data_from_df(df, **kwargs):
    """
    Processes DataFrame to extract relevant job information.

    Parameters:
    df (pd.DataFrame): DataFrame containing job information.
    
    Returns:
    list: List of job dictionaries.
    """
    job_list = []
    
    # Loop through the DataFrame rows to extract job information
    for _, row in df.iterrows():
        nodes_required = row['nnumr'] if 'nnumr' in df.columns else 0
        name = row['jnam'] if 'jnam' in df.columns else 'unknown'
        cpu_trace = row['perf1'] if 'perf1' in df.columns else 0  # Assuming some performance metric as cpu_trace
        gpu_trace = 0  # Set to 0 as GPU trace is not explicitly provided
        wall_time = row['duration'] if 'duration' in df.columns else 0
        end_state = row['exit state'] if 'exit state' in df.columns else 'unknown'
        scheduled_nodes = row['nnuma'] if 'nnuma' in df.columns else 0
        time_offset = row['adt'] if 'adt' in df.columns else pd.Timestamp(0)  # Submission time
        job_id = row['jid'] if 'jid' in df.columns else 'unknown'
        priority = row['pri'] if 'pri' in df.columns else 0
        
        # Create job dictionary
        job_info = job_dict(
            nodes_required=nodes_required,
            name=name,
            cpu_trace=cpu_trace,
            gpu_trace=gpu_trace,
            wall_time=wall_time,
            end_state=end_state,
            scheduled_nodes=scheduled_nodes,
            time_offset=time_offset,
            job_id=job_id,
            priority=priority
        )
        
        job_list.append(job_info)
    
    return job_list

# Sample usage:
# fugaku_jobs = load_data(['/path/to/21_04.parquet'])
