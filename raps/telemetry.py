"""
This module provides functionality for handling telemetry data, including encryption,
index conversion, and job data parsing. It supports reading and saving snapshots,
parsing parquet files, and generating job state information.

The module defines a `Telemetry` class for managing telemetry data and several
helper functions for data encryption and conversion between xname and index formats.

Functions
---------
encrypt(name)
    Encrypts a given name using SHA-256 and returns the hexadecimal digest.
xname_to_index(xname)
    Converts an xname string to an index value based on system configuration.
index_to_xname(index)
    Converts an index value back to an xname string based on system configuration.

Classes
-------
Telemetry
    A class for handling telemetry data, including reading/parsing job data,
    saving snapshots, and converting job dataframes to job states.
"""

import argparse
import hashlib
import pandas as pd
import math
import numpy as np
import random

from .config import load_config_variables, is_config_initialized, initialize_config
if not is_config_initialized():
    initialize_config('frontier')

from .scheduler import Job
from .utils import power_to_utilization, next_arrival

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


def encrypt(name):
    """
    Encrypts a given name using SHA-256 and returns the hexadecimal digest.

    Parameters
    ----------
    name : str
        The name to be encrypted.

    Returns
    -------
    str
        The hexadecimal digest of the SHA-256 hash of the name.
    """
    encoded_name = name.encode()
    hash_object = hashlib.sha256(encoded_name)
    return hash_object.hexdigest()


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


class Telemetry:
    """
    A class for handling telemetry data, including reading/parsing job data,
    saving snapshots, and converting job dataframes to job states.

    Methods
    -------
    save_snapshot(jobs, filename)
        Saves a snapshot of the jobs to a compressed file.
    read_snapshot(snapshot)
        Reads a snapshot from a compressed file and returns the jobs.
    read_parquets(jobs_path, jobprofile_path)
        Reads job and job profile data from parquet files and parses them.
    parse_dataframes(jobs_df, jobprofile_df, min_time=None)
        Parses job and job profile dataframes to extract job state information.
    """

    def __init__(self, **kwargs):
        """
        Constructs all the necessary attributes for the Telemetry object.

        Parameters
        ----------
        encrypt: bool, optional
            Whether to encrypt job names (default is False).
        reschedule: bool, optional
            Whether to reschedule the workloads (as opposed to play as scheduled, default is False).
        validate : bool, optional
            Whether to validate job profiles (default is False).
        jid : str, optional
            Job ID to filter for specific jobs (default is '*').
        """
        self.encrypt = kwargs.get('encrypt')
        self.reschedule = kwargs.get('reschedule')
        self.validate = kwargs.get('validate')
        self.jid = kwargs.get('jid')


    def save_snapshot(self, jobs, filename):
        """
        Saves a snapshot of the jobs to a compressed file.

        Parameters
        ----------
        jobs : list
            The list of jobs to save.
        filename : str
            The name of the file to save the jobs to.
        """
        np.savez_compressed(filename, jobs=jobs)


    def read_snapshot(self, snapshot):
        """
        Reads a snapshot from a compressed file and returns the jobs.

        Parameters
        ----------
        snapshot : str
            The name of the snapshot file to read.

        Returns
        -------
        list
            The list of jobs from the snapshot file.
        """
        jobs = np.load(snapshot, allow_pickle=True)
        return jobs['jobs'].tolist()


    def read_parquets(self, jobs_path, jobprofile_path):
        """
        Reads job and job profile data from parquet files and parses them.

        Parameters
        ----------
        jobs_path : str
            The path to the jobs parquet file.
        jobprofile_path : str
            The path to the job profile parquet file.

        Returns
        -------
        list
            The list of parsed jobs.
        """
        jobs_df = pd.read_parquet(jobs_path, engine='pyarrow')
        jobprofile_df = pd.read_parquet(jobprofile_path, engine='pyarrow')
        return self.parse_dataframes(jobs_df, jobprofile_df)


    def parse_dataframes(self, jobs_df, jobprofile_df, min_time=None):
        """
        This function parses two pandas dataframes, joblive and jobprofile, to
        extract the necessary information to replay the jobs.

        Parameters:
            min_time: Value to use as zero time. Setting this to None will
                      autocompute the min value.
            snapshot: Boolean value to output a compressed .npz file for faster restarts

        Returns:
            jobs (arr): List of lists containing properties of each job
        """

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
            if self.encrypt:
                name = encrypt(name)

            if self.validate:
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

            xnames = jobs_df.loc[jidx, 'xnames']
            # Don't replay any job with an empty set of xnames
            if '' in xnames: continue

            if self.reschedule: # Let the scheduler reschedule the jobs
                scheduled_nodes = None
                time_offset = next_arrival()
            else: # Prescribed replay
                scheduled_nodes = []
                for xname in xnames:
                    indices = xname_to_index(xname)
                    scheduled_nodes.append(indices)

            if gpu_trace.size > 0 and (self.jid == job_id or self.jid == '*'):
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


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Telemetry data validator')
    parser.add_argument('-f', '--replay', nargs=2, type=str, default=[],
                        help='Paths of two telemetry parquet files')
    parser.add_argument('-v', '--verbose', action='store_true', help='Enable verbose output')
    args = parser.parse_args()

    td = Telemetry()
    jobs = td.read_parquets(args.replay[0], args.replay[1])
    timesteps = int(max(job[4] + job[7] for job in jobs))

    dt_list = []
    wt_list = []
    nr_list = []
    last = 0
    for job_vector in jobs:
        job = Job(job_vector, 0)
        wt_list.append(job.wall_time)
        nr_list.append(job.nodes_required)
        if job.submit_time > 0:
            dt = job.submit_time - last
            dt_list.append(dt)
            last = job.submit_time

        if args.verbose:
            print('jobid:', job.id, '\tlen(gpu_trace):', len(job.gpu_trace),
                  '\twall_time(s):', job.wall_time, '\tsubmit_time:', job.submit_time,
                  '\tend_time:', job.submit_time + job.wall_time)

    print(f'Simulation will run for {timesteps} seconds')
    print(f'Average job arrival time is: {np.mean(dt_list):.2f}s')
    print(f'Average wall time is: {np.mean(wt_list):.2f}s')
    print(f'Nodes required (avg): {np.mean(nr_list):.2f}')
    print(f'Nodes required (max): {np.max(nr_list)}')
    print(f'Nodes required (std): {np.std(nr_list):.2f}')

# =============================================================================
#     # Plot CPU/GPU Power and Calculate stats given specific job name
#     import matplotlib.pyplot as plt
#     import numpy as np
#     # Plotting
#     plt.figure(figsize=(10, 6))
#     plt.plot(td.job_cpu_data, label='CPU Power')
#     plt.plot(td.job_gpu_data, label='GPU Power')
#     plt.title('CPU and GPU Powers')
#     plt.xlabel('Index')
#     plt.ylabel('Power')
#     plt.legend()
#     plt.show()
#
# # Computation
#     cpu_avg = np.mean(td.job_cpu_data)
#     gpu_avg = np.mean(td.job_gpu_data)
#     cpu_min = np.min(td.job_cpu_data)
#     gpu_min = np.min(td.job_gpu_data)
#     cpu_max = np.max(td.job_cpu_data)
#     gpu_max = np.max(td.job_gpu_data)
#     cpu_std = np.std(td.job_cpu_data)
#     gpu_std = np.std(td.job_gpu_data)
#
#     # Print statements
#     print(f'CPU Average: {cpu_avg}')
#     print(f'GPU Average: {gpu_avg}')
#     print(f'CPU Minimum: {cpu_min}')
#     print(f'GPU Minimum: {gpu_min}')
#     print(f'CPU Maximum: {cpu_max}')
#     print(f'GPU Maximum: {gpu_max}')
#     print(f'CPU Standard Deviation: {cpu_std}')
#     print(f'GPU Standard Deviation: {gpu_std}')
# =============================================================================
