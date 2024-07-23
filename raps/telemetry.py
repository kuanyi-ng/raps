"""
This module provides functionality for handling telemetry data, including encryption,
index conversion, and job data parsing. It supports reading and saving snapshots,
parsing parquet files, and generating job state information.

The module defines a `Telemetry` class for managing telemetry data and several
helper functions for data encryption and conversion between xname and index formats.

Classes
-------
Telemetry
    A class for handling telemetry data, including reading/parsing job data,
    saving snapshots, and converting job dataframes to job states.
"""

import argparse
import hashlib
import importlib
import numpy as np

from .scheduler import Job

def encrypt(name):
    """Encrypts a given name using SHA-256 and returns the hexadecimal digest."""

    """
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


class Telemetry:
    """
    A class for handling telemetry data, including reading/parsing job data,
    saving snapshots, and converting job dataframes to job states.

    Methods
    -------
    save_snapshot(jobs, filename)
        Saves a snapshot of the jobs to a compressed file.
    load_snapshot(snapshot)
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
        self.kwargs = kwargs
        self.system = kwargs.get('system')
        #self.encrypt = kwargs.get('encrypt')
        #self.reschedule = kwargs.get('reschedule')
        #self.validate = kwargs.get('validate')
        #self.jid = kwargs.get('jid')


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


    def load_snapshot(self, snapshot):
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

    def load_data(self, files):
        module = importlib.import_module('raps.dataloaders.' + self.system)
        return module.load_data(files, **self.kwargs)


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
