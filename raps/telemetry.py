"""
This module provides functionality for handling telemetry data, including encryption,
index conversion, and job data parsing. It supports reading and saving snapshots,
parsing parquet files, and generating job state information.

The module defines a `Telemetry` class for managing telemetry data and several
helper functions for data encryption and conversion between xname and index formats.
"""

import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Telemetry data validator')
    parser.add_argument('--jid', type=str, default='*', help='Replay job id')
    parser.add_argument('-f', '--replay', nargs='+', type=str, 
                        help='Either: path/to/joblive path/to/jobprofile' + \
                             ' -or- filename.npz (overrides --workload option)')
    parser.add_argument('-p', '--plot', nargs='+', choices=['power', 'loss', 'pue', 'temp'],
                        help='Specify one or more types of plots to generate: power, loss, pue, temp')
    parser.add_argument('--system', type=str, default='frontier', help='System config to use')
    parser.add_argument('-v', '--verbose', action='store_true', help='Enable verbose output')
    args = parser.parse_args()

    from .config import is_config_initialized, initialize_config
    if not is_config_initialized():
        initialize_config(args.system)

import importlib
import numpy as np
import re
from datetime import datetime
from .scheduler import Job


class Telemetry:
    """A class for handling telemetry data, including reading/parsing job data, and loading/saving snapshots."""

    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.system = kwargs.get('system')


    def save_snapshot(self, jobs: list, filename: str):
        """Saves a snapshot of the jobs to a compressed file. """
        np.savez_compressed(filename, jobs=jobs)


    def load_snapshot(self, snapshot: str) -> list:
        """Reads a snapshot from a compressed file and returns the jobs."""
        # Add meta data for start date
        match = re.search(r'\d{4}-\d{2}-\d{2}', snapshot)
        if match:
            date_str = match.group()  # Extract the date string
            
            # Convert to datetime object
            start = datetime.strptime(date_str, "%Y-%m-%d")
        
        jobs = np.load(snapshot, allow_pickle=True)
        return jobs['jobs'].tolist(), start


    def load_data(self, files):
        """Load telemetry data using custom data loaders."""
        module = importlib.import_module(f".dataloaders.{self.system}", package=__package__)
        return module.load_data(files, **self.kwargs)


    def load_data_from_df(self, *args, **kwargs):
        """Load telemetry data using custom data loaders."""
        module = importlib.import_module(f".dataloaders.{self.system}", package=__package__)
        return module.load_data_from_df(*args, **kwargs)


if __name__ == "__main__":

    args_dict = vars(args)
    td = Telemetry(**args_dict)
    jobs = td.load_data(args.replay)
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
        if args.verbose: print(job)

    print(f'Simulation will run for {timesteps} seconds')
    print(f'Average job arrival time is: {np.mean(dt_list):.2f}s')
    print(f'Average wall time is: {np.mean(wt_list):.2f}s')
    print(f'Nodes required (avg): {np.mean(nr_list):.2f}')
    print(f'Nodes required (max): {np.max(nr_list)}')
    print(f'Nodes required (std): {np.std(nr_list):.2f}')
