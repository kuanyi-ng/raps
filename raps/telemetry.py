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
    parser.add_argument('-p', '--plot', action='store_true', help='Output plots') 
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
        jobs = np.load(snapshot, allow_pickle=True)
        return jobs['jobs'].tolist()


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
    timesteps = int(max(job['wall_time'] + job['submit_time'] for job in jobs))

    dt_list = []
    wt_list = []
    nr_list = []
    submit_times = []
    last = 0
    for job_vector in jobs:
        job = Job(job_vector, 0)
        wt_list.append(job.wall_time)
        nr_list.append(job.nodes_required)
        submit_times.append(job.submit_time)
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


    if args.plot:

        import matplotlib.pyplot as plt

        print("plotting nodes required histogram...")
        # Define the number of bins you want
        num_bins = 25
        data = nr_list
        # Create logarithmically spaced bins
        bins = np.logspace(np.log2(min(data)), np.log2(max(data)), num=num_bins, base=2)
        # Set up the figure 
        plt.figure(figsize=(10, 3))
        # Create the histogram
        plt.hist(nr_list, bins=bins, edgecolor='black')
        # Add a title and labels
        plt.xlabel('Number of Nodes')
        plt.ylabel('Frequency')
        # Set the axes to logarithmic scale
        plt.xscale('log', base=2)
        plt.yscale('log')
        # Customize the x-ticks: Choose the positions (1, 8, 64, etc.)
        ticks = [2**i for i in range(0, 14)]
        plt.xticks(ticks, labels=[str(tick) for tick in ticks])
        # Set min-max axis bounds
        plt.xlim(1, max(data))
        # Save the histogram to a file
        plt.savefig('histogram.png', dpi=300, bbox_inches='tight')

        print("plotting submit times...")
        # Plot number of nodes over time
        plt.clf()
        plt.figure(figsize=(10, 2))
        # Create a bar chart
        plt.bar(submit_times, nr_list, width=10.0, color='blue', edgecolor='black', alpha=0.7)
        # Add labels and title
        plt.xlabel('Submit Time (s)')
        plt.ylabel('Number of Nodes')
        # Set min-max axix bounds
        plt.xlim(1, max(submit_times))
        # Set the y-axis to logarithmic scale with base 2
        plt.yscale('log', base=2)
        y_ticks = [2**i for i in range(0, 14)]
        plt.yticks(y_ticks, labels=[str(tick) for tick in y_ticks])
        # Save the plot to a file
        plt.savefig('nodes_time.png', dpi=300, bbox_inches='tight')

