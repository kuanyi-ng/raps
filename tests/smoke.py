import os
import subprocess

# Define the data path
DATAPATH = os.path.expanduser("~/data")

# Standardize the time setting
DEFAULT_TIME = "1h"

# Define systems and their corresponding filenames
SYSTEMS = {
    "frontier": "frontier/slurm/joblive/date=2024-01-18 frontier/jobprofile/date=2024-01-18",
    "fugaku": "fugaku/21_04.parquet",
    "marconi100": "marconi100/job_table.parquet",
    "lassen": "lassen/Lassen-Supercomputer-Job-Dataset"
}

def run_command(command):
    """Helper function to run a shell command."""
    print(f"Running: {command}")
    result = subprocess.run(command, shell=True)
    if result.returncode != 0:
        print(f"Error: Command failed with return code {result.returncode}")
        exit(-1)

def build_command(system, file_paths, additional_args=""):
    """Build the command string for the given system and file paths."""
    full_paths = " ".join([os.path.join(DATAPATH, path) for path in file_paths.split()])
    return f"python main.py --system {system} -f {full_paths} -t {DEFAULT_TIME} {additional_args}".strip()

def execute_system_tests():
    """Execute tests for all systems."""
    for system, file_paths in SYSTEMS.items():
        command = build_command(system, file_paths)
        run_command(command)

def synthetic_workload_tests():
    """Run synthetic workload tests."""
    print("Starting synthetic workload tests...")
    run_command(f"python main.py -t {DEFAULT_TIME}")
    run_command(f"python main.py -w benchmark -t {DEFAULT_TIME}")
    run_command(f"python main.py -w peak -t {DEFAULT_TIME}")
    run_command(f"python main.py -w idle -t {DEFAULT_TIME}")

def main():
    """Main function to run all tests."""
    synthetic_workload_tests()
    execute_system_tests()

if __name__ == "__main__":
    main()

