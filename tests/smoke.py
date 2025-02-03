import os
import argparse
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
    "lassen": "lassen/Lassen-Supercomputer-Job-Dataset",
    "adastraMI250": "adastra/AdastaJobsMI250_15days.parquet"
}

VALID_CHOICES = list(SYSTEMS.keys()) + ["synthetic"]

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

def execute_system_tests(system=None):
    """Execute tests for all systems or a specific system."""
    if system:
        command = build_command(system, SYSTEMS[system])
        run_command(command)
    else:
        for sys_name, file_paths in SYSTEMS.items():
            command = build_command(sys_name, file_paths)
            run_command(command)

def synthetic_workload_tests():
    """Run synthetic workload tests."""
    print("Starting synthetic workload tests...")
    run_command(f"python main.py -t {DEFAULT_TIME}")
    run_command(f"python main.py -w benchmark -t {DEFAULT_TIME}")
    run_command(f"python main.py -w peak -t {DEFAULT_TIME}")
    run_command(f"python main.py -w idle -t {DEFAULT_TIME}")
    run_command(f"python multi-part-sim.py -x setonix/part-cpu setonix/part-gpu -t {DEFAULT_TIME}")

def main():
    """Main function to parse arguments and run tests."""
    parser = argparse.ArgumentParser(description="Run smoke tests for HPC systems.")
    parser.add_argument(
        "system",
        nargs="?",  # Optional argument
        choices=VALID_CHOICES,  # Allow specific systems and 'synthetic'
        help="Run tests for a specific system (e.g., 'frontier') or 'synthetic' workloads. If omitted, all tests run.",
    )

    args = parser.parse_args()

    if args.system == "synthetic":
        synthetic_workload_tests()
    elif args.system:
        execute_system_tests(args.system)
    else:
        # If no argument, run all tests
        synthetic_workload_tests()
        execute_system_tests()

if __name__ == "__main__":
    main()

