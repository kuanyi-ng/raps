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

VALID_CHOICES = list(SYSTEMS.keys()) + ["synthetic", "hetero"]

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

def execute_system_tests(systems):
    """Execute tests for selected systems."""
    for system in systems:
        if system in SYSTEMS:
            command = build_command(system, SYSTEMS[system])
            run_command(command)

def synthetic_workload_tests():
    """Run synthetic workload tests."""
    print("Starting synthetic workload tests...")
    run_command(f"python main.py -t {DEFAULT_TIME}")
    run_command(f"python main.py -w benchmark -t {DEFAULT_TIME}")
    run_command(f"python main.py -w peak -t {DEFAULT_TIME}")
    run_command(f"python main.py -w idle -t {DEFAULT_TIME}")

def hetero_tests():
    """Run heterogeneous workload tests."""
    print("Starting heterogeneous workload tests...")
    run_command(f"python multi-part-sim.py -x setonix/part-cpu setonix/part-gpu -t {DEFAULT_TIME}")

def main():
    """Main function to parse arguments and run tests."""
    parser = argparse.ArgumentParser(description="Run smoke tests for HPC systems.")
    parser.add_argument(
        "tests",
        nargs="*",  # Allow multiple test selections
        choices=VALID_CHOICES,  # Allow specific systems, 'synthetic', and 'hetero'
        help="Run tests for one or more specific systems (e.g., 'frontier lassen'), 'synthetic' workloads, or 'hetero'. If omitted, all tests run.",
    )

    args = parser.parse_args()
    
    # If no arguments, run all tests
    if not args.tests:
        synthetic_workload_tests()
        hetero_tests()
        execute_system_tests(SYSTEMS.keys())
    else:
        if "synthetic" in args.tests:
            synthetic_workload_tests()
        if "hetero" in args.tests:
            hetero_tests()
        system_tests = [test for test in args.tests if test in SYSTEMS]
        if system_tests:
            execute_system_tests(system_tests)

if __name__ == "__main__":
    main()
