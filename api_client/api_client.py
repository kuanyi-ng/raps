import os
import argparse
import requests
import pandas as pd
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
URL = os.getenv("BASE_URL")
RAPS_URL = os.path.join(URL, "exadigit/api")

def read_token():
    with open('.api-token', 'r') as token_file:
        return token_file.read().strip()

def call_api(endpoint, method="GET", params=None, data=None):
    TOKEN = read_token()
    url = f"{RAPS_URL}{endpoint}"
    headers = {"Authorization": f"Bearer {TOKEN}"}
    
    response = requests.request(method, url, headers=headers, params=params, json=data)
    
    if response.status_code == 200:
        return response.json()
    else:
        print(f"Error: {response.status_code} - {response.text}")
        return None

def handle_run(args):
    data = {"system": args.system, "policy": args.policy, "parameters": args.parameters}
    response = call_api('/simulation/run', method="POST", data=data)
    print(response)

def handle_list(args):
    response = call_api('/simulation/list')
    if response:
        results = response.get('results', [])
        df = pd.DataFrame(results)
        #pd.set_option('display.max_columns', None)
        #pd.set_option('display.max_colwidth', None)
        #pd.set_option('display.width', None)
        print(df)

def handle_simulation_details(args):
    response = call_api(f'/simulation/{args.id}')
    print(response)

def handle_cooling_cdu(args):
    response = call_api(f'/simulation/{args.id}/cooling/cdu')
    print(response)

def handle_cooling_cep(args):
    response = call_api(f'/simulation/{args.id}/cooling/cep')
    print(response)

def handle_scheduler_jobs(args):
    response = call_api(f'/simulation/{args.id}/scheduler/jobs')
    print(response)

def handle_power_history(args):
    response = call_api(f'/simulation/{args.id}/scheduler/jobs/{args.job_id}/power-history')
    print(response)

def handle_scheduler_system(args):
    response = call_api(f'/simulation/{args.id}/scheduler/system')
    print(response)

def handle_system_info(args):
    response = call_api(f'/system-info/{args.system}')
    print(response)

def main():
    parser = argparse.ArgumentParser(description="Interact with the SimulationServer REST API.")
    subparsers = parser.add_subparsers(title="commands", dest="command")
    
    # Run simulation
    run_parser = subparsers.add_parser("run", help="Run a simulation.")
    run_parser.add_argument("--system", required=True, help="System to run the simulation on.")
    run_parser.add_argument("--policy", required=True, help="Policy to use.")
    run_parser.add_argument("--parameters", type=dict, default={}, help="Simulation parameters.")
    run_parser.set_defaults(func=handle_run)
    
    # List simulations
    list_parser = subparsers.add_parser("list", help="List all simulations.")
    list_parser.set_defaults(func=handle_list)
    
    # Get simulation details
    details_parser = subparsers.add_parser("details", help="Get details of a simulation.")
    details_parser.add_argument("--id", required=True, help="Simulation ID.")
    details_parser.set_defaults(func=handle_simulation_details)
    
    # Cooling CDU
    cdu_parser = subparsers.add_parser("cooling-cdu", help="Get cooling CDU data for a simulation.")
    cdu_parser.add_argument("--id", required=True, help="Simulation ID.")
    cdu_parser.set_defaults(func=handle_cooling_cdu)
    
    # Cooling CEP
    cep_parser = subparsers.add_parser("cooling-cep", help="Get cooling CEP data for a simulation.")
    cep_parser.add_argument("--id", required=True, help="Simulation ID.")
    cep_parser.set_defaults(func=handle_cooling_cep)
    
    # Scheduler jobs
    jobs_parser = subparsers.add_parser("scheduler-jobs", help="Get scheduler jobs for a simulation.")
    jobs_parser.add_argument("--id", required=True, help="Simulation ID.")
    jobs_parser.set_defaults(func=handle_scheduler_jobs)
    
    # Power history
    power_parser = subparsers.add_parser("power-history", help="Get power history for a specific job in a simulation.")
    power_parser.add_argument("--id", required=True, help="Simulation ID.")
    power_parser.add_argument("--job-id", required=True, help="Job ID.")
    power_parser.set_defaults(func=handle_power_history)
    
    # Scheduler system
    scheduler_parser = subparsers.add_parser("scheduler-system", help="Get scheduler system data for a simulation.")
    scheduler_parser.add_argument("--id", required=True, help="Simulation ID.")
    scheduler_parser.set_defaults(func=handle_scheduler_system)
    
    # System info
    system_info_parser = subparsers.add_parser("system-info", help="Get system information.")
    system_info_parser.add_argument("--system", required=True, help="System name.")
    system_info_parser.set_defaults(func=handle_system_info)
    
    # Parse and execute
    args = parser.parse_args()
    if args.command:
        args.func(args)
    else:
        parser.print_help()

if __name__ == "__main__":
    main()
