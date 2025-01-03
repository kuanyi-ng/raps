import argparse
from raps.policy import PolicyType

parser = argparse.ArgumentParser(description='Resource Allocator & Power Simulator (RAPS)')
parser.add_argument('-c', '--cooling', action='store_true', help='Include FMU cooling model')
parser.add_argument('--start', type=str, help='ISO8061 string for start of simulation')
parser.add_argument('--end', type=str, help='ISO8061 string for end of simulation')
parser.add_argument('-d', '--debug', action='store_true', help='Enable debug mode and disable rich layout')
parser.add_argument('-e', '--encrypt', action='store_true', help='Encrypt any sensitive data in telemetry')
parser.add_argument('-n', '--numjobs', type=int, default=1000, help='Number of jobs to schedule')
parser.add_argument('-t', '--time', type=str, default=None, help='Length of time to simulate, e.g., 123, 123s, 27m, 3h, 7d')
parser.add_argument('-ff', '--fastforward', type=str, default=None, help='Fast-forward by time amount (uses same units as -t)')
parser.add_argument('-v', '--verbose', action='store_true', help='Enable verbose output')
parser.add_argument('--seed', action='store_true', help='Set random number seed for deterministic simulation')
parser.add_argument('-f', '--replay', nargs='+', type=str, help='Either: path/to/joblive path/to/jobprofile' + \
                                                                ' -or- filename.npz (overrides --workload option)')
parser.add_argument('--reschedule', action='store_true', help='Reschedule the telemetry workload')
parser.add_argument('-u', '--uncertainties', action='store_true',
                    help='Change from floating point units to floating point units with uncertainties.' + \
                                                                ' Very expensive w.r.t simulation time!')
parser.add_argument('--jid', type=str, default='*', help='Replay job id')
parser.add_argument('--validate', action='store_true', help='Use node power instead of CPU/GPU utilizations')
parser.add_argument('-o', '--output', action='store_true', help='Output power, cooling, and loss models for later analysis')
parser.add_argument('-p', '--plot', nargs='+', choices=['power', 'loss', 'pue', 'temp', 'util'],
                    help='Specify one or more types of plots to generate: power, loss, pue, util, temp')
choices = ['png', 'svg', 'jpg', 'pdf', 'eps']
parser.add_argument('--imtype', type=str, choices=choices, default=choices[0], help='Plot image type')
parser.add_argument('--system', type=str, default='frontier', help='System config to use')
choices = [policy.value for policy in PolicyType]
parser.add_argument('-s', '--schedule', type=str, choices=choices, default=choices[0], help='Schedule policy to use')
choices = ['random', 'benchmark', 'peak', 'idle']
parser.add_argument('-w', '--workload', type=str, choices=choices, default=choices[0], help='Type of synthetic workload')
choices = ['layout1', 'layout2']
parser.add_argument('--layout', type=str, choices=choices, default=choices[0], help='Layout of UI')
args = parser.parse_args()
args_dict = vars(args)
print(args_dict)
