import numpy as np
from .utils import linear_to_3d_index
import raps.telemetry as telemetry

class FLOPSManager():

    def __init__(self, **config):
        self.config = config
        self.flop_state = np.zeros(self.config['SC_SHAPE'])

    def update_flop_state(self, scheduled_nodes, cpu_util, gpu_util):
        node_indices = linear_to_3d_index(scheduled_nodes, self.config['SC_SHAPE'])
        validate = telemetry.telemetry_args.get('validate')
        if validate:   # cpu_util is in fact node_Watts in this case
            self.flop_state[node_indices] = \
                (self.config['CPU_FP_RATIO']*self.config['CPU_PEAK_FLOPS'] + self.config['GPU_FP_RATIO'] * self.config['GPU_PEAK_FLOPS']) * (cpu_util / (self.config['POWER_CPU_MAX']*self.config['CPUS_PER_NODE'] + self.config['POWER_GPU_MAX']*self.config['GPUS_PER_NODE']+ self.config['POWER_NIC']*self.config['NICS_PER_NODE']+self.config['POWER_NVME']))
        else:   
            self.flop_state[node_indices] = \
                self.config['CPU_FP_RATIO'] * cpu_util * self.config['CPU_PEAK_FLOPS'] + \
                self.config['GPU_FP_RATIO'] * gpu_util * self.config['GPU_PEAK_FLOPS']



    def get_rpeak(self):
        node_peak_flops = self.config['CPUS_PER_NODE'] * self.config['CPU_PEAK_FLOPS'] \
                        + self.config['GPUS_PER_NODE'] * self.config['GPU_PEAK_FLOPS']
        system_peak_flops = self.config['AVAILABLE_NODES'] * node_peak_flops
        return system_peak_flops

    def get_system_performance(self):
        return np.sum(self.flop_state)
