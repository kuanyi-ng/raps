# Cinea nearly 32 theoretical peak petaFLOPS of computing power,
# https://newsroom.ibm.com/2020-07-07-CINECA-Chooses-IBM-POWER9-with-NVIDIA-GPUs-and-InfiniBand-Network-for-Marconi100-Accelerated-Cluster
import numpy as np
from .utils import linear_to_3d_index

#from .config import initialize_config, load_config_variables
#initialize_config(args.system)
#load_config_variables(['CPUS_PER_NODE', 'GPUS_PER_NODE'], globals())

# Frontier
PRECISION = 64
CPU_CORES = 64
FLOPS_PER_GPU = 52
FLOPS_PER_CPU_CORE_PER_CYCLE = 16
# See https://docs.lumi-supercomputer.eu/hardware/lumig/
# The cores of this CPU are "Zen 3" compute cores supporting AVX2 256-bit vector instructions 
# for a maximum throughput of 16 double precision FLOP/clock (AVX2 FMA operations). 
GHZ = 2E9

cpu_peak_flops = CPU_CORES * FLOPS_PER_CPU_CORE_PER_CYCLE * GHZ
print(cpu_peak_flops)

gpu_peak_flops = 52E12

node_peak_flops = cpu_peak_flops + 4 * gpu_peak_flops

print(f"{node_peak_flops:.2e}")

system_peak_flops = 74 * 128 * node_peak_flops

print(f"{system_peak_flops:.2e}")


def compute_node_flops(cpu_util, gpu_util):
    return cpu_util * cpu_peak_flops + gpu_util * gpu_peak_flops


class FLOPSManager():

    def __init__(self, sc_shape):
        self.sc_shape = sc_shape
        self.flop_state = np.zeros(sc_shape)

    def update_flop_state(self, scheduled_nodes, cpu_util, gpu_util):
        node_indices = linear_to_3d_index(scheduled_nodes, self.sc_shape)
        self.flop_state[node_indices] = compute_node_flops(cpu_util, gpu_util)

    def get_system_performance(self):
        return np.sum(self.flop_state)

if __name__ == "__main__":
    fm = FLOPManager((25, 3, 128))
    print(fm.flop_state.shape)
