"""
Module Description:
This module contains functions and classes related to workload management and power calculation.

Classes:
- PowerManager: Manages power consumption and loss calculations in the system.

Functions:
- sivoc_loss: Calculate the power input required considering Sivoc power loss.
- rectifier_loss: Calculate the power input required considering Rectifier power loss.
- compute_node_power: Calculate the total power consumption for given CPU and GPU utilization.
- compute_node_power_validate: Calculate the total power consumption for a given mean and standard deviation of node power.
"""

import numpy as np
import pandas as pd
import uncertainties as uf
from .utils import linear_to_3d_index
from .config import load_config_variables


load_config_variables([
    'AVAILABLE_NODES',
    'BLADES_PER_CHASSIS',
    'CHASSIS_PER_RACK',
    'NUM_CDUS',
    'NUM_RACKS',
    'POWER_CDUS',
    'POWER_GPU_IDLE',
    'POWER_GPU_MAX',
    'POWER_GPU_UNCERTAINTY',
    'CPUS_PER_NODE',
    'GPUS_PER_NODE',
    'NICS_PER_NODE',
    'POWER_CPU_IDLE',
    'POWER_CPU_MAX',
    'POWER_CPU_UNCERTAINTY',
    'POWER_MEM',
    'POWER_MEM_UNCERTAINTY',
    'POWER_NIC',
    'POWER_NIC_UNCERTAINTY',
    'POWER_NVME',
    'POWER_NVME_UNCERTAINTY',
    'POWER_SWITCH',
    'POWER_SWITCH_UNCERTAINTY',
    'RACKS_PER_CDU',
    'SIVOC_LOSS_CONSTANT',
    'SIVOC_EFFICIENCY',
    'NODES_PER_BLADE',
    'NODES_PER_RACK',
    'NODES_PER_RECTIFIER',
    'NODE_POWER_UNCERTAINTY',
    'RECTIFIER_PEAK_THRESHOLD',
    'RECTIFIERS_PER_CHASSIS',
    'RECTIFIER_LOSS_CONSTANT',
    'RECTIFIER_EFFICIENCY',
    'SWITCHES_PER_CHASSIS',
    'POWER_DF_HEADER',
    'MISSING_RACKS'
], globals())


def custom_str_uncertainties(self):
    return f"{self.nominal_value} ± {self.std_dev}"


def custom_repr_uncertainties(self):
    return f"{self.nominal_value}+/-{self.std_dev}"


def custom_format_uncertainties(self, fmt_spec):
    return f"{self.nominal_value:{fmt_spec}} ±{self.std_dev:{fmt_spec}}"


#In stats unicode is printed as unocde abbreviation! To be fixed!
uf.Variable.__str__ = custom_str_uncertainties
uf.Variable.__repr__ = custom_repr_uncertainties
uf.Variable.__format__ = custom_format_uncertainties


def sivoc_loss(p_out):
    """Calculate the power input required considering Sivoc power loss."""
    p_in = (p_out + SIVOC_LOSS_CONSTANT) / SIVOC_EFFICIENCY
    return p_in


def rectifier_loss(p_out):
    """Calculate the power input required considering Rectifier power loss."""
    p_in = (p_out + RECTIFIER_LOSS_CONSTANT) / RECTIFIER_EFFICIENCY
    return p_in


def compute_node_power(cpu_util, gpu_util, verbose=False):
    """
    Calculate the total power consumption for given CPU and GPU utilization.

    :param cpu_util: The utilization of the CPU.
    :param gpu_util: The utilization of the GPU.
    :param verbose: Flag for verbose output.
    :return: Total power consumption after accounting for power loss.
    """
    power_cpu = cpu_util * POWER_CPU_MAX + (CPUS_PER_NODE - cpu_util) * POWER_CPU_IDLE
    power_gpu = gpu_util * POWER_GPU_MAX + (GPUS_PER_NODE - gpu_util) * POWER_GPU_IDLE

    power_total = power_cpu + power_gpu + POWER_MEM + NICS_PER_NODE * POWER_NIC + POWER_NVME

    # Apply power loss due to Sivoc and Rectifier
    power_with_sivoc_loss = sivoc_loss(power_total)
    power_sivoc_loss_only = power_with_sivoc_loss - power_total

    if verbose:
        print(f"*** Power (CPU + GPU + MEM + NICS): {power_total}")
        print(f"*** SIVOC loss: {power_sivoc_loss_only}")
        print(f"*** Power before SIVOC loss: {power_with_sivoc_loss}")

    return power_with_sivoc_loss, power_sivoc_loss_only


def compute_node_power_uncertainties(cpu_util, gpu_util, verbose=False):
    """
    Calculate the total power consumption for given CPU and GPU utilization.

    :param cpu_util: The utilization of the CPU.
    :param gpu_util: The utilization of the GPU.
    :param verbose: Flag for verbose output.
    :return: Total power consumption after accounting for power loss.
    """
    power_cpu = cpu_util \
                * uf.ufloat(POWER_CPU_MAX, POWER_CPU_MAX * POWER_CPU_UNCERTAINTY) \
                + (CPUS_PER_NODE - cpu_util) \
                * uf.ufloat(POWER_CPU_IDLE, POWER_CPU_IDLE * POWER_CPU_UNCERTAINTY)
    power_gpu = gpu_util \
                * uf.ufloat(POWER_GPU_MAX, POWER_GPU_MAX * POWER_GPU_UNCERTAINTY) \
                + (GPUS_PER_NODE - gpu_util) \
                * uf.ufloat(POWER_GPU_IDLE, POWER_GPU_IDLE * POWER_GPU_UNCERTAINTY)

    power_total = power_cpu + power_gpu \
                  + uf.ufloat(POWER_MEM, POWER_MEM * POWER_MEM_UNCERTAINTY) \
                  + uf.ufloat(POWER_NIC, POWER_NIC * POWER_NIC_UNCERTAINTY) \
                  + uf.ufloat(POWER_NVME, POWER_NVME * POWER_NVME_UNCERTAINTY)

    # Apply power loss due to Sivoc and Rectifier
    power_with_sivoc_loss = sivoc_loss(power_total)
    power_sivoc_loss_only = power_with_sivoc_loss - power_total

    if verbose:
        print(f"*** Power (CPU + GPU + MEM + NICS): {power_total}")
        print(f"*** SIVOC loss: {power_sivoc_loss_only}")
        print(f"*** Power before SIVOC loss: {power_with_sivoc_loss}")

    return power_with_sivoc_loss, power_sivoc_loss_only


def compute_node_power_validate(mean_node_power, stddev_node_power, verbose=False):
    """
    Calculate the total power consumption for given mean and standard deviation of node power.

    Parameters:
    - mean_node_power: float
        Mean node power consumption.
    - stddev_node_power: float
        Standard deviation of node power consumption.
    - verbose: bool, optional
        Flag for verbose output. Default is False.

    Returns:
    tuple
        Total power consumption after accounting for power loss and Sivoc loss.
    """
    power_total = mean_node_power
    power_with_sivoc_loss = sivoc_loss(power_total)
    power_sivoc_loss_only = power_with_sivoc_loss - power_total
    if verbose:
        print(f"*** Power (CPU + GPU + MEM + NICS): {power_total}")
        print(f"*** SIVOC loss: {power_sivoc_loss_only}")
        print(f"*** Power before SIVOC loss: {power_with_sivoc_loss}")
    return power_with_sivoc_loss, power_sivoc_loss_only


def compute_node_power_validate_uncertainties(mean_node_power, stddev_node_power, verbose=False):
    """
    Calculate the total power consumption for given mean and standard deviation of node power.

    Parameters:
    - mean_node_power: float
        Mean node power consumption.
    - stddev_node_power: float
        Standard deviation of node power consumption.
    - verbose: bool, optional
        Flag for verbose output. Default is False.

    Returns:
    tuple
        Total power consumption after accounting for power loss and Sivoc loss.
    """
    power_total = uf.ufloat(mean_node_power, mean_node_power * POWER_NODE_UNCERTAINTY)
    power_with_sivoc_loss = sivoc_loss(power_total)
    power_sivoc_loss_only = power_with_sivoc_loss - power_total
    if verbose:
        print(f"*** Power (CPU + GPU + MEM + NICS): {power_total}")
        print(f"*** SIVOC loss: {power_sivoc_loss_only}")
        print(f"*** Power before SIVOC loss: {power_with_sivoc_loss}")
    return power_with_sivoc_loss, power_sivoc_loss_only


class PowerManager:
    """
    Class Description:
    Manages power consumption and loss calculations in the system.

    Attributes:
    - sc_shape: Shape of the system configuration.
    - power_func: Function for calculating power consumption.
    - power_state: Current power state of the system.
    - rectifier_loss: Loss due to rectifier inefficiency.
    - sivoc_loss: Loss due to Sivoc inefficiency.
    - history: History of power states.
    - loss_history: History of power losses.
    - down_nodes: Nodes that are currently down.
    - down_rack: Rack number of down nodes.
    """
    def __init__(self, sc_shape, down_nodes, power_func=compute_node_power):
        """
        Initialize the PowerManager object.

        Parameters:
        - sc_shape: tuple
            Shape of the system configuration.
        - down_nodes: list
            Nodes that are currently down.
        - power_func: function, optional
            Function for calculating power consumption. Default is compute_node_power.
        """
        self.sc_shape = sc_shape
        self.power_func = power_func
        self.power_state = self.initialize_power_state()
        self.rectifier_loss = self.initialize_rectifier_loss()
        self.sivoc_loss = self.initialize_sivoc_loss()
        self.history = []
        self.loss_history = []
        self.down_nodes = down_nodes
        self.uncertainties = False
        if power_func in [compute_node_power_uncertainties, \
                          compute_node_power_validate_uncertainties]:
            self.uncertainties = True
        if self.down_nodes: self.apply_down_nodes()

    def get_peak_power(self):
        """Estimate peak power of system for setting max value of gauges in dashboard"""
        node_power = compute_node_power(CPUS_PER_NODE, GPUS_PER_NODE)[0]
        blades_per_rectifier = BLADES_PER_CHASSIS / RECTIFIERS_PER_CHASSIS
        rectifier_load = blades_per_rectifier * NODES_PER_BLADE * node_power
        rectifier_power = rectifier_loss(rectifier_load) # with AC-DC conversion losses
        chassis_power = BLADES_PER_CHASSIS * rectifier_power / blades_per_rectifier \
                      + SWITCHES_PER_CHASSIS * POWER_SWITCH
        rack_power = chassis_power * CHASSIS_PER_RACK 
        total_power = rack_power * NUM_RACKS + POWER_CDUS
        return total_power

    def initialize_power_state(self):
        """Initialize the power state array with idle power consumption values."""
        initial_power, _ = self.power_func(0, 0)
        return np.full(self.sc_shape, initial_power)

    def initialize_sivoc_loss(self):
        """Initialize the Sivoc loss array with idle power consumption values."""
        _, initial_sivoc_loss = self.power_func(0, 0)
        return np.full(self.sc_shape, initial_sivoc_loss)

    def initialize_rectifier_loss(self):
        """ Initialize the power state array """
        initial_power, _ = self.power_func(0, 0)
        # Rectifier loss curvefit is done at rectifier level, so we simply
        # approximate by scaling up to number of rectifiers, applying loss
        # and then dividing by number of rectifiers.
        # For Frontier there are four nodes per rectifier.
        power_with_loss = rectifier_loss(initial_power * NODES_PER_RECTIFIER) \
                          / NODES_PER_RECTIFIER
        return np.full(self.sc_shape, power_with_loss)

    def apply_down_nodes(self):
        """ Apply the down nodes to the power state, setting their power to zero """
        down_indices = linear_to_3d_index(self.down_nodes, self.sc_shape)
        self.power_state[down_indices] = 0
        self.rectifier_loss[down_indices] = 0
        self.sivoc_loss[down_indices] = 0

    def set_idle(self, node_indices):
        """
        Set the power consumption of specified nodes to idle.

        Parameters:
        - node_indices: list
            Indices of the nodes to set to idle.
        """
        node_indices = linear_to_3d_index(node_indices, self.sc_shape)
        self.power_state[node_indices], self.sivoc_loss[node_indices] \
            = compute_node_power(0, 0)

    def update_power_state(self, scheduled_nodes, cpu_util, gpu_util):
        """
        Update the power state of scheduled nodes based on CPU and GPU utilization.
        Note: this is only used to test smart load-sharing "what-if" scenario

        Parameters:
        - scheduled_nodes: list
            Indices of the scheduled nodes.
        - cpu_util: float
            CPU utilization.
        - gpu_util: float
            GPU utilization.

        Returns:
        float
            Total power consumption of the scheduled nodes.
        """
        node_indices = linear_to_3d_index(scheduled_nodes, self.sc_shape)
        power_value, sivoc_loss = self.power_func(cpu_util, gpu_util)
        self.power_state[node_indices] = power_value
        self.sivoc_loss[node_indices] = sivoc_loss
        return power_value * len(scheduled_nodes)

    def calculate_rectifiers_needed(self, power_state_summed):
        """
        Calculate the number of rectifiers needed based on the total power consumption.

        Parameters:
        - power_state_summed: float
            Summed power consumption.

        Returns:
        int
            Number of rectifiers needed.
        """
        value = int((power_state_summed - 1) // RECTIFIER_PEAK_THRESHOLD + 1)
        return min(value, RECTIFIERS_PER_CHASSIS)

    def compute_rack_power(self, smart_load_sharing=False):
        """
        Compute the power consumption of each rack in the system.

        Parameters:
        - smart_load_sharing: bool, optional
            Flag for enabling smart load sharing. Default is False.

        Returns:
        tuple
            Tuple containing rack power and rectifier losses.
        """
        shape = (self.sc_shape[0], self.sc_shape[1], CHASSIS_PER_RACK, -1)
        power_state_reshaped = np.reshape(self.power_state, shape)
        chassis_power = np.sum(power_state_reshaped, axis=-1)

        # Add in switch power
        chassis_power += SWITCHES_PER_CHASSIS * POWER_SWITCH

        # Divide the power by the number of rectifiers and apply losses per rectifier
        # Smart load sharing dynamically stages rectifers as needed, e.g., when
        # all nodes are idle, only a single rectifier is used. When all
        # nodes are fully utilized, four rectifiers are used, and in between.
        if smart_load_sharing:
            vectorized_function = np.vectorize(self.calculate_rectifiers_needed)
            num_rectifiers_array = vectorized_function(chassis_power)

            # Initialize the array to hold the divided powers, using NaN for unused elements
            rectifier_power = np.full((*chassis_power.shape, RECTIFIERS_PER_CHASSIS), np.nan)
            power_with_losses = np.copy(rectifier_power)

            # Chassis_power.shape for Frontier is (25, 3, 8)
            for i in range(chassis_power.shape[0]):
                for j in range(chassis_power.shape[1]):
                    for k in range(chassis_power.shape[2]):
                        num_rectifiers = num_rectifiers_array[i, j, k]
                        power_per_rectifier = chassis_power[i, j, k] / num_rectifiers
                        rectifier_power[i, j, k, :num_rectifiers] = power_per_rectifier
                        power_with_losses[i, j, k, :num_rectifiers] = rectifier_loss(power_per_rectifier)

            rectifier_power = np.nan_to_num(rectifier_power)
            power_with_losses = np.nan_to_num(power_with_losses)

        else:
            divisor = np.array([4, 4, 4, 4]).reshape(1, 1, 1, 4)
            rectifier_power = chassis_power[:, :, :, np.newaxis] / divisor
            power_with_losses = rectifier_loss(rectifier_power)

        # Compute just the losses
        rect_losses = power_with_losses - rectifier_power

        # Sum to 75 racks
        summed_power_with_losses = np.sum(power_with_losses/1000, axis=(2, 3))
        # Zero out power for missing racks
        for rack in MISSING_RACKS:
            cdu = rack // RACKS_PER_CDU
            rack2d = (cdu, rack % RACKS_PER_CDU)
            summed_power_with_losses[rack2d] = 0
        summed_rect_losses = np.sum(rect_losses/1000, axis=(2, 3))

        # Add CDU numbers to table
        rows = self.sc_shape[0]
        row_numbers = np.arange(1, rows + 1).reshape(-1, 1)

        # Calculate the sum of racks 1, 2, and 3 for each row
        power_with_rows = np.hstack((row_numbers, summed_power_with_losses))
        rack_power_sum = power_with_rows[:, 1:].sum(axis=1).reshape(-1, 1)
        power_with_rows = np.hstack((power_with_rows, rack_power_sum))

        rect_loss_with_rows = np.hstack((row_numbers, summed_rect_losses))
        rack_rect_loss_sum = rect_loss_with_rows[:, 1:].sum(axis=1).reshape(-1, 1)
        rect_loss_with_rows = np.hstack((rect_loss_with_rows, rack_rect_loss_sum))

        # Return rectifier losses summed at CDU level
        return power_with_rows, rect_loss_with_rows


    def compute_sivoc_losses(self):
        """
        Compute SIVOC losses for each CDU in the system.

        Returns:
        np.ndarray
            Array containing SIVOC losses for each CDU.
        """
        # Aggregate SIVOC losses
        summed_sivoc_losses = np.sum(self.sivoc_loss/1000, axis=2)  # kW
        rows = self.sc_shape[0]

        # Add CDU numbers to table
        row_numbers = np.arange(1, rows + 1).reshape(-1, 1)
        sivoc_loss_with_rows = np.hstack((row_numbers, summed_sivoc_losses))

        # Calculate the sum of racks 1, 2, and 3 for each row/CDU
        rack_sivoc_loss_sum = sivoc_loss_with_rows[:, 1:].sum(axis=1).reshape(-1, 1)
        sivoc_loss_with_rows = np.hstack((sivoc_loss_with_rows, rack_sivoc_loss_sum))

        return sivoc_loss_with_rows
    
    def get_power_df(self, rack_power, rack_loss):
        # Initialize the columns for power_df
        power_columns = POWER_DF_HEADER
        power_data = []

        # Generate power_df
        for i, (row_pow, row_loss) in enumerate(zip(rack_power, rack_loss)):
            # Include only the required values from power row and loss row
            power_data.append((
                str(i + 1),    # CDU Number
                *row_pow[1:],  # Skip the first element of the power row (First col is CDU Number)
                *row_loss[1:]  # Skip the first element of the loss row
            ))

        power_df = pd.DataFrame(power_data, columns=power_columns)

        return power_df
