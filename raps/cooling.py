"""
This module provides functionality for simulating a thermo-fluids model using 
an FMU (Functional Mock-up Unit).

The module defines a `ThermoFluidsModel` class that encapsulates the 
initialization, simulation step execution,
data conversion, and cleanup processes for the FMU-based model. Additionally, 
it includes a helper function to merge dictionaries.

Functions
---------
merge_dicts(dict1, dict2)
    Merge two dictionaries into one.

Classes
-------
ThermoFluidsModel
    A class to represent a thermo-fluids model using an FMU.
"""

import shutil
import re
import numpy as np
import pandas as pd
from uncertainties import unumpy

from fmpy import read_model_description, extract
from fmpy.fmi2 import FMU2Slave
from .config import load_config_variables
from collections import OrderedDict
from raps.weather import Weather
from datetime import datetime, timedelta

load_config_variables(['FMU_OUTPUT_KEYS','NUM_CDUS', 'COOLING_EFFICIENCY','WET_BULB_TEMP', 'RACKS_PER_CDU', 'ZIP_CODE', 'COUNTRY_CODE'], globals())

# Define the Merge function outside of the class
def merge_dicts(dict1, dict2):
    """
    Merge two dictionaries into one.

    Parameters
    ----------
    dict1 : dict
        The first dictionary to merge.
    dict2 : dict
        The second dictionary to merge. If there are duplicate keys, the values
        from this dictionary will overwrite those from the first dictionary.

    Returns
    -------
    merged_dict : dict
        A new dictionary containing all the keys and values from both input dictionaries.
        If there are duplicate keys, the values from `dict2` will overwrite those from `dict1`.
    """
    merged_dict = {**dict1, **dict2}
    return merged_dict

def get_matching_variables(variables, pattern):
    # Regex pattern to match strings containing .summary
    pattern = re.compile(pattern)

    # Filtering the list using the regex pattern
    filtered_vars = [var for var in variables if pattern.match(var)]
    
    return filtered_vars


class ThermoFluidsModel:
    """
    A class to represent a thermo-fluids model using an FMU (Functional Mock-up Unit).

    Attributes
    ----------
    FMU_PATH : str
        The file path to the FMU file.
    fmu_history : list
        A list to store the history of FMU states.
    inputs : list
        A list of input variables for the FMU.
    outputs : list
        A list of output variables for the FMU.
    unzipdir : str
        The directory where the FMU file is extracted.
    fmu : FMU2Slave
        The instantiated FMU object.

    Methods
    -------
    initialize():
        Initializes the FMU by extracting the file and setting up the model.
    step(current_time, fmu_inputs, step_size):
        Executes a simulation step with the given inputs and step size.
    convert_rowsdict_to_array(data):
        Converts the row dictionary data to a numpy array.
    terminate():
        Terminates the FMU instance.
    cleanup():
        Cleans up the extracted FMU directory.
    """

    def __init__(self, FMU_PATH, start=None):
        """
        Constructs all the necessary attributes for the ThermoFluidsModel object.

        Parameters
        ----------
        FMU_PATH : str
            The file path to the FMU file.
        """
        self.FMU_PATH = FMU_PATH
        self.fmu_history = []
        self.inputs = None
        self.outputs = None
        self.unzipdir = None
        self.fmu = None
        self.template = None
        self.fmu_output_keys = []
        self.current_result = None
        self.weather = None
    
    def initialize(self):
        """
        Initializes the FMU by extracting the file and setting up the model.

        This method unzips the FMU file, reads the model description,
        collects value references for input and output variables,
        and initializes the FMU for simulation.
        """
        # Notify user that FMU is initializing
        print('Initializing FMU...')

        # Unzip the FMU file and get the unzip directory
        self.unzipdir = extract(self.FMU_PATH)
        model_description = read_model_description(self.FMU_PATH)

        # Add to list of variable names
        var_model = []
        for variable in model_description.modelVariables:
            var_model.append(variable.name)

        outputs = get_matching_variables(var_model, r'.*(\.summary\.|^summary).*')

        # Get the value references for the variables we want to get/set
        self.inputs = [v for v in model_description.modelVariables if v.causality == 'input']
        self.outputs = [v for v in model_description.modelVariables if v.name in outputs]
        
        # Instantiate and initialize the FMU
        self.fmu = FMU2Slave(guid=model_description.guid,
                             unzipDirectory=self.unzipdir,
                             modelIdentifier=model_description.coSimulation.modelIdentifier,
                             instanceName='instance1')
        self.fmu.instantiate()
        self.fmu.setupExperiment(startTime=0.0)
        self.fmu.enterInitializationMode()
        self.fmu.exitInitializationMode()

    def generate_runtime_values(self, cdu_power, sc):
        """
        Generate the runtime values for the FMU inputs dynamically.
    
        Parameters:
        cdu_power (array): The array of CDU powers.
        wetbulb_temp (float): The wetbulb temperature.
    
        Returns:
        dict: A dictionary with the runtime values for the FMU inputs.
        """
        runtime_values = {}

        # Dynamically generate the power inputs
        for i in range(NUM_CDUS):
           key = f"simulator_1_datacenter_1_computeBlock_{i+1}_cabinet_1_sources_Q_flow_total"
           runtime_values[key] = cdu_power[i] * COOLING_EFFICIENCY / RACKS_PER_CDU

        # If replay get temperature based on datetime and location
        if sc.replay:
            if self.weather.has_coords:
                # Convert total seconds to timedelta object
                delta = timedelta(seconds=sc.current_time)

                # Extract hours, minutes, and seconds from timedelta
                hours, remainder = divmod(delta.seconds, 3600)  # Get hours and the remaining seconds
                minutes, seconds = divmod(remainder, 60)  # Get minutes and the remaining seconds

                # Initialize target_datetime using the calculated hours, minutes, and seconds
                target_datetime = datetime(2024, 4, 7, hours, minutes, seconds)  # YYYY-MM-DD HH:MM:SS format
                #print(f"Target Datetime: {target_datetime}")

                # Get temperature
                temperature = self.weather.get_temperature(target_datetime)
        
                if temperature is not None:
                    #print(f"The temperature on {target_datetime.strftime('%Y-%m-%d %H:%M')} for ZIP code {ZIP_CODE} was {temperature:.2f} K.")
                    runtime_values["simulator_1_centralEnergyPlant_1_coolingTowerLoop_1_sources_Towb"] = temperature
                    #breakpoint()
                else:
                    #print("Failed to retrieve weather data.")
                    runtime_values["simulator_1_centralEnergyPlant_1_coolingTowerLoop_1_sources_Towb"] = WET_BULB_TEMP
                    #breakpoint()
            else:
                #print("Failed to retrieve coordinates.")
                runtime_values["simulator_1_centralEnergyPlant_1_coolingTowerLoop_1_sources_Towb"] = WET_BULB_TEMP
                #breakpoint()

        # Otherwise just use constant temp from config
        else:
            #print('SIMULATED MODE')
            #breakpoint()
            runtime_values["simulator_1_centralEnergyPlant_1_coolingTowerLoop_1_sources_Towb"] = WET_BULB_TEMP

        return runtime_values
    
    def generate_fmu_inputs(self, runtime_values, uncertainties=False):
        """
        Convert the runtime values based on the cooling model's inputs to a list suitable for FMU inputs.
        Raises an error if any input key is missing in runtime values.
        """
        # Initialize an empty list for FMU inputs
        fmu_inputs = []

        # Iterate through the cooling model's inputs
        for input_var in self.inputs:
            input_name = input_var.name  # Get the name of the input variable
            # Check if the input name matches any key in the runtime values
            if input_name in runtime_values:
                # Append the value from runtime values to fmu_inputs
                if uncertainties:
                    # Strip only the power values of the uncertainty, others should not be a ufloat
                    # #Alternative uncomment line below and remove pattern match:
                    # #fmu_inputs.append(unumpy.nominal_values(runtime_values[input_name]))
                    pattern = re.compile(r"power", re.IGNORECASE)
                    if bool(pattern.search(input_name)):
                        fmu_inputs.append(unumpy.nominal_values(runtime_values[input_name]))
                    else:
                        fmu_inputs.append(runtime_values[input_name])
                else:
                    fmu_inputs.append(runtime_values[input_name])
            else:
                # If you have additional values that the fmu isn't expecting
                # nothing will happen. However, an error will be raised
                # if a value for an expected key is missing in runtime values
                raise KeyError(f"Missing value for key '{input_name}' in runtime values.")

        return fmu_inputs

    def calculate_pue(self, cooling_input, datacenter_output, cep_output):
        # Convert values from kW to Watts
        W_HTWPs = np.array(cep_output['simulator[1].centralEnergyPlant[1].hotWaterLoop[1].summary.W_flow_HTWP_kW']) * 1e3
        W_CTWPs = np.array(cep_output['simulator[1].centralEnergyPlant[1].coolingTowerLoop[1].summary.W_flow_CTWP_kW']) * 1e3
        W_CTs = np.array(cep_output['simulator[1].centralEnergyPlant[1].coolingTowerLoop[1].summary.W_flow_CT_kW']) * 1e3

        # Initialize W_CDUPs as zero array of the same shape as datacenter output
        W_CDUPs = np.zeros_like(W_HTWPs)

        # Loop over all compute blocks (CDUs)
        for idx in range(NUM_CDUS):
            colName = f'simulator[1].datacenter[1].computeBlock[{idx+1}].cdu[1].summary.W_flow_CDUP_kW'
            # Accumulate the power values for all CDUs
            W_CDUPs += np.array(datacenter_output[colName]) * 1e3

        # Sum all values in the cooling_input dictionary
        total_cooling_input_power = np.sum(list(cooling_input.values()))

        # Ensure a non-zero value for total input power to avoid division by zero
        total_input_power = np.maximum(total_cooling_input_power, 1e-3)

        # Calculate PUE
        pue = (total_input_power + np.sum(W_CDUPs) + np.sum(W_HTWPs) + np.sum(W_CTWPs) + np.sum(W_CTs)) / total_input_power
        
        return pue
    
    def step(self, current_time, fmu_inputs, step_size):
        """
        Executes a simulation step with the given inputs and step size.

        Parameters
        ----------
        current_time : float
            The current simulation time.
        fmu_inputs : list
            A list of input values to set in the FMU.
        step_size : float
            The size of the simulation step.

        Returns
        -------
        data_array : numpy.ndarray
            A numpy array containing the simulation results for the current step.
        """
        # Simulation Loop
        for index, v in enumerate(self.inputs):
            self.fmu.setReal([v.valueReference], [fmu_inputs[index]])

        # Perform one step
        self.fmu.doStep(currentCommunicationPoint=current_time, communicationStepSize=step_size)

        # Get the values for 'inputs' and 'outputs'
        val_inputs = {}
        for v in self.inputs:
            val_inputs[v.name] = self.fmu.getReal([v.valueReference])[0]

        val_outputs_datacenter = {}
        val_outputs_cep = {}
        for v in self.outputs:
            if "datacenter" in v.name:
                val_outputs_datacenter[v.name] = self.fmu.getReal([v.valueReference])[0]

            if "centralEnergyPlant" in v.name:
                val_outputs_cep[v.name] = self.fmu.getReal([v.valueReference])[0]

        val_time = {'time': current_time}
        
        # Append the results
        cooling_input = val_inputs
        datacenter_output = val_outputs_datacenter
        cep_output = val_outputs_cep
        self.fmu_history.append(merge_dicts(merge_dicts(val_time, val_inputs), merge_dicts(datacenter_output, cep_output)))
        pue = self.calculate_pue(cooling_input, datacenter_output, cep_output)

        return cooling_input, datacenter_output, cep_output, pue

    def terminate(self):
        """
        Terminates the FMU instance.

        This method properly terminates the FMU instance, ensuring that all
        resources are released.
        """
        # Close the FMU
        self.fmu.terminate()
        self.fmu.freeInstance()

    def cleanup(self):
        """
        Cleans up the extracted FMU directory.

        This method removes the directory where the FMU file was extracted,
        ensuring no temporary files are left behind.
        """
        # Cleanup - at the end of the simulation
        shutil.rmtree(self.unzipdir, ignore_errors=True)
