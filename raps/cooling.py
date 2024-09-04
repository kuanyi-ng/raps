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

        # Dynamically determine the FMU Output Keys
        self.fmu_output_keys = self.generate_fmu_output_keys()
        
        # Instantiate and initialize the FMU
        self.fmu = FMU2Slave(guid=model_description.guid,
                             unzipDirectory=self.unzipdir,
                             modelIdentifier=model_description.coSimulation.modelIdentifier,
                             instanceName='instance1')
        self.fmu.instantiate()
        self.fmu.setupExperiment(startTime=0.0)
        self.fmu.enterInitializationMode()
        self.fmu.exitInitializationMode()

    def generate_fmu_output_keys(self):
        """
        Generates the fmu output keys dynamically based on FMU's output variable names,
        preserving the order in which they appear.

        Returns
        -------
        output_keys : list of str
            A list of unique base names of the output variables in their order of appearance.
        """
        seen_keys = OrderedDict()
        for output in self.outputs:
            # Split the name at the first '[' and take the base part
            base_name = output.name.split('[')[0]
            if base_name not in seen_keys:
                seen_keys[base_name] = None
        
        # Return the keys as a list
        return list(seen_keys.keys())

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
                print(f"Target Datetime: {target_datetime}")

                # Get temperature
                temperature = self.weather.get_temperature(target_datetime)
        
                if temperature is not None:
                    print(f"The temperature on {target_datetime.strftime('%Y-%m-%d %H:%M')} for ZIP code {ZIP_CODE} was {temperature:.2f} K.")
                    runtime_values["simulator_1_centralEnergyPlant_1_coolingTowerLoop_1_sources_Towb"] = temperature
                    #breakpoint()
                else:
                    print("Failed to retrieve weather data.")
                    runtime_values["simulator_1_centralEnergyPlant_1_coolingTowerLoop_1_sources_Towb"] = WET_BULB_TEMP
                    #breakpoint()
            else:
                print("Failed to retrieve coordinates.")
                runtime_values["simulator_1_centralEnergyPlant_1_coolingTowerLoop_1_sources_Towb"] = WET_BULB_TEMP
                #breakpoint()

        # Otherwise just use constant temp from config
        else:
            print('SIMULATED MODE')
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

        val_outputs = {}
        for v in self.outputs:
            val_outputs[v.name] = self.fmu.getReal([v.valueReference])[0]

        val_time = {'time': current_time}
        # Append the results
        rows_dict = merge_dicts(merge_dicts(val_time, val_inputs), val_outputs)
        self.fmu_history.append(rows_dict)
        data_array = self.convert_dict_to_array(val_outputs)
        self.current_result = data_array # Store the current fmu results for this timestep
        print(rows_dict)
        print('\n\n')
        #breakpoint()

        return data_array

    def convert_dict_to_array(self, data):
        """
        Converts the row dictionary data to a numpy array.

        Parameters
        ----------
        data : dict
            A dictionary containing the row data.

        Returns
        -------
        data_array : numpy.ndarray
            A numpy array with the extracted data values.
        """
        data_array = np.zeros((NUM_CDUS, len(self.fmu_output_keys)))

        keys_to_extract = [f'{base}[{i}]' for base in self.fmu_output_keys for i in range(1, NUM_CDUS + 1)]
        # Iterate through the keys in data and extract relevant values to fill the array
        for key, value in data.items():
            if key in keys_to_extract:
                # Extract the unit number from the key, e.g.:
                #('cdu_coolingsubsystem_0_liquidoutlet_0_
                # liquidflow_secondary[1]' -> 1)
                parts = key.split('[')
                base_key = parts[0]
                unit_number = int(parts[1].split(']')[0])

                # Adjust the unit_number to be 1-based (Python uses 0-based indexing)
                unit_number -= 1

                # Find the index of base_key in self.fmu_output_keys
                base_index = self.fmu_output_keys.index(base_key)

                # Fill the corresponding element in the array
                data_array[unit_number, base_index] = value
        return data_array

    def get_cooling_df(self):
        # Initialize the columns for cooling_df
        cooling_columns = self.fmu_output_keys

        # Generate cooling_df
        cooling_df = pd.DataFrame(self.current_result, columns=cooling_columns)

        return cooling_df

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
