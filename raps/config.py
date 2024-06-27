import json
import os
from typing import Dict, Any
from pathlib import Path

CONFIG_PATH = Path(os.environ.get("RAPS_CONFIG", 'config')).resolve()

class ConfigManager:
    def __init__(self, system_name: str):
        self.config: Dict[str, Any] = {}
        self.load_system_config(system_name)

    def load_system_config(self, system_name: str) -> None:
        base_path = CONFIG_PATH / system_name
        config_files = ['system.json', 'power.json', 'cooling.json', 'scheduler.json']
        
        for config_file in config_files:
            config_data = self.load_config_file(base_path / config_file)
            self.config.update(config_data)
        
    @staticmethod
    def load_config_file(file_path: Path) -> dict[str, Any]:
        with open(file_path, 'r') as file:
            return json.load(file)

    def get(self, key: str) -> Any:
        return self.config.get(key)

config_manager = None # Placeholder for global ConfigManager instance

def initialize_config(system_name: str) -> None:
    global config_manager
    config_manager = ConfigManager(system_name=system_name)

def is_config_initialized() -> bool:
    return config_manager is not None

def get_config() -> ConfigManager:
    return config_manager

def load_config_variables(variable_names: list[str], namespace: dict[str, Any]) -> None:
    namespace.update({var: config_manager.get(var) for var in variable_names})
