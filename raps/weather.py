import requests
import urllib3
import json
from datetime import datetime, timedelta

# Disable SSL warnings when verify=False is used (temporary debugging purpose)
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
from .config import load_config_variables
load_config_variables(['ZIP_CODE', 'COUNTRY_CODE'], globals())

class Weather:
    def __init__(self):
        """
        Initialize the Weather class with configuration loaded from a JSON file.
        If zip_code and country_code are provided, the coordinates (lat, lon)
        will be retrieved once and stored.
        """
        self.zip_code = ZIP_CODE
        self.country_code = COUNTRY_CODE
        self.lat = None
        self.lon = None
        self.weather_cache = {}  # Cache for storing weather data for the entire day
        self.has_coords = False
        
        # Retrieve coordinates if zip_code and country_code are provided
        if self.zip_code and self.country_code:
            self.lat, self.lon = self.get_coordinates()
            if self.lat is None or self.lon is None:
                print("Warning: Unable to retrieve coordinates. Please check the zip code and country code.")
            else:
                self.has_coords = True
                self.retrieve_weather_data_for_day(datetime.now().date())  # Pre-fetch weather data for the current day
        else:
            print("Warning: zip_code and country_code are not specified. Coordinates will be None.")

    def get_coordinates(self):
        """
        Retrieve coordinates for a given ZIP code using Nominatim.
        Returns:
        tuple: (lat, lon) if found, otherwise (None, None).
        """
        if not self.zip_code or not self.country_code:
            print("Error: ZIP code or country code is not specified.")
            return None, None
        
        geocoding_url = f'https://nominatim.openstreetmap.org/search?postalcode={self.zip_code}&country={self.country_code}&format=json'
        headers = {
            'User-Agent': 'ExaDigiT'  # Custom User-Agent header
        }
        response = requests.get(geocoding_url, headers=headers, verify=False)  # Disable SSL verification temporarily
        
        # Check for successful response
        if response.status_code == 200:
            try:
                data = response.json()  # Attempt to parse the JSON response
                if len(data) > 0:
                    return float(data[0]['lat']), float(data[0]['lon'])
                else:
                    print("No data found for the provided ZIP code.")
                    return None, None
            except requests.exceptions.JSONDecodeError:
                print("Error: Response is not in JSON format.")
                return None, None
        else:
            print(f"Error fetching coordinates. Status Code: {response.status_code}")
            return None, None

    def retrieve_weather_data_for_day(self, date):
        """
        Retrieve all weather data for a specific date and cache it.
        """
        if self.lat is None or self.lon is None:
            print("Error: Latitude and longitude are not set. Please provide valid ZIP code and country code.")
            return
        
        weather_url = f'https://archive-api.open-meteo.com/v1/archive?latitude={self.lat}&longitude={self.lon}&start_date={date}&end_date={date}&temperature_unit=celsius&hourly=temperature_2m'
        response = requests.get(weather_url, verify=False)  # Disable SSL verification temporarily
        
        # Check for successful response
        if response.status_code == 200:
            try:
                data = response.json()  # Attempt to parse the JSON response
                if 'hourly' in data and 'temperature_2m' in data['hourly']:
                    times = data['hourly']['time']
                    temperatures = data['hourly']['temperature_2m']
                    
                    # Cache the weather data for fast lookup
                    for i, time in enumerate(times):
                        temp_celsius = temperatures[i]
                        if temp_celsius is not None:  # Check if temperature data is valid
                            self.weather_cache[time] = temp_celsius + 273.15  # Convert to Kelvin and store
                        else:
                            print(f"Warning: Missing temperature data for {time}. Skipping entry.")
                else:
                    print("Error fetching weather data.")
            except requests.exceptions.JSONDecodeError:
                print("Error: Response is not in JSON format.")
        else:
            print(f"Error fetching weather data. Status Code: {response.status_code}")


    def get_temperature(self, target_datetime):
        """
        Get temperature for a specific datetime from cached data.
        """
        if not self.has_coords:
            print("Error: Latitude and longitude are not set. Please provide valid ZIP code and country code.")
            return None
        
        # Round target_datetime to the nearest previous hour
        target_hour = target_datetime.replace(minute=0, second=0, microsecond=0)
        target_hour_str = target_hour.isoformat(timespec='minutes')  # Format to 'YYYY-MM-DDTHH:MM'
        
        # Retrieve from cache
        if target_hour_str in self.weather_cache:
            return self.weather_cache[target_hour_str]
        else:
            # If not cached, retrieve weather data for the day and retry
            self.retrieve_weather_data_for_day(target_datetime.date())
            return self.weather_cache.get(target_hour_str, None)
