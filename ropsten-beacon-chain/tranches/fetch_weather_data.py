
import requests
import json
import os
from datetime import datetime

def get_current_weather(city_name, api_key=None):
    """
    Fetch current weather data for a given city using OpenWeatherMap API.
    
    Args:
        city_name (str): Name of the city to get weather for
        api_key (str, optional): OpenWeatherMap API key. If not provided,
                                will try to get from environment variable.
    
    Returns:
        dict: Weather data including temperature, humidity, description, etc.
    
    Raises:
        ValueError: If API key is not provided and not found in environment
        requests.exceptions.RequestException: If API request fails
    """
    if api_key is None:
        api_key = os.environ.get('OPENWEATHER_API_KEY')
        if not api_key:
            raise ValueError(
                "API key must be provided either as argument "
                "or as OPENWEATHER_API_KEY environment variable"
            )
    
    base_url = "http://api.openweathermap.org/data/2.5/weather"
    
    params = {
        'q': city_name,
        'appid': api_key,
        'units': 'metric'  # Use metric units (Celsius)
    }
    
    try:
        response = requests.get(base_url, params=params, timeout=10)
        response.raise_for_status()
        
        data = response.json()
        
        if data.get('cod') != 200:
            error_message = data.get('message', 'Unknown error')
            raise requests.exceptions.HTTPError(f"API Error: {error_message}")
        
        return {
            'city': data['name'],
            'country': data['sys']['country'],
            'temperature': data['main']['temp'],
            'feels_like': data['main']['feels_like'],
            'humidity': data['main']['humidity'],
            'pressure': data['main']['pressure'],
            'weather_description': data['weather'][0]['description'],
            'weather_main': data['weather'][0]['main'],
            'wind_speed': data['wind']['speed'],
            'wind_direction': data['wind'].get('deg', 'N/A'),
            'cloudiness': data['clouds']['all'],
            'visibility': data.get('visibility', 'N/A'),
            'sunrise': datetime.fromtimestamp(data['sys']['sunrise']).strftime('%H:%M:%S'),
            'sunset': datetime.fromtimestamp(data['sys']['sunset']).strftime('%H:%M:%S'),
            'timestamp': datetime.fromtimestamp(data['dt']).strftime('%Y-%m-%d %H:%M:%S'),
            'timezone_offset': data['timezone']
        }
        
    except requests.exceptions.Timeout:
        raise requests.exceptions.Timeout("Request timed out. Please try again.")
    except requests.exceptions.ConnectionError:
        raise requests.exceptions.ConnectionError("Network connection error.")
    except json.JSONDecodeError:
        raise ValueError("Failed to parse API response.")
    except KeyError as e:
        raise KeyError(f"Unexpected API response structure. Missing key: {e}")

def display_weather_data(weather_data):
    """
    Display weather data in a readable format.
    
    Args:
        weather_data (dict): Weather data dictionary from get_current_weather
    """
    if not weather_data:
        print("No weather data to display.")
        return
    
    print("\n" + "="*50)
    print(f"Weather Report for {weather_data['city']}, {weather_data['country']}")
    print("="*50)
    print(f"Time: {weather_data['timestamp']}")
    print(f"Temperature: {weather_data['temperature']:.1f}°C")
    print(f"Feels like: {weather_data['feels_like']:.1f}°C")
    print(f"Weather: {weather_data['weather_description'].title()}")
    print(f"Humidity: {weather_data['humidity']}%")
    print(f"Pressure: {weather_data['pressure']} hPa")
    print(f"Wind: {weather_data['wind_speed']} m/s at {weather_data['wind_direction']}°")
    print(f"Cloudiness: {weather_data['cloudiness']}%")
    if weather_data['visibility'] != 'N/A':
        print(f"Visibility: {weather_data['visibility'] / 1000:.1f} km")
    print(f"Sunrise: {weather_data['sunrise']}")
    print(f"Sunset: {weather_data['sunset']}")
    print("="*50)

def save_weather_to_file(weather_data, filename='weather_report.json'):
    """
    Save weather data to a JSON file.
    
    Args:
        weather_data (dict): Weather data dictionary
        filename (str): Name of the file to save to
    """
    try:
        with open(filename, 'w') as f:
            json.dump(weather_data, f, indent=2)
        print(f"Weather data saved to {filename}")
    except (IOError, OSError) as e:
        print(f"Error saving to file: {e}")

if __name__ == "__main__":
    # Example usage
    try:
        # Get weather for London
        weather = get_current_weather("London")
        display_weather_data(weather)
        save_weather_to_file(weather)
        
    except ValueError as e:
        print(f"Configuration error: {e}")
        print("Please set your OpenWeatherMap API key:")
        print("1. As an argument to get_current_weather()")
        print("2. Or set OPENWEATHER_API_KEY environment variable")
    except requests.exceptions.RequestException as e:
        print(f"API request failed: {e}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")import requests
import json
import sys

def get_weather(api_key, city):
    base_url = "http://api.openweathermap.org/data/2.5/weather"
    params = {
        'q': city,
        'appid': api_key,
        'units': 'metric'
    }
    try:
        response = requests.get(base_url, params=params)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        print(f"Error fetching weather data: {e}")
        return None

def display_weather(weather_data):
    if weather_data and weather_data.get('cod') == 200:
        main = weather_data['main']
        weather = weather_data['weather'][0]
        print(f"City: {weather_data['name']}")
        print(f"Temperature: {main['temp']}°C")
        print(f"Humidity: {main['humidity']}%")
        print(f"Weather: {weather['description']}")
        print(f"Pressure: {main['pressure']} hPa")
    else:
        print("City not found or invalid data received.")

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python fetch_weather_data.py <api_key> <city>")
        sys.exit(1)
    
    api_key = sys.argv[1]
    city = sys.argv[2]
    
    weather_data = get_weather(api_key, city)
    display_weather(weather_data)