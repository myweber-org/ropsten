import requests
import json
import os
from datetime import datetime

def get_weather(city_name, api_key=None):
    """
    Fetch current weather data for a given city.
    """
    if api_key is None:
        api_key = os.getenv('OPENWEATHER_API_KEY')
        if api_key is None:
            raise ValueError("API key not provided and OPENWEATHER_API_KEY environment variable not set.")

    base_url = "http://api.openweathermap.org/data/2.5/weather"
    params = {
        'q': city_name,
        'appid': api_key,
        'units': 'metric'
    }

    try:
        response = requests.get(base_url, params=params, timeout=10)
        response.raise_for_status()
        data = response.json()

        if data.get('cod') != 200:
            raise Exception(f"API Error: {data.get('message', 'Unknown error')}")

        main_info = data.get('main', {})
        weather_info = data.get('weather', [{}])[0]
        wind_info = data.get('wind', {})
        sys_info = data.get('sys', {})

        result = {
            'city': data.get('name'),
            'country': sys_info.get('country'),
            'temperature': main_info.get('temp'),
            'feels_like': main_info.get('feels_like'),
            'humidity': main_info.get('humidity'),
            'pressure': main_info.get('pressure'),
            'weather': weather_info.get('description'),
            'wind_speed': wind_info.get('speed'),
            'wind_direction': wind_info.get('deg'),
            'timestamp': datetime.fromtimestamp(data.get('dt')).isoformat(),
            'sunrise': datetime.fromtimestamp(sys_info.get('sunrise')).isoformat(),
            'sunset': datetime.fromtimestamp(sys_info.get('sunset')).isoformat()
        }
        return result

    except requests.exceptions.RequestException as e:
        raise Exception(f"Network error occurred: {e}")
    except json.JSONDecodeError as e:
        raise Exception(f"Failed to parse API response: {e}")
    except KeyError as e:
        raise Exception(f"Unexpected data structure in API response: {e}")

def display_weather(weather_data):
    """
    Print formatted weather information.
    """
    if not weather_data:
        print("No weather data to display.")
        return

    print("\n" + "="*40)
    print(f"Weather in {weather_data['city']}, {weather_data['country']}")
    print("="*40)
    print(f"Condition:   {weather_data['weather'].title()}")
    print(f"Temperature: {weather_data['temperature']}°C")
    print(f"Feels like:  {weather_data['feels_like']}°C")
    print(f"Humidity:    {weather_data['humidity']}%")
    print(f"Pressure:    {weather_data['pressure']} hPa")
    print(f"Wind:        {weather_data['wind_speed']} m/s at {weather_data['wind_direction']}°")
    print(f"Sunrise:     {weather_data['sunrise']}")
    print(f"Sunset:      {weather_data['sunset']}")
    print(f"Updated:     {weather_data['timestamp']}")
    print("="*40)

if __name__ == "__main__":
    # Example usage
    try:
        city = "London"
        print(f"Fetching weather data for {city}...")
        weather = get_weather(city)
        display_weather(weather)
    except Exception as e:
        print(f"Error: {e}")import requests
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
        data = response.json()
        return data
    except requests.exceptions.RequestException as e:
        print(f"Error fetching weather data: {e}")
        return None

def display_weather(data):
    if data is None:
        print("No data to display.")
        return
    if data.get('cod') != 200:
        print(f"Error: {data.get('message', 'Unknown error')}")
        return

    city = data['name']
    country = data['sys']['country']
    temp = data['main']['temp']
    feels_like = data['main']['feels_like']
    humidity = data['main']['humidity']
    weather_desc = data['weather'][0]['description']
    wind_speed = data['wind']['speed']

    print(f"Weather in {city}, {country}:")
    print(f"  Temperature: {temp}°C (feels like {feels_like}°C)")
    print(f"  Conditions: {weather_desc}")
    print(f"  Humidity: {humidity}%")
    print(f"  Wind Speed: {wind_speed} m/s")

def main():
    if len(sys.argv) < 3:
        print("Usage: python fetch_weather_data.py <API_KEY> <CITY_NAME>")
        sys.exit(1)

    api_key = sys.argv[1]
    city = ' '.join(sys.argv[2:])

    weather_data = get_weather(api_key, city)
    display_weather(weather_data)

if __name__ == "__main__":
    main()