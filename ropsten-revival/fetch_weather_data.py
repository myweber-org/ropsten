import requests
import json

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

def display_weather(weather_data):
    if weather_data is None:
        print("No weather data available.")
        return
    try:
        city = weather_data['name']
        country = weather_data['sys']['country']
        temp = weather_data['main']['temp']
        humidity = weather_data['main']['humidity']
        description = weather_data['weather'][0]['description']
        print(f"Weather in {city}, {country}:")
        print(f"Temperature: {temp}°C")
        print(f"Humidity: {humidity}%")
        print(f"Conditions: {description}")
    except KeyError as e:
        print(f"Unexpected data format: missing key {e}")

if __name__ == "__main__":
    API_KEY = "your_api_key_here"
    city_name = input("Enter city name: ")
    weather = get_weather(API_KEY, city_name)
    display_weather(weather)
import requests
import json
import os
from datetime import datetime

def get_weather(city_name, api_key=None):
    """
    Fetch current weather data for a given city using OpenWeatherMap API.
    """
    if api_key is None:
        api_key = os.environ.get('OPENWEATHER_API_KEY')
        if api_key is None:
            raise ValueError("API key not provided and OPENWEATHER_API_KEY environment variable not set")

    base_url = "http://api.openweathermap.org/data/2.5/weather"
    params = {
        'q': city_name,
        'appid': api_key,
        'units': 'metric'
    }

    try:
        response = requests.get(base_url, params=params)
        response.raise_for_status()
        data = response.json()

        if data.get('cod') != 200:
            return {'error': data.get('message', 'Unknown error')}

        weather_info = {
            'city': data['name'],
            'country': data['sys']['country'],
            'temperature': data['main']['temp'],
            'feels_like': data['main']['feels_like'],
            'humidity': data['main']['humidity'],
            'pressure': data['main']['pressure'],
            'weather': data['weather'][0]['main'],
            'description': data['weather'][0]['description'],
            'wind_speed': data['wind']['speed'],
            'wind_direction': data['wind'].get('deg', 'N/A'),
            'visibility': data.get('visibility', 'N/A'),
            'cloudiness': data['clouds']['all'],
            'sunrise': datetime.fromtimestamp(data['sys']['sunrise']).strftime('%H:%M:%S'),
            'sunset': datetime.fromtimestamp(data['sys']['sunset']).strftime('%H:%M:%S'),
            'timestamp': datetime.fromtimestamp(data['dt']).strftime('%Y-%m-%d %H:%M:%S')
        }
        return weather_info

    except requests.exceptions.RequestException as e:
        return {'error': f'Network error: {str(e)}'}
    except (KeyError, json.JSONDecodeError) as e:
        return {'error': f'Data parsing error: {str(e)}'}

def display_weather(weather_data):
    """
    Display weather information in a readable format.
    """
    if 'error' in weather_data:
        print(f"Error: {weather_data['error']}")
        return

    print("\n" + "="*50)
    print(f"Weather in {weather_data['city']}, {weather_data['country']}")
    print(f"Last updated: {weather_data['timestamp']}")
    print("="*50)
    print(f"Temperature: {weather_data['temperature']}°C (Feels like: {weather_data['feels_like']}°C)")
    print(f"Weather: {weather_data['weather']} - {weather_data['description']}")
    print(f"Humidity: {weather_data['humidity']}%")
    print(f"Pressure: {weather_data['pressure']} hPa")
    print(f"Wind: {weather_data['wind_speed']} m/s at {weather_data['wind_direction']}°")
    print(f"Visibility: {weather_data['visibility']} meters")
    print(f"Cloudiness: {weather_data['cloudiness']}%")
    print(f"Sunrise: {weather_data['sunrise']}")
    print(f"Sunset: {weather_data['sunset']}")
    print("="*50 + "\n")

if __name__ == "__main__":
    # Example usage
    city = "London"
    api_key = os.environ.get('OPENWEATHER_API_KEY')
    
    if not api_key:
        print("Please set OPENWEATHER_API_KEY environment variable")
        print("Example: export OPENWEATHER_API_KEY='your_api_key_here'")
    else:
        weather = get_weather(city, api_key)
        display_weather(weather)