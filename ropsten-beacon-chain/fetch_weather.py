
import requests
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
        
        if data['cod'] != 200:
            print(f"Error: {data.get('message', 'Unknown error')}")
            return None
            
        return {
            'city': data['name'],
            'country': data['sys']['country'],
            'temperature': data['main']['temp'],
            'feels_like': data['main']['feels_like'],
            'humidity': data['main']['humidity'],
            'pressure': data['main']['pressure'],
            'weather': data['weather'][0]['description'],
            'wind_speed': data['wind']['speed']
        }
        
    except requests.exceptions.RequestException as e:
        print(f"Network error: {e}")
        return None
    except (KeyError, json.JSONDecodeError) as e:
        print(f"Data parsing error: {e}")
        return None

def display_weather(weather_data):
    if not weather_data:
        return
        
    print(f"Weather in {weather_data['city']}, {weather_data['country']}:")
    print(f"  Temperature: {weather_data['temperature']}°C")
    print(f"  Feels like: {weather_data['feels_like']}°C")
    print(f"  Conditions: {weather_data['weather']}")
    print(f"  Humidity: {weather_data['humidity']}%")
    print(f"  Pressure: {weather_data['pressure']} hPa")
    print(f"  Wind Speed: {weather_data['wind_speed']} m/s")

def main():
    if len(sys.argv) < 3:
        print("Usage: python fetch_weather.py <api_key> <city>")
        print("Example: python fetch_weather.py abc123 London")
        sys.exit(1)
    
    api_key = sys.argv[1]
    city = ' '.join(sys.argv[2:])
    
    weather_data = get_weather(api_key, city)
    display_weather(weather_data)

if __name__ == "__main__":
    main()import requests
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
    if data and data.get('cod') == 200:
        city = data['name']
        country = data['sys']['country']
        temp = data['main']['temp']
        humidity = data['main']['humidity']
        description = data['weather'][0]['description']
        print(f"Weather in {city}, {country}:")
        print(f"  Temperature: {temp}°C")
        print(f"  Humidity: {humidity}%")
        print(f"  Conditions: {description.capitalize()}")
    else:
        print("City not found or invalid data received.")

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python fetch_weather.py <api_key> <city>")
        sys.exit(1)
    
    api_key = sys.argv[1]
    city = sys.argv[2]
    weather_data = get_weather(api_key, city)
    display_weather(weather_data)
import requests
import json
import os
from datetime import datetime

class WeatherFetcher:
    def __init__(self, api_key=None):
        self.api_key = api_key or os.getenv('OPENWEATHER_API_KEY')
        self.base_url = "http://api.openweathermap.org/data/2.5/weather"
        self.units = "metric"
        
    def get_weather_by_city(self, city_name):
        if not self.api_key:
            raise ValueError("API key not provided. Set OPENWEATHER_API_KEY environment variable.")
        
        params = {
            'q': city_name,
            'appid': self.api_key,
            'units': self.units
        }
        
        try:
            response = requests.get(self.base_url, params=params)
            response.raise_for_status()
            return self._parse_weather_data(response.json())
        except requests.exceptions.RequestException as e:
            return f"Error fetching weather data: {e}"
    
    def _parse_weather_data(self, data):
        weather_info = {
            'city': data.get('name'),
            'country': data.get('sys', {}).get('country'),
            'temperature': data.get('main', {}).get('temp'),
            'feels_like': data.get('main', {}).get('feels_like'),
            'humidity': data.get('main', {}).get('humidity'),
            'pressure': data.get('main', {}).get('pressure'),
            'weather': data.get('weather', [{}])[0].get('description'),
            'wind_speed': data.get('wind', {}).get('speed'),
            'wind_direction': data.get('wind', {}).get('deg'),
            'visibility': data.get('visibility'),
            'sunrise': self._format_timestamp(data.get('sys', {}).get('sunrise')),
            'sunset': self._format_timestamp(data.get('sys', {}).get('sunset')),
            'timestamp': self._format_timestamp(data.get('dt'))
        }
        return weather_info
    
    def _format_timestamp(self, timestamp):
        if timestamp:
            return datetime.fromtimestamp(timestamp).strftime('%Y-%m-%d %H:%M:%S')
        return None
    
    def display_weather(self, weather_data):
        if isinstance(weather_data, str):
            print(weather_data)
            return
        
        print(f"Weather in {weather_data['city']}, {weather_data['country']}:")
        print(f"  Temperature: {weather_data['temperature']}°C (feels like {weather_data['feels_like']}°C)")
        print(f"  Conditions: {weather_data['weather'].title()}")
        print(f"  Humidity: {weather_data['humidity']}%")
        print(f"  Pressure: {weather_data['pressure']} hPa")
        print(f"  Wind: {weather_data['wind_speed']} m/s at {weather_data['wind_direction']}°")
        print(f"  Visibility: {weather_data['visibility']} meters")
        print(f"  Sunrise: {weather_data['sunrise']}")
        print(f"  Sunset: {weather_data['sunset']}")
        print(f"  Last updated: {weather_data['timestamp']}")

def main():
    fetcher = WeatherFetcher()
    
    cities = ["London", "New York", "Tokyo", "Paris"]
    
    for city in cities:
        print(f"\n{'='*50}")
        weather = fetcher.get_weather_by_city(city)
        fetcher.display_weather(weather)

if __name__ == "__main__":
    main()
import requests
import sys

API_KEY = "your_api_key_here"
BASE_URL = "http://api.openweathermap.org/data/2.5/weather"

def get_weather(city_name):
    params = {
        "q": city_name,
        "appid": API_KEY,
        "units": "metric"
    }
    try:
        response = requests.get(BASE_URL, params=params)
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
    try:
        city = data["name"]
        country = data["sys"]["country"]
        temp = data["main"]["temp"]
        humidity = data["main"]["humidity"]
        description = data["weather"][0]["description"]
        print(f"Weather in {city}, {country}:")
        print(f"  Temperature: {temp}°C")
        print(f"  Humidity: {humidity}%")
        print(f"  Conditions: {description}")
    except KeyError as e:
        print(f"Unexpected data format: missing key {e}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python fetch_weather.py <city_name>")
        sys.exit(1)
    city = " ".join(sys.argv[1:])
    weather_data = get_weather(city)
    display_weather(weather_data)import requests
import json
from datetime import datetime

def fetch_weather_data(api_key, city):
    """
    Fetch current weather data for a given city using OpenWeatherMap API.
    """
    base_url = "http://api.openweathermap.org/data/2.5/weather"
    params = {
        'q': city,
        'appid': api_key,
        'units': 'metric'
    }
    
    try:
        response = requests.get(base_url, params=params, timeout=10)
        response.raise_for_status()
        
        data = response.json()
        
        if data.get('cod') != 200:
            return {'error': data.get('message', 'Unknown error')}
        
        return {
            'city': data['name'],
            'country': data['sys']['country'],
            'temperature': data['main']['temp'],
            'feels_like': data['main']['feels_like'],
            'humidity': data['main']['humidity'],
            'pressure': data['main']['pressure'],
            'weather': data['weather'][0]['description'],
            'wind_speed': data['wind']['speed'],
            'timestamp': datetime.fromtimestamp(data['dt']).isoformat()
        }
        
    except requests.exceptions.RequestException as e:
        return {'error': f'Network error: {str(e)}'}
    except (KeyError, ValueError, json.JSONDecodeError) as e:
        return {'error': f'Data parsing error: {str(e)}'}

def display_weather(weather_data):
    """
    Display weather information in a readable format.
    """
    if 'error' in weather_data:
        print(f"Error: {weather_data['error']}")
        return
    
    print("\n" + "="*40)
    print(f"Weather in {weather_data['city']}, {weather_data['country']}")
    print("="*40)
    print(f"Temperature: {weather_data['temperature']}°C")
    print(f"Feels like: {weather_data['feels_like']}°C")
    print(f"Weather: {weather_data['weather'].title()}")
    print(f"Humidity: {weather_data['humidity']}%")
    print(f"Pressure: {weather_data['pressure']} hPa")
    print(f"Wind Speed: {weather_data['wind_speed']} m/s")
    print(f"Last Updated: {weather_data['timestamp']}")
    print("="*40)

if __name__ == "__main__":
    API_KEY = "your_api_key_here"
    CITY = "London"
    
    print(f"Fetching weather data for {CITY}...")
    weather_data = fetch_weather_data(API_KEY, CITY)
    display_weather(weather_data)