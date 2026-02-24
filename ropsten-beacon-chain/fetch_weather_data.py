
import requests
import json
import os
from datetime import datetime

def get_weather(city_name, api_key=None):
    """
    Fetch current weather data for a given city using OpenWeatherMap API.
    
    Args:
        city_name (str): Name of the city to get weather for
        api_key (str): OpenWeatherMap API key. If None, tries to get from env var.
    
    Returns:
        dict: Weather data if successful, None otherwise
    """
    if api_key is None:
        api_key = os.getenv('OPENWEATHER_API_KEY')
    
    if not api_key:
        print("Error: API key not provided and OPENWEATHER_API_KEY env var not set")
        return None
    
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
            print(f"Error: {data.get('message', 'Unknown error')}")
            return None
        
        return {
            'city': data['name'],
            'country': data['sys']['country'],
            'temperature': data['main']['temp'],
            'feels_like': data['main']['feels_like'],
            'humidity': data['main']['humidity'],
            'pressure': data['main']['pressure'],
            'weather': data['weather'][0]['main'],
            'description': data['weather'][0]['description'],
            'wind_speed': data['wind']['speed'],
            'wind_deg': data['wind']['deg'],
            'visibility': data.get('visibility', 'N/A'),
            'clouds': data['clouds']['all'],
            'sunrise': datetime.fromtimestamp(data['sys']['sunrise']).strftime('%H:%M:%S'),
            'sunset': datetime.fromtimestamp(data['sys']['sunset']).strftime('%H:%M:%S'),
            'timestamp': datetime.fromtimestamp(data['dt']).strftime('%Y-%m-%d %H:%M:%S')
        }
        
    except requests.exceptions.RequestException as e:
        print(f"Network error: {e}")
        return None
    except (KeyError, ValueError, json.JSONDecodeError) as e:
        print(f"Data parsing error: {e}")
        return None

def display_weather(weather_data):
    """Display weather data in a readable format."""
    if not weather_data:
        print("No weather data to display")
        return
    
    print("\n" + "="*50)
    print(f"Weather in {weather_data['city']}, {weather_data['country']}")
    print(f"Last updated: {weather_data['timestamp']}")
    print("="*50)
    print(f"Temperature: {weather_data['temperature']}°C (Feels like: {weather_data['feels_like']}°C)")
    print(f"Weather: {weather_data['weather']} - {weather_data['description']}")
    print(f"Humidity: {weather_data['humidity']}%")
    print(f"Pressure: {weather_data['pressure']} hPa")
    print(f"Wind: {weather_data['wind_speed']} m/s at {weather_data['wind_deg']}°")
    print(f"Cloudiness: {weather_data['clouds']}%")
    print(f"Visibility: {weather_data['visibility']} meters")
    print(f"Sunrise: {weather_data['sunrise']}")
    print(f"Sunset: {weather_data['sunset']}")
    print("="*50)

if __name__ == "__main__":
    # Example usage
    city = "London"
    api_key = "your_api_key_here"  # Replace with your actual API key or set env var
    
    print(f"Fetching weather data for {city}...")
    weather = get_weather(city, api_key)
    
    if weather:
        display_weather(weather)
    else:
        print("Failed to fetch weather data")import requests
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
    if weather_data is None:
        print("No weather data to display.")
        return
    if weather_data.get('cod') != 200:
        print(f"Error: {weather_data.get('message', 'Unknown error')}")
        return

    city = weather_data['name']
    country = weather_data['sys']['country']
    temp = weather_data['main']['temp']
    feels_like = weather_data['main']['feels_like']
    humidity = weather_data['main']['humidity']
    description = weather_data['weather'][0]['description']
    wind_speed = weather_data['wind']['speed']

    print(f"Weather in {city}, {country}:")
    print(f"  Temperature: {temp}°C (feels like {feels_like}°C)")
    print(f"  Conditions: {description}")
    print(f"  Humidity: {humidity}%")
    print(f"  Wind Speed: {wind_speed} m/s")

def main():
    if len(sys.argv) < 2:
        print("Usage: python fetch_weather_data.py <city_name>")
        print("Please set your OPENWEATHER_API_KEY environment variable.")
        sys.exit(1)

    city = ' '.join(sys.argv[1:])
    api_key = "YOUR_API_KEY_HERE"

    if api_key == "YOUR_API_KEY_HERE":
        import os
        api_key = os.environ.get('OPENWEATHER_API_KEY')
        if not api_key:
            print("Error: API key not found.")
            print("Please set your OPENWEATHER_API_KEY environment variable or replace the placeholder in the script.")
            sys.exit(1)

    weather_data = get_weather(api_key, city)
    display_weather(weather_data)

if __name__ == "__main__":
    main()import requests
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
        
        if data['cod'] != 200:
            return f"Error: {data.get('message', 'Unknown error')}"
        
        weather_info = {
            'city': data['name'],
            'temperature': data['main']['temp'],
            'humidity': data['main']['humidity'],
            'description': data['weather'][0]['description'],
            'wind_speed': data['wind']['speed']
        }
        return weather_info
        
    except requests.exceptions.RequestException as e:
        return f"Network error: {str(e)}"
    except (KeyError, json.JSONDecodeError) as e:
        return f"Data parsing error: {str(e)}"

def display_weather(weather_data):
    if isinstance(weather_data, dict):
        print(f"Weather in {weather_data['city']}:")
        print(f"  Temperature: {weather_data['temperature']}°C")
        print(f"  Humidity: {weather_data['humidity']}%")
        print(f"  Conditions: {weather_data['description']}")
        print(f"  Wind Speed: {weather_data['wind_speed']} m/s")
    else:
        print(weather_data)

if __name__ == "__main__":
    API_KEY = "your_api_key_here"
    CITY = "London"
    
    result = get_weather(API_KEY, CITY)
    display_weather(result)import requests
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
        print("No weather data to display.")
        return
    try:
        city = data['name']
        country = data['sys']['country']
        temp = data['main']['temp']
        feels_like = data['main']['feels_like']
        humidity = data['main']['humidity']
        weather_desc = data['weather'][0]['description']
        wind_speed = data['wind']['speed']
        print(f"Weather in {city}, {country}:")
        print(f"  Temperature: {temp}°C (Feels like: {feels_like}°C)")
        print(f"  Conditions: {weather_desc}")
        print(f"  Humidity: {humidity}%")
        print(f"  Wind Speed: {wind_speed} m/s")
    except KeyError as e:
        print(f"Unexpected data structure: missing key {e}")

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python fetch_weather_data.py <API_KEY> <CITY_NAME>")
        sys.exit(1)
    api_key = sys.argv[1]
    city = sys.argv[2]
    weather_data = get_weather(api_key, city)
    display_weather(weather_data)