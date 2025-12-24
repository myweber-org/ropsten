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

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python fetch_weather_data.py <API_KEY> <CITY_NAME>")
        sys.exit(1)

    api_key = sys.argv[1]
    city = ' '.join(sys.argv[2:])
    weather_data = get_weather(api_key, city)
    display_weather(weather_data)
import requests
import json
from datetime import datetime

def get_weather(api_key, city_name):
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
        
        if data['cod'] != 200:
            return f"Error: {data.get('message', 'Unknown error')}"
        
        weather_info = {
            'city': data['name'],
            'country': data['sys']['country'],
            'temperature': data['main']['temp'],
            'feels_like': data['main']['feels_like'],
            'humidity': data['main']['humidity'],
            'pressure': data['main']['pressure'],
            'weather': data['weather'][0]['description'],
            'wind_speed': data['wind']['speed'],
            'timestamp': datetime.fromtimestamp(data['dt']).strftime('%Y-%m-%d %H:%M:%S')
        }
        return weather_info
        
    except requests.exceptions.RequestException as e:
        return f"Network error occurred: {e}"
    except (KeyError, json.JSONDecodeError) as e:
        return f"Data parsing error: {e}"

def display_weather(weather_data):
    if isinstance(weather_data, dict):
        print(f"Weather in {weather_data['city']}, {weather_data['country']}:")
        print(f"  Temperature: {weather_data['temperature']}°C")
        print(f"  Feels like: {weather_data['feels_like']}°C")
        print(f"  Conditions: {weather_data['weather'].title()}")
        print(f"  Humidity: {weather_data['humidity']}%")
        print(f"  Pressure: {weather_data['pressure']} hPa")
        print(f"  Wind Speed: {weather_data['wind_speed']} m/s")
        print(f"  Last Updated: {weather_data['timestamp']}")
    else:
        print(weather_data)

if __name__ == "__main__":
    API_KEY = "your_api_key_here"
    CITY = "London"
    
    result = get_weather(API_KEY, CITY)
    display_weather(result)import requests
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

def display_weather(data):
    if data and data.get('cod') == 200:
        city = data['name']
        country = data['sys']['country']
        temp = data['main']['temp']
        humidity = data['main']['humidity']
        description = data['weather'][0]['description']
        print(f"Weather in {city}, {country}:")
        print(f"Temperature: {temp}°C")
        print(f"Humidity: {humidity}%")
        print(f"Conditions: {description}")
    else:
        print("City not found or invalid data received.")

if __name__ == "__main__":
    API_KEY = "your_api_key_here"
    city_name = input("Enter city name: ")
    weather_data = get_weather(API_KEY, city_name)
    display_weather(weather_data)