
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
        error_msg = data.get('message', 'Unknown error') if data else 'No data received'
        print(f"Failed to retrieve weather. Error: {error_msg}")

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python fetch_weather_data.py <API_KEY> <CITY_NAME>")
        sys.exit(1)
    api_key = sys.argv[1]
    city = sys.argv[2]
    weather_data = get_weather(api_key, city)
    display_weather(weather_data)
import requests
import json
import os
from datetime import datetime

def get_weather_data(city_name, api_key):
    base_url = "http://api.openweathermap.org/data/2.5/weather"
    params = {
        'q': city_name,
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

def display_weather_info(weather_data):
    if not weather_data:
        return
    
    main_info = weather_data['main']
    weather_desc = weather_data['weather'][0]['description']
    wind_speed = weather_data['wind']['speed']
    city = weather_data['name']
    country = weather_data['sys']['country']
    
    print(f"Weather in {city}, {country}:")
    print(f"Temperature: {main_info['temp']}°C")
    print(f"Feels like: {main_info['feels_like']}°C")
    print(f"Humidity: {main_info['humidity']}%")
    print(f"Pressure: {main_info['pressure']} hPa")
    print(f"Weather: {weather_desc}")
    print(f"Wind Speed: {wind_speed} m/s")

def save_to_file(weather_data, filename="weather_log.json"):
    if not weather_data:
        return
    
    log_entry = {
        'timestamp': datetime.now().isoformat(),
        'data': weather_data
    }
    
    try:
        if os.path.exists(filename):
            with open(filename, 'r') as f:
                data = json.load(f)
        else:
            data = []
        
        data.append(log_entry)
        
        with open(filename, 'w') as f:
            json.dump(data, f, indent=2)
        
        print(f"Weather data saved to {filename}")
    except Exception as e:
        print(f"Error saving to file: {e}")

def main():
    api_key = os.environ.get('OPENWEATHER_API_KEY')
    
    if not api_key:
        print("Please set OPENWEATHER_API_KEY environment variable")
        return
    
    city = input("Enter city name: ").strip()
    
    if not city:
        print("City name cannot be empty")
        return
    
    weather_data = get_weather_data(city, api_key)
    
    if weather_data and weather_data.get('cod') == 200:
        display_weather_info(weather_data)
        save_to_file(weather_data)
    else:
        print(f"Could not fetch weather data for {city}")
        if weather_data:
            print(f"Error: {weather_data.get('message', 'Unknown error')}")

if __name__ == "__main__":
    main()
import requests
import json
import os
from datetime import datetime

def get_weather(city_name, api_key):
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
            print(f"Error: {data.get('message', 'Unknown error')}")
            return None
            
        return data
        
    except requests.exceptions.RequestException as e:
        print(f"Network error occurred: {e}")
        return None
    except json.JSONDecodeError:
        print("Error decoding JSON response")
        return None

def display_weather(data):
    if not data:
        return
        
    main = data['main']
    weather = data['weather'][0]
    sys = data['sys']
    
    print(f"\nWeather in {data['name']}, {sys['country']}")
    print(f"Temperature: {main['temp']}°C (Feels like: {main['feels_like']}°C)")
    print(f"Conditions: {weather['description'].title()}")
    print(f"Humidity: {main['humidity']}%")
    print(f"Pressure: {main['pressure']} hPa")
    print(f"Wind Speed: {data['wind']['speed']} m/s")
    
    if 'sunrise' in sys and 'sunset' in sys:
        sunrise = datetime.fromtimestamp(sys['sunrise']).strftime('%H:%M')
        sunset = datetime.fromtimestamp(sys['sunset']).strftime('%H:%M')
        print(f"Sunrise: {sunrise}, Sunset: {sunset}")

def main():
    api_key = os.environ.get('OPENWEATHER_API_KEY')
    
    if not api_key:
        print("Please set OPENWEATHER_API_KEY environment variable")
        return
        
    city = input("Enter city name: ").strip()
    
    if not city:
        print("City name cannot be empty")
        return
        
    weather_data = get_weather(city, api_key)
    display_weather(weather_data)

if __name__ == "__main__":
    main()