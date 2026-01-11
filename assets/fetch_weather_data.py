
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
    except json.JSONDecodeError:
        print("Error parsing response")
        return None

def display_weather(weather_data):
    if not weather_data:
        return
        
    print(f"Weather in {weather_data['city']}, {weather_data['country']}:")
    print(f"  Temperature: {weather_data['temperature']}°C")
    print(f"  Feels like: {weather_data['feels_like']}°C")
    print(f"  Humidity: {weather_data['humidity']}%")
    print(f"  Pressure: {weather_data['pressure']} hPa")
    print(f"  Conditions: {weather_data['weather']}")
    print(f"  Wind speed: {weather_data['wind_speed']} m/s")

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python fetch_weather_data.py <api_key> <city>")
        print("Example: python fetch_weather_data.py abc123 London")
        sys.exit(1)
    
    api_key = sys.argv[1]
    city = ' '.join(sys.argv[2:])
    
    weather = get_weather(api_key, city)
    display_weather(weather)
import requests
import json
import sys
from datetime import datetime

class WeatherFetcher:
    def __init__(self, api_key):
        self.api_key = api_key
        self.base_url = "http://api.openweathermap.org/data/2.5/weather"
    
    def get_weather(self, city_name):
        params = {
            'q': city_name,
            'appid': self.api_key,
            'units': 'metric'
        }
        
        try:
            response = requests.get(self.base_url, params=params)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            print(f"Error fetching weather data: {e}")
            return None
    
    def display_weather(self, weather_data):
        if not weather_data:
            return
        
        main = weather_data.get('main', {})
        weather = weather_data.get('weather', [{}])[0]
        sys_info = weather_data.get('sys', {})
        
        print(f"Weather Report for {weather_data.get('name', 'Unknown')}")
        print(f"Country: {sys_info.get('country', 'N/A')}")
        print(f"Temperature: {main.get('temp', 'N/A')}°C")
        print(f"Feels like: {main.get('feels_like', 'N/A')}°C")
        print(f"Humidity: {main.get('humidity', 'N/A')}%")
        print(f"Pressure: {main.get('pressure', 'N/A')} hPa")
        print(f"Weather: {weather.get('description', 'N/A').title()}")
        print(f"Wind Speed: {weather_data.get('wind', {}).get('speed', 'N/A')} m/s")
        print(f"Visibility: {weather_data.get('visibility', 'N/A')} meters")
        print(f"Sunrise: {datetime.fromtimestamp(sys_info.get('sunrise', 0)).strftime('%H:%M:%S')}")
        print(f"Sunset: {datetime.fromtimestamp(sys_info.get('sunset', 0)).strftime('%H:%M:%S')}")
    
    def save_to_file(self, weather_data, filename="weather_data.json"):
        if not weather_data:
            return False
        
        try:
            with open(filename, 'w') as f:
                json.dump(weather_data, f, indent=2)
            print(f"Weather data saved to {filename}")
            return True
        except IOError as e:
            print(f"Error saving to file: {e}")
            return False

def main():
    api_key = "your_api_key_here"
    
    if api_key == "your_api_key_here":
        print("Please replace 'your_api_key_here' with your actual OpenWeatherMap API key")
        print("Get your API key from: https://openweathermap.org/api")
        return
    
    fetcher = WeatherFetcher(api_key)
    
    if len(sys.argv) > 1:
        city = ' '.join(sys.argv[1:])
    else:
        city = input("Enter city name: ")
    
    weather_data = fetcher.get_weather(city)
    
    if weather_data:
        if weather_data.get('cod') != 200:
            print(f"Error: {weather_data.get('message', 'Unknown error')}")
            return
        
        fetcher.display_weather(weather_data)
        
        save_option = input("\nSave to file? (y/n): ").lower()
        if save_option == 'y':
            filename = input("Enter filename (default: weather_data.json): ").strip()
            if not filename:
                filename = "weather_data.json"
            fetcher.save_to_file(weather_data, filename)

if __name__ == "__main__":
    main()