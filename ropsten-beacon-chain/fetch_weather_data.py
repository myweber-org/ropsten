import requests
import json
from datetime import datetime
import logging

class WeatherFetcher:
    def __init__(self, api_key, base_url="http://api.openweathermap.org/data/2.5/weather"):
        self.api_key = api_key
        self.base_url = base_url
        self.session = requests.Session()
        logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
        self.logger = logging.getLogger(__name__)

    def get_weather_by_city(self, city_name, country_code=None):
        query = city_name
        if country_code:
            query = f"{city_name},{country_code}"
        
        params = {
            'q': query,
            'appid': self.api_key,
            'units': 'metric'
        }
        
        try:
            self.logger.info(f"Fetching weather data for: {query}")
            response = self.session.get(self.base_url, params=params, timeout=10)
            response.raise_for_status()
            
            data = response.json()
            return self._parse_weather_data(data)
            
        except requests.exceptions.RequestException as e:
            self.logger.error(f"Request failed: {e}")
            return None
        except json.JSONDecodeError as e:
            self.logger.error(f"Failed to parse JSON response: {e}")
            return None
        except KeyError as e:
            self.logger.error(f"Unexpected data structure: {e}")
            return None

    def _parse_weather_data(self, data):
        weather_info = {
            'city': data.get('name', 'Unknown'),
            'country': data.get('sys', {}).get('country', 'Unknown'),
            'temperature': data.get('main', {}).get('temp', 0),
            'feels_like': data.get('main', {}).get('feels_like', 0),
            'humidity': data.get('main', {}).get('humidity', 0),
            'pressure': data.get('main', {}).get('pressure', 0),
            'weather_description': data.get('weather', [{}])[0].get('description', 'No description'),
            'wind_speed': data.get('wind', {}).get('speed', 0),
            'wind_direction': data.get('wind', {}).get('deg', 0),
            'cloudiness': data.get('clouds', {}).get('all', 0),
            'visibility': data.get('visibility', 0),
            'sunrise': datetime.fromtimestamp(data.get('sys', {}).get('sunrise', 0)).strftime('%H:%M:%S'),
            'sunset': datetime.fromtimestamp(data.get('sys', {}).get('sunset', 0)).strftime('%H:%M:%S'),
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }
        return weather_info

    def display_weather(self, weather_data):
        if not weather_data:
            print("No weather data available.")
            return
        
        print("\n" + "="*50)
        print(f"Weather Report for {weather_data['city']}, {weather_data['country']}")
        print("="*50)
        print(f"Temperature: {weather_data['temperature']}°C (Feels like: {weather_data['feels_like']}°C)")
        print(f"Weather: {weather_data['weather_description'].title()}")
        print(f"Humidity: {weather_data['humidity']}%")
        print(f"Pressure: {weather_data['pressure']} hPa")
        print(f"Wind: {weather_data['wind_speed']} m/s at {weather_data['wind_direction']}°")
        print(f"Cloudiness: {weather_data['cloudiness']}%")
        print(f"Visibility: {weather_data['visibility']} meters")
        print(f"Sunrise: {weather_data['sunrise']}")
        print(f"Sunset: {weather_data['sunset']}")
        print(f"Report Time: {weather_data['timestamp']}")
        print("="*50)

def main():
    API_KEY = "your_api_key_here"
    
    fetcher = WeatherFetcher(API_KEY)
    
    cities_to_check = [
        ("London", "GB"),
        ("New York", "US"),
        ("Tokyo", "JP"),
        ("Sydney", "AU")
    ]
    
    for city, country in cities_to_check:
        weather_data = fetcher.get_weather_by_city(city, country)
        if weather_data:
            fetcher.display_weather(weather_data)
        else:
            print(f"Failed to fetch weather data for {city}, {country}")

if __name__ == "__main__":
    main()import requests
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
    try:
        city = weather_data['name']
        country = weather_data['sys']['country']
        temp = weather_data['main']['temp']
        feels_like = weather_data['main']['feels_like']
        humidity = weather_data['main']['humidity']
        description = weather_data['weather'][0]['description']
        wind_speed = weather_data['wind']['speed']

        print(f"Weather in {city}, {country}:")
        print(f"  Temperature: {temp}°C (feels like {feels_like}°C)")
        print(f"  Humidity: {humidity}%")
        print(f"  Conditions: {description}")
        print(f"  Wind Speed: {wind_speed} m/s")
    except KeyError as e:
        print(f"Unexpected data structure: missing key {e}")

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python fetch_weather_data.py <API_KEY> <CITY_NAME>")
        sys.exit(1)

    api_key = sys.argv[1]
    city_name = ' '.join(sys.argv[2:])
    weather_info = get_weather(api_key, city_name)
    display_weather(weather_info)import requests
import json

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
            
        weather_info = {
            'city': data['name'],
            'country': data['sys']['country'],
            'temperature': data['main']['temp'],
            'feels_like': data['main']['feels_like'],
            'humidity': data['main']['humidity'],
            'pressure': data['main']['pressure'],
            'weather': data['weather'][0]['description'],
            'wind_speed': data['wind']['speed']
        }
        
        return weather_info
        
    except requests.exceptions.RequestException as e:
        print(f"Network error occurred: {e}")
        return None
    except json.JSONDecodeError:
        print("Error decoding JSON response")
        return None
    except KeyError as e:
        print(f"Unexpected data structure: missing key {e}")
        return None

def display_weather(weather_data):
    if weather_data:
        print(f"Weather in {weather_data['city']}, {weather_data['country']}:")
        print(f"  Temperature: {weather_data['temperature']}°C")
        print(f"  Feels like: {weather_data['feels_like']}°C")
        print(f"  Humidity: {weather_data['humidity']}%")
        print(f"  Pressure: {weather_data['pressure']} hPa")
        print(f"  Conditions: {weather_data['weather']}")
        print(f"  Wind Speed: {weather_data['wind_speed']} m/s")
    else:
        print("No weather data available")

if __name__ == "__main__":
    API_KEY = "your_api_key_here"
    city = input("Enter city name: ")
    
    if city:
        weather = get_weather(city, API_KEY)
        display_weather(weather)
    else:
        print("City name cannot be empty")
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
    print(f"  Humidity: {weather_data['humidity']}%")
    print(f"  Pressure: {weather_data['pressure']} hPa")
    print(f"  Conditions: {weather_data['weather'].title()}")
    print(f"  Wind Speed: {weather_data['wind_speed']} m/s")

def main():
    if len(sys.argv) < 3:
        print("Usage: python fetch_weather_data.py <api_key> <city>")
        print("Example: python fetch_weather_data.py your_api_key London")
        sys.exit(1)
    
    api_key = sys.argv[1]
    city = ' '.join(sys.argv[2:])
    
    print(f"Fetching weather data for {city}...")
    weather_data = get_weather(api_key, city)
    
    if weather_data:
        display_weather(weather_data)

if __name__ == "__main__":
    main()