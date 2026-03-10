import requests
import json
from datetime import datetime

class WeatherFetcher:
    def __init__(self, api_key):
        self.api_key = api_key
        self.base_url = "http://api.openweathermap.org/data/2.5/weather"
        self.session = requests.Session()

    def get_weather(self, city_name, units="metric"):
        params = {
            "q": city_name,
            "appid": self.api_key,
            "units": units
        }
        
        try:
            response = self.session.get(self.base_url, params=params, timeout=10)
            response.raise_for_status()
            data = response.json()
            
            return {
                "city": data["name"],
                "country": data["sys"]["country"],
                "temperature": data["main"]["temp"],
                "feels_like": data["main"]["feels_like"],
                "humidity": data["main"]["humidity"],
                "pressure": data["main"]["pressure"],
                "weather": data["weather"][0]["description"],
                "wind_speed": data["wind"]["speed"],
                "wind_direction": data["wind"]["deg"],
                "visibility": data.get("visibility", "N/A"),
                "cloudiness": data["clouds"]["all"],
                "sunrise": datetime.fromtimestamp(data["sys"]["sunrise"]).strftime("%H:%M:%S"),
                "sunset": datetime.fromtimestamp(data["sys"]["sunset"]).strftime("%H:%M:%S"),
                "timestamp": datetime.now().isoformat()
            }
            
        except requests.exceptions.RequestException as e:
            print(f"Network error occurred: {e}")
            return None
        except KeyError as e:
            print(f"Unexpected API response format: {e}")
            return None
        except json.JSONDecodeError as e:
            print(f"Failed to parse JSON response: {e}")
            return None

    def save_to_file(self, weather_data, filename="weather_data.json"):
        if weather_data:
            try:
                with open(filename, "a") as f:
                    json.dump(weather_data, f, indent=2)
                    f.write("\n")
                print(f"Weather data saved to {filename}")
                return True
            except IOError as e:
                print(f"Error saving to file: {e}")
                return False
        return False

def main():
    api_key = "your_api_key_here"
    fetcher = WeatherFetcher(api_key)
    
    cities = ["London", "New York", "Tokyo", "Paris", "Sydney"]
    
    for city in cities:
        print(f"\nFetching weather for {city}...")
        weather = fetcher.get_weather(city)
        
        if weather:
            print(f"Temperature in {weather['city']}, {weather['country']}: {weather['temperature']}°C")
            print(f"Weather: {weather['weather']}")
            print(f"Humidity: {weather['humidity']}%")
            print(f"Wind Speed: {weather['wind_speed']} m/s")
            
            fetcher.save_to_file(weather, "weather_log.json")
        else:
            print(f"Failed to fetch weather data for {city}")

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
        data = response.json()
        return data
    except requests.exceptions.RequestException as e:
        print(f"Error fetching weather data: {e}", file=sys.stderr)
        return None
    except json.JSONDecodeError:
        print("Error decoding JSON response", file=sys.stderr)
        return None

def display_weather(data):
    if data and data.get('cod') == 200:
        city = data['name']
        country = data['sys']['country']
        temp = data['main']['temp']
        weather_desc = data['weather'][0]['description']
        humidity = data['main']['humidity']
        wind_speed = data['wind']['speed']
        
        print(f"Weather in {city}, {country}:")
        print(f"  Temperature: {temp}°C")
        print(f"  Conditions: {weather_desc}")
        print(f"  Humidity: {humidity}%")
        print(f"  Wind Speed: {wind_speed} m/s")
    else:
        error_msg = data.get('message', 'Unknown error') if data else 'No data received'
        print(f"Failed to get weather data: {error_msg}")

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python fetch_weather_data.py <api_key> <city>")
        sys.exit(1)
    
    api_key = sys.argv[1]
    city = sys.argv[2]
    
    weather_data = get_weather(api_key, city)
    display_weather(weather_data)import requests
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
    display_weather(weather_data)import requests
import json
from datetime import datetime
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class WeatherFetcher:
    def __init__(self, api_key, base_url="http://api.openweathermap.org/data/2.5"):
        self.api_key = api_key
        self.base_url = base_url
        self.session = requests.Session()
        
    def get_current_weather(self, city_name, country_code=None):
        try:
            query = city_name
            if country_code:
                query = f"{city_name},{country_code}"
                
            params = {
                'q': query,
                'appid': self.api_key,
                'units': 'metric'
            }
            
            response = self.session.get(
                f"{self.base_url}/weather",
                params=params,
                timeout=10
            )
            response.raise_for_status()
            
            data = response.json()
            return self._parse_weather_data(data)
            
        except requests.exceptions.RequestException as e:
            logger.error(f"API request failed: {e}")
            return None
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse JSON response: {e}")
            return None
            
    def _parse_weather_data(self, raw_data):
        return {
            'city': raw_data.get('name'),
            'country': raw_data.get('sys', {}).get('country'),
            'temperature': raw_data.get('main', {}).get('temp'),
            'feels_like': raw_data.get('main', {}).get('feels_like'),
            'humidity': raw_data.get('main', {}).get('humidity'),
            'pressure': raw_data.get('main', {}).get('pressure'),
            'description': raw_data.get('weather', [{}])[0].get('description'),
            'wind_speed': raw_data.get('wind', {}).get('speed'),
            'wind_direction': raw_data.get('wind', {}).get('deg'),
            'visibility': raw_data.get('visibility'),
            'cloudiness': raw_data.get('clouds', {}).get('all'),
            'sunrise': self._format_timestamp(raw_data.get('sys', {}).get('sunrise')),
            'sunset': self._format_timestamp(raw_data.get('sys', {}).get('sunset')),
            'timestamp': datetime.now().isoformat()
        }
        
    def _format_timestamp(self, timestamp):
        if timestamp:
            return datetime.fromtimestamp(timestamp).strftime('%H:%M:%S')
        return None
        
    def save_to_file(self, data, filename="weather_data.json"):
        try:
            with open(filename, 'a') as f:
                json.dump(data, f, indent=2)
                f.write('\n')
            logger.info(f"Weather data saved to {filename}")
        except IOError as e:
            logger.error(f"Failed to save data to file: {e}")

def main():
    api_key = "your_api_key_here"
    fetcher = WeatherFetcher(api_key)
    
    cities = [
        ("London", "GB"),
        ("New York", "US"),
        ("Tokyo", "JP")
    ]
    
    for city, country in cities:
        logger.info(f"Fetching weather for {city}, {country}")
        weather_data = fetcher.get_current_weather(city, country)
        
        if weather_data:
            print(f"\nWeather in {weather_data['city']}, {weather_data['country']}:")
            print(f"Temperature: {weather_data['temperature']}°C")
            print(f"Feels like: {weather_data['feels_like']}°C")
            print(f"Conditions: {weather_data['description']}")
            print(f"Humidity: {weather_data['humidity']}%")
            print(f"Wind: {weather_data['wind_speed']} m/s")
            
            fetcher.save_to_file(weather_data)
        else:
            print(f"Failed to fetch weather for {city}")

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
        return data
    except requests.exceptions.RequestException as e:
        print(f"Error fetching weather data: {e}")
        return None

def display_weather(data):
    if data and data.get('cod') == 200:
        city = data['name']
        country = data['sys']['country']
        temp = data['main']['temp']
        weather_desc = data['weather'][0]['description']
        humidity = data['main']['humidity']
        wind_speed = data['wind']['speed']
        
        print(f"Weather in {city}, {country}:")
        print(f"  Temperature: {temp}°C")
        print(f"  Conditions: {weather_desc}")
        print(f"  Humidity: {humidity}%")
        print(f"  Wind Speed: {wind_speed} m/s")
    else:
        print("City not found or invalid data received.")

if __name__ == "__main__":
    API_KEY = "your_api_key_here"
    city_name = input("Enter city name: ")
    weather_data = get_weather(API_KEY, city_name)
    display_weather(weather_data)