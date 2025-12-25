
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
        return response.json()
    except requests.exceptions.RequestException as e:
        print(f"Error fetching weather data: {e}")
        return None

def display_weather(data):
    if not data:
        print("No weather data available.")
        return
    
    main = data['main']
    weather = data['weather'][0]
    
    print(f"City: {data['name']}")
    print(f"Temperature: {main['temp']}°C")
    print(f"Feels like: {main['feels_like']}°C")
    print(f"Humidity: {main['humidity']}%")
    print(f"Pressure: {main['pressure']} hPa")
    print(f"Weather: {weather['description'].capitalize()}")
    print(f"Wind Speed: {data['wind']['speed']} m/s")

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python fetch_weather_data.py <api_key> <city>")
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
            'wind_speed': data['wind']['speed'],
            'wind_direction': data['wind'].get('deg', 'N/A'),
            'visibility': data.get('visibility', 'N/A'),
            'clouds': data['clouds']['all'],
            'sunrise': datetime.fromtimestamp(data['sys']['sunrise']).strftime('%H:%M:%S'),
            'sunset': datetime.fromtimestamp(data['sys']['sunset']).strftime('%H:%M:%S'),
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }
        
    except requests.exceptions.RequestException as e:
        print(f"Network error: {e}")
        return None
    except json.JSONDecodeError as e:
        print(f"JSON decode error: {e}")
        return None
    except KeyError as e:
        print(f"Unexpected data structure: {e}")
        return None

def save_weather_data(data, filename='weather_data.json'):
    if data is None:
        return False
        
    try:
        existing_data = []
        if os.path.exists(filename):
            with open(filename, 'r') as f:
                existing_data = json.load(f)
        
        existing_data.append(data)
        
        with open(filename, 'w') as f:
            json.dump(existing_data, f, indent=2)
        
        print(f"Weather data saved to {filename}")
        return True
        
    except (IOError, json.JSONDecodeError) as e:
        print(f"Error saving data: {e}")
        return False

def display_weather_data(data):
    if data is None:
        print("No weather data to display")
        return
        
    print("\n" + "="*50)
    print(f"Weather Report for {data['city']}, {data['country']}")
    print(f"Timestamp: {data['timestamp']}")
    print("="*50)
    print(f"Temperature: {data['temperature']}°C")
    print(f"Feels like: {data['feels_like']}°C")
    print(f"Weather: {data['weather'].title()}")
    print(f"Humidity: {data['humidity']}%")
    print(f"Pressure: {data['pressure']} hPa")
    print(f"Wind: {data['wind_speed']} m/s at {data['wind_direction']}°")
    print(f"Cloudiness: {data['clouds']}%")
    print(f"Visibility: {data['visibility']} meters")
    print(f"Sunrise: {data['sunrise']}")
    print(f"Sunset: {data['sunset']}")
    print("="*50)

def main():
    api_key = os.environ.get('OPENWEATHER_API_KEY')
    
    if not api_key:
        print("Please set OPENWEATHER_API_KEY environment variable")
        return
    
    city_name = input("Enter city name: ").strip()
    
    if not city_name:
        print("City name cannot be empty")
        return
    
    print(f"Fetching weather data for {city_name}...")
    weather_data = get_weather_data(city_name, api_key)
    
    if weather_data:
        display_weather_data(weather_data)
        
        save_option = input("\nSave this data? (y/n): ").strip().lower()
        if save_option == 'y':
            save_weather_data(weather_data)
    else:
        print("Failed to fetch weather data")

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
        
        if data['cod'] != 200:
            print(f"Error: {data.get('message', 'Unknown error')}")
            return None
            
        return data
        
    except requests.exceptions.RequestException as e:
        print(f"Network error: {e}")
        return None
    except json.JSONDecodeError:
        print("Error: Invalid response from server")
        return None

def display_weather(weather_data):
    if not weather_data:
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
    if len(sys.argv) < 3:
        print("Usage: python fetch_weather_data.py <api_key> <city>")
        print("Example: python fetch_weather_data.py abc123 London")
        sys.exit(1)
    
    api_key = sys.argv[1]
    city = ' '.join(sys.argv[2:])
    
    weather_data = get_weather(api_key, city)
    
    if weather_data:
        display_weather(weather_data)
    else:
        print("Failed to fetch weather data")

if __name__ == "__main__":
    main()
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
            
        return data
        
    except requests.exceptions.RequestException as e:
        print(f"Network error: {e}")
        return None
    except json.JSONDecodeError:
        print("Error parsing API response")
        return None

def display_weather(data):
    if not data:
        return
        
    city = data['name']
    country = data['sys']['country']
    temp = data['main']['temp']
    feels_like = data['main']['feels_like']
    humidity = data['main']['humidity']
    description = data['weather'][0]['description']
    wind_speed = data['wind']['speed']
    
    print(f"Weather in {city}, {country}:")
    print(f"  Temperature: {temp}°C (feels like {feels_like}°C)")
    print(f"  Conditions: {description}")
    print(f"  Humidity: {humidity}%")
    print(f"  Wind Speed: {wind_speed} m/s")

def main():
    if len(sys.argv) < 3:
        print("Usage: python fetch_weather_data.py <api_key> <city>")
        print("Example: python fetch_weather_data.py abc123 London")
        sys.exit(1)
    
    api_key = sys.argv[1]
    city = ' '.join(sys.argv[2:])
    
    weather_data = get_weather(api_key, city)
    
    if weather_data:
        display_weather(weather_data)

if __name__ == "__main__":
    main()
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
    print(f"  Conditions: {weather_data['weather'].title()}")
    print(f"  Humidity: {weather_data['humidity']}%")
    print(f"  Pressure: {weather_data['pressure']} hPa")
    print(f"  Wind Speed: {weather_data['wind_speed']} m/s")

def main():
    if len(sys.argv) < 3:
        print("Usage: python fetch_weather_data.py <api_key> <city>")
        print("Example: python fetch_weather_data.py abc123 London")
        sys.exit(1)
    
    api_key = sys.argv[1]
    city = ' '.join(sys.argv[2:])
    
    print(f"Fetching weather data for {city}...")
    weather_data = get_weather(api_key, city)
    
    if weather_data:
        display_weather(weather_data)
    else:
        print("Failed to retrieve weather data.")

if __name__ == "__main__":
    main()
import requests
import json
from datetime import datetime
import logging

class WeatherFetcher:
    def __init__(self, api_key, base_url="http://api.openweathermap.org/data/2.5"):
        self.api_key = api_key
        self.base_url = base_url
        self.session = requests.Session()
        logging.basicConfig(level=logging.INFO)
        
    def get_current_weather(self, city_name, units="metric"):
        endpoint = f"{self.base_url}/weather"
        params = {
            "q": city_name,
            "appid": self.api_key,
            "units": units
        }
        
        try:
            response = self.session.get(endpoint, params=params, timeout=10)
            response.raise_for_status()
            data = response.json()
            
            return {
                "city": data["name"],
                "temperature": data["main"]["temp"],
                "humidity": data["main"]["humidity"],
                "description": data["weather"][0]["description"],
                "wind_speed": data["wind"]["speed"],
                "timestamp": datetime.fromtimestamp(data["dt"]).isoformat()
            }
            
        except requests.exceptions.RequestException as e:
            logging.error(f"API request failed: {e}")
            return None
        except (KeyError, ValueError) as e:
            logging.error(f"Data parsing error: {e}")
            return None
    
    def save_to_file(self, data, filename="weather_data.json"):
        if data:
            try:
                with open(filename, 'a') as f:
                    json.dump(data, f)
                    f.write('\n')
                logging.info(f"Weather data saved to {filename}")
            except IOError as e:
                logging.error(f"File save error: {e}")
    
    def fetch_multiple_cities(self, cities, units="metric"):
        results = []
        for city in cities:
            weather = self.get_current_weather(city, units)
            if weather:
                results.append(weather)
                self.save_to_file(weather)
        return results

def main():
    API_KEY = "your_api_key_here"
    fetcher = WeatherFetcher(API_KEY)
    
    cities = ["London", "New York", "Tokyo", "Paris"]
    
    print("Fetching weather data...")
    results = fetcher.fetch_multiple_cities(cities)
    
    for result in results:
        if result:
            print(f"{result['city']}: {result['temperature']}°C, {result['description']}")

if __name__ == "__main__":
    main()