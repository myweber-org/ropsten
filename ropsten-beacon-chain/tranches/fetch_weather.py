import requests
import json
from datetime import datetime
from typing import Optional, Dict, Any

class WeatherFetcher:
    BASE_URL = "http://api.openweathermap.org/data/2.5/weather"
    
    def __init__(self, api_key: str):
        self.api_key = api_key
        
    def get_weather(self, city: str, country_code: Optional[str] = None) -> Dict[str, Any]:
        query = city
        if country_code:
            query = f"{city},{country_code}"
            
        params = {
            'q': query,
            'appid': self.api_key,
            'units': 'metric'
        }
        
        try:
            response = requests.get(self.BASE_URL, params=params, timeout=10)
            response.raise_for_status()
            data = response.json()
            
            return {
                'city': data['name'],
                'country': data['sys']['country'],
                'temperature': data['main']['temp'],
                'feels_like': data['main']['feels_like'],
                'humidity': data['main']['humidity'],
                'pressure': data['main']['pressure'],
                'weather': data['weather'][0]['description'],
                'wind_speed': data['wind']['speed'],
                'timestamp': datetime.fromtimestamp(data['dt']).isoformat(),
                'sunrise': datetime.fromtimestamp(data['sys']['sunrise']).isoformat(),
                'sunset': datetime.fromtimestamp(data['sys']['sunset']).isoformat()
            }
            
        except requests.exceptions.RequestException as e:
            return {'error': f'Network error: {str(e)}'}
        except KeyError as e:
            return {'error': f'Invalid response format: missing key {str(e)}'}
        except json.JSONDecodeError:
            return {'error': 'Invalid JSON response'}
            
    def display_weather(self, weather_data: Dict[str, Any]) -> None:
        if 'error' in weather_data:
            print(f"Error: {weather_data['error']}")
            return
            
        print(f"Weather in {weather_data['city']}, {weather_data['country']}:")
        print(f"  Temperature: {weather_data['temperature']}°C")
        print(f"  Feels like: {weather_data['feels_like']}°C")
        print(f"  Conditions: {weather_data['weather'].title()}")
        print(f"  Humidity: {weather_data['humidity']}%")
        print(f"  Pressure: {weather_data['pressure']} hPa")
        print(f"  Wind Speed: {weather_data['wind_speed']} m/s")
        print(f"  Sunrise: {weather_data['sunrise']}")
        print(f"  Sunset: {weather_data['sunset']}")
        print(f"  Last Updated: {weather_data['timestamp']}")

def main():
    API_KEY = "your_api_key_here"  # Replace with actual API key
    
    fetcher = WeatherFetcher(API_KEY)
    
    cities = [
        ("London", "UK"),
        ("New York", "US"),
        ("Tokyo", "JP"),
        ("Paris", "FR")
    ]
    
    for city, country in cities:
        print("\n" + "="*50)
        weather = fetcher.get_weather(city, country)
        fetcher.display_weather(weather)

if __name__ == "__main__":
    main()
import requests
import json
import logging
from datetime import datetime
from typing import Optional, Dict, Any

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class WeatherFetcher:
    BASE_URL = "http://api.openweathermap.org/data/2.5/weather"
    
    def __init__(self, api_key: str):
        self.api_key = api_key
        
    def get_weather(self, city: str, country_code: Optional[str] = None) -> Dict[str, Any]:
        query = city
        if country_code:
            query = f"{city},{country_code}"
            
        params = {
            'q': query,
            'appid': self.api_key,
            'units': 'metric'
        }
        
        try:
            response = requests.get(self.BASE_URL, params=params, timeout=10)
            response.raise_for_status()
            data = response.json()
            
            if data.get('cod') != 200:
                raise ValueError(f"API error: {data.get('message', 'Unknown error')}")
                
            return self._parse_weather_data(data)
            
        except requests.exceptions.RequestException as e:
            logger.error(f"Network error fetching weather for {query}: {e}")
            raise
        except json.JSONDecodeError as e:
            logger.error(f"Invalid JSON response for {query}: {e}")
            raise
        except ValueError as e:
            logger.error(f"API error for {query}: {e}")
            raise
            
    def _parse_weather_data(self, data: Dict[str, Any]) -> Dict[str, Any]:
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
            'wind_deg': data['wind'].get('deg', 0),
            'visibility': data.get('visibility', 0),
            'clouds': data['clouds']['all'],
            'sunrise': datetime.fromtimestamp(data['sys']['sunrise']).isoformat(),
            'sunset': datetime.fromtimestamp(data['sys']['sunset']).isoformat(),
            'timestamp': datetime.now().isoformat()
        }
        
    def save_to_file(self, data: Dict[str, Any], filename: str = "weather_data.json"):
        try:
            with open(filename, 'a') as f:
                json.dump(data, f, indent=2)
                f.write('\n')
            logger.info(f"Weather data saved to {filename}")
        except IOError as e:
            logger.error(f"Error saving to file {filename}: {e}")
            raise

def main():
    api_key = "your_api_key_here"
    fetcher = WeatherFetcher(api_key)
    
    cities = [
        ("London", "UK"),
        ("New York", "US"),
        ("Tokyo", "JP")
    ]
    
    for city, country in cities:
        try:
            weather_data = fetcher.get_weather(city, country)
            print(f"Weather in {weather_data['city']}, {weather_data['country']}:")
            print(f"  Temperature: {weather_data['temperature']}°C")
            print(f"  Conditions: {weather_data['weather']} - {weather_data['description']}")
            print(f"  Humidity: {weather_data['humidity']}%")
            print(f"  Wind: {weather_data['wind_speed']} m/s")
            print()
            
            fetcher.save_to_file(weather_data)
            
        except Exception as e:
            logger.error(f"Failed to fetch weather for {city}, {country}: {e}")
            continue

if __name__ == "__main__":
    main()