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
        self.logger = logging.getLogger(__name__)

    def get_current_weather(self, city_name, country_code=None):
        location = f"{city_name},{country_code}" if country_code else city_name
        endpoint = f"{self.base_url}/weather"
        params = {
            'q': location,
            'appid': self.api_key,
            'units': 'metric'
        }
        
        try:
            response = self.session.get(endpoint, params=params, timeout=10)
            response.raise_for_status()
            data = response.json()
            
            if data.get('cod') != 200:
                self.logger.error(f"API error: {data.get('message', 'Unknown error')}")
                return None
            
            return self._parse_weather_data(data)
            
        except requests.exceptions.RequestException as e:
            self.logger.error(f"Request failed: {e}")
            return None
        except json.JSONDecodeError as e:
            self.logger.error(f"Invalid JSON response: {e}")
            return None

    def _parse_weather_data(self, raw_data):
        parsed = {
            'timestamp': datetime.fromtimestamp(raw_data['dt']).isoformat(),
            'location': raw_data['name'],
            'country': raw_data['sys']['country'],
            'temperature': raw_data['main']['temp'],
            'feels_like': raw_data['main']['feels_like'],
            'humidity': raw_data['main']['humidity'],
            'pressure': raw_data['main']['pressure'],
            'weather': raw_data['weather'][0]['main'],
            'description': raw_data['weather'][0]['description'],
            'wind_speed': raw_data['wind']['speed'],
            'wind_direction': raw_data['wind'].get('deg', 'N/A'),
            'visibility': raw_data.get('visibility', 'N/A'),
            'cloudiness': raw_data['clouds']['all']
        }
        return parsed

    def save_to_file(self, data, filename="weather_data.json"):
        if not data:
            self.logger.warning("No data to save")
            return False
        
        try:
            with open(filename, 'a') as f:
                json.dump(data, f, indent=2)
                f.write('\n')
            self.logger.info(f"Weather data saved to {filename}")
            return True
        except IOError as e:
            self.logger.error(f"Failed to save file: {e}")
            return False

def main():
    api_key = "your_api_key_here"
    fetcher = WeatherFetcher(api_key)
    
    cities = [
        ("London", "UK"),
        ("New York", "US"),
        ("Tokyo", "JP"),
        ("Sydney", "AU")
    ]
    
    for city, country in cities:
        print(f"Fetching weather for {city}, {country}...")
        weather_data = fetcher.get_current_weather(city, country)
        
        if weather_data:
            print(f"Temperature in {weather_data['location']}: {weather_data['temperature']}°C")
            print(f"Conditions: {weather_data['weather']} - {weather_data['description']}")
            print(f"Humidity: {weather_data['humidity']}%")
            print(f"Wind: {weather_data['wind_speed']} m/s\n")
            
            fetcher.save_to_file(weather_data)
        else:
            print(f"Failed to fetch weather data for {city}\n")

if __name__ == "__main__":
    main()