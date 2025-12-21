import requests
import json
from datetime import datetime
import logging

class WeatherFetcher:
    def __init__(self, api_key, base_url="http://api.openweathermap.org/data/2.5/weather"):
        self.api_key = api_key
        self.base_url = base_url
        logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
        self.logger = logging.getLogger(__name__)

    def get_weather_by_city(self, city_name, units="metric"):
        params = {
            "q": city_name,
            "appid": self.api_key,
            "units": units
        }
        try:
            response = requests.get(self.base_url, params=params, timeout=10)
            response.raise_for_status()
            data = response.json()
            return self._parse_weather_data(data)
        except requests.exceptions.RequestException as e:
            self.logger.error(f"Request failed for city {city_name}: {e}")
            return None
        except (KeyError, ValueError) as e:
            self.logger.error(f"Data parsing error for city {city_name}: {e}")
            return None

    def _parse_weather_data(self, raw_data):
        parsed = {
            "city": raw_data.get("name"),
            "country": raw_data.get("sys", {}).get("country"),
            "temperature": raw_data.get("main", {}).get("temp"),
            "feels_like": raw_data.get("main", {}).get("feels_like"),
            "humidity": raw_data.get("main", {}).get("humidity"),
            "pressure": raw_data.get("main", {}).get("pressure"),
            "weather_description": raw_data.get("weather", [{}])[0].get("description"),
            "wind_speed": raw_data.get("wind", {}).get("speed"),
            "wind_direction": raw_data.get("wind", {}).get("deg"),
            "visibility": raw_data.get("visibility"),
            "cloudiness": raw_data.get("clouds", {}).get("all"),
            "sunrise": datetime.fromtimestamp(raw_data.get("sys", {}).get("sunrise")).isoformat() if raw_data.get("sys", {}).get("sunrise") else None,
            "sunset": datetime.fromtimestamp(raw_data.get("sys", {}).get("sunset")).isoformat() if raw_data.get("sys", {}).get("sunset") else None,
            "data_timestamp": datetime.fromtimestamp(raw_data.get("dt")).isoformat() if raw_data.get("dt") else None,
            "timezone_offset": raw_data.get("timezone")
        }
        return parsed

    def save_weather_data(self, weather_data, filename="weather_data.json"):
        if weather_data:
            try:
                with open(filename, 'w') as f:
                    json.dump(weather_data, f, indent=2)
                self.logger.info(f"Weather data saved to {filename}")
                return True
            except IOError as e:
                self.logger.error(f"Failed to save data to {filename}: {e}")
                return False
        return False

def main():
    api_key = "your_api_key_here"
    fetcher = WeatherFetcher(api_key)
    
    cities = ["London", "New York", "Tokyo", "Paris", "Sydney"]
    
    for city in cities:
        print(f"Fetching weather for {city}...")
        weather = fetcher.get_weather_by_city(city)
        if weather:
            print(f"Temperature in {weather['city']}: {weather['temperature']}°C")
            print(f"Weather: {weather['weather_description']}")
            print(f"Humidity: {weather['humidity']}%")
            print(f"Wind Speed: {weather['wind_speed']} m/s")
            print("-" * 40)
            
            filename = f"weather_{city.lower().replace(' ', '_')}.json"
            fetcher.save_weather_data(weather, filename)
        else:
            print(f"Failed to fetch weather data for {city}")
            print("-" * 40)

if __name__ == "__main__":
    main()