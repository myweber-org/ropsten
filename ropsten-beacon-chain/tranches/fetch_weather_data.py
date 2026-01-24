
import requests
import json
from datetime import datetime
from typing import Optional, Dict, Any

class WeatherFetcher:
    def __init__(self, api_key: str, base_url: str = "http://api.openweathermap.org/data/2.5/weather"):
        self.api_key = api_key
        self.base_url = base_url
        self.session = requests.Session()
    
    def get_weather_by_city(self, city_name: str, country_code: Optional[str] = None) -> Dict[str, Any]:
        query = city_name
        if country_code:
            query = f"{city_name},{country_code}"
        
        params = {
            'q': query,
            'appid': self.api_key,
            'units': 'metric'
        }
        
        try:
            response = self.session.get(self.base_url, params=params, timeout=10)
            response.raise_for_status()
            data = response.json()
            
            return {
                'success': True,
                'city': data['name'],
                'country': data['sys']['country'],
                'temperature': data['main']['temp'],
                'feels_like': data['main']['feels_like'],
                'humidity': data['main']['humidity'],
                'pressure': data['main']['pressure'],
                'weather': data['weather'][0]['description'],
                'wind_speed': data['wind']['speed'],
                'timestamp': datetime.fromtimestamp(data['dt']).isoformat(),
                'raw_data': data
            }
        except requests.exceptions.RequestException as e:
            return {
                'success': False,
                'error': f"Network error: {str(e)}",
                'timestamp': datetime.now().isoformat()
            }
        except (KeyError, json.JSONDecodeError) as e:
            return {
                'success': False,
                'error': f"Data parsing error: {str(e)}",
                'timestamp': datetime.now().isoformat()
            }
    
    def save_to_file(self, data: Dict[str, Any], filename: str = "weather_data.json"):
        try:
            with open(filename, 'w') as f:
                json.dump(data, f, indent=2)
            return True
        except IOError as e:
            print(f"Error saving file: {e}")
            return False

def main():
    api_key = "your_api_key_here"
    fetcher = WeatherFetcher(api_key)
    
    cities = [
        ("London", "UK"),
        ("New York", "US"),
        ("Tokyo", "JP")
    ]
    
    all_weather_data = []
    
    for city, country in cities:
        print(f"Fetching weather for {city}, {country}...")
        weather_data = fetcher.get_weather_by_city(city, country)
        
        if weather_data['success']:
            print(f"  Temperature: {weather_data['temperature']}°C")
            print(f"  Conditions: {weather_data['weather']}")
            all_weather_data.append(weather_data)
        else:
            print(f"  Error: {weather_data['error']}")
    
    if all_weather_data:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"weather_report_{timestamp}.json"
        fetcher.save_to_file(all_weather_data, filename)
        print(f"\nWeather data saved to {filename}")

if __name__ == "__main__":
    main()