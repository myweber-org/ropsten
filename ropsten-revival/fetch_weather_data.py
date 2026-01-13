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
            query += f",{country_code}"
        
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
                'sunrise': datetime.fromtimestamp(data['sys']['sunrise']).isoformat(),
                'sunset': datetime.fromtimestamp(data['sys']['sunset']).isoformat()
            }
            
        except requests.exceptions.RequestException as e:
            return {
                'success': False,
                'error': f"Network error: {str(e)}",
                'city': city_name
            }
        except (KeyError, json.JSONDecodeError) as e:
            return {
                'success': False,
                'error': f"Data parsing error: {str(e)}",
                'city': city_name
            }
    
    def get_weather_by_coordinates(self, lat: float, lon: float) -> Dict[str, Any]:
        params = {
            'lat': lat,
            'lon': lon,
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
                'timestamp': datetime.fromtimestamp(data['dt']).isoformat()
            }
            
        except requests.exceptions.RequestException as e:
            return {
                'success': False,
                'error': f"Network error: {str(e)}",
                'coordinates': (lat, lon)
            }
        except (KeyError, json.JSONDecodeError) as e:
            return {
                'success': False,
                'error': f"Data parsing error: {str(e)}",
                'coordinates': (lat, lon)
            }
    
    def format_weather_report(self, weather_data: Dict[str, Any]) -> str:
        if not weather_data['success']:
            return f"Error fetching weather data: {weather_data['error']}"
        
        report_lines = [
            f"Weather Report for {weather_data['city']}, {weather_data['country']}",
            f"Temperature: {weather_data['temperature']}°C (Feels like: {weather_data['feels_like']}°C)",
            f"Conditions: {weather_data['weather'].title()}",
            f"Humidity: {weather_data['humidity']}%",
            f"Pressure: {weather_data['pressure']} hPa",
            f"Wind Speed: {weather_data['wind_speed']} m/s",
            f"Report Time: {weather_data['timestamp']}"
        ]
        
        if 'sunrise' in weather_data and 'sunset' in weather_data:
            report_lines.extend([
                f"Sunrise: {weather_data['sunrise']}",
                f"Sunset: {weather_data['sunset']}"
            ])
        
        return "\n".join(report_lines)

def main():
    api_key = "your_api_key_here"
    fetcher = WeatherFetcher(api_key)
    
    test_cities = ["London", "New York", "Tokyo"]
    
    for city in test_cities:
        print(f"\nFetching weather for {city}...")
        weather_data = fetcher.get_weather_by_city(city)
        report = fetcher.format_weather_report(weather_data)
        print(report)

if __name__ == "__main__":
    main()