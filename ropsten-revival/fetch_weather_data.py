
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
        print("Error decoding response")
        return None

def display_weather(data):
    if not data:
        return
        
    city = data['name']
    country = data['sys']['country']
    temp = data['main']['temp']
    feels_like = data['main']['feels_like']
    humidity = data['main']['humidity']
    weather_desc = data['weather'][0]['description']
    wind_speed = data['wind']['speed']
    
    print(f"Weather in {city}, {country}:")
    print(f"  Temperature: {temp}°C (feels like {feels_like}°C)")
    print(f"  Conditions: {weather_desc}")
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
    display_weather(weather_data)

if __name__ == "__main__":
    main()import requests
import json
from datetime import datetime

class WeatherFetcher:
    def __init__(self, api_key, base_url="http://api.openweathermap.org/data/2.5/weather"):
        self.api_key = api_key
        self.base_url = base_url

    def get_weather_by_city(self, city_name, units="metric"):
        params = {
            'q': city_name,
            'appid': self.api_key,
            'units': units
        }
        try:
            response = requests.get(self.base_url, params=params)
            response.raise_for_status()
            return self._parse_weather_data(response.json())
        except requests.exceptions.RequestException as e:
            return {"error": f"Failed to fetch weather data: {e}"}

    def _parse_weather_data(self, data):
        if data.get("cod") != 200:
            return {"error": data.get("message", "Unknown error")}

        main = data.get("main", {})
        weather = data.get("weather", [{}])[0]
        wind = data.get("wind", {})

        return {
            "city": data.get("name"),
            "country": data.get("sys", {}).get("country"),
            "temperature": main.get("temp"),
            "feels_like": main.get("feels_like"),
            "humidity": main.get("humidity"),
            "pressure": main.get("pressure"),
            "weather": weather.get("description"),
            "wind_speed": wind.get("speed"),
            "wind_direction": wind.get("deg"),
            "timestamp": datetime.fromtimestamp(data.get("dt")).isoformat()
        }

def main():
    api_key = "your_api_key_here"
    fetcher = WeatherFetcher(api_key)

    cities = ["London", "New York", "Tokyo", "Paris"]
    for city in cities:
        print(f"\nFetching weather for {city}...")
        weather = fetcher.get_weather_by_city(city)
        
        if "error" in weather:
            print(f"Error: {weather['error']}")
        else:
            print(json.dumps(weather, indent=2))

if __name__ == "__main__":
    main()import requests
import json
from datetime import datetime
from typing import Optional, Dict, Any

class WeatherFetcher:
    def __init__(self, api_key: str, base_url: str = "http://api.openweathermap.org/data/2.5"):
        self.api_key = api_key
        self.base_url = base_url
        self.session = requests.Session()
    
    def get_current_weather(self, city: str, country_code: Optional[str] = None) -> Dict[str, Any]:
        location = f"{city},{country_code}" if country_code else city
        params = {
            'q': location,
            'appid': self.api_key,
            'units': 'metric'
        }
        
        try:
            response = self.session.get(
                f"{self.base_url}/weather",
                params=params,
                timeout=10
            )
            response.raise_for_status()
            data = response.json()
            
            return {
                'timestamp': datetime.now().isoformat(),
                'location': data['name'],
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
                'clouds': data['clouds']['all']
            }
            
        except requests.exceptions.RequestException as e:
            return {
                'error': True,
                'message': f"Network error: {str(e)}",
                'timestamp': datetime.now().isoformat()
            }
        except (KeyError, json.JSONDecodeError) as e:
            return {
                'error': True,
                'message': f"Data parsing error: {str(e)}",
                'timestamp': datetime.now().isoformat()
            }
    
    def get_weather_forecast(self, city: str, days: int = 5) -> Dict[str, Any]:
        params = {
            'q': city,
            'appid': self.api_key,
            'units': 'metric',
            'cnt': days * 8
        }
        
        try:
            response = self.session.get(
                f"{self.base_url}/forecast",
                params=params,
                timeout=10
            )
            response.raise_for_status()
            data = response.json()
            
            forecast = []
            for item in data['list']:
                forecast.append({
                    'datetime': item['dt_txt'],
                    'temperature': item['main']['temp'],
                    'feels_like': item['main']['feels_like'],
                    'humidity': item['main']['humidity'],
                    'weather': item['weather'][0]['main'],
                    'description': item['weather'][0]['description'],
                    'wind_speed': item['wind']['speed']
                })
            
            return {
                'city': data['city']['name'],
                'country': data['city']['country'],
                'forecast_days': days,
                'forecast': forecast,
                'timestamp': datetime.now().isoformat()
            }
            
        except requests.exceptions.RequestException as e:
            return {
                'error': True,
                'message': f"Network error: {str(e)}",
                'timestamp': datetime.now().isoformat()
            }
        except (KeyError, json.JSONDecodeError) as e:
            return {
                'error': True,
                'message': f"Data parsing error: {str(e)}",
                'timestamp': datetime.now().isoformat()
            }

def save_weather_data(data: Dict[str, Any], filename: str = "weather_data.json"):
    try:
        with open(filename, 'w') as f:
            json.dump(data, f, indent=2)
        return True
    except (IOError, TypeError) as e:
        print(f"Error saving data: {e}")
        return False

def display_weather_summary(weather_data: Dict[str, Any]):
    if weather_data.get('error'):
        print(f"Error: {weather_data['message']}")
        return
    
    print(f"Weather in {weather_data['location']}, {weather_data['country']}:")
    print(f"Temperature: {weather_data['temperature']}°C")
    print(f"Feels like: {weather_data['feels_like']}°C")
    print(f"Conditions: {weather_data['weather']} - {weather_data['description']}")
    print(f"Humidity: {weather_data['humidity']}%")
    print(f"Wind: {weather_data['wind_speed']} m/s")

if __name__ == "__main__":
    API_KEY = "your_api_key_here"
    
    fetcher = WeatherFetcher(API_KEY)
    
    current_weather = fetcher.get_current_weather("London", "UK")
    display_weather_summary(current_weather)
    
    if not current_weather.get('error'):
        save_weather_data(current_weather, "london_weather.json")
    
    forecast = fetcher.get_weather_forecast("Tokyo", 3)
    if not forecast.get('error'):
        save_weather_data(forecast, "tokyo_forecast.json")