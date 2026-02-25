
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
    main()