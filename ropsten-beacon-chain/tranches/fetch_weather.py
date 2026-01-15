import requests
import sys
import os

def get_weather(city, api_key):
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
    if data is None:
        print("No weather data available.")
        return
    try:
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
    except KeyError as e:
        print(f"Unexpected data format: missing key {e}")

def main():
    if len(sys.argv) < 2:
        print("Usage: python fetch_weather.py <city_name>")
        sys.exit(1)
    
    city = ' '.join(sys.argv[1:])
    api_key = os.environ.get('OWM_API_KEY')
    
    if not api_key:
        print("Error: OWM_API_KEY environment variable not set.")
        print("Please set your OpenWeatherMap API key as OWM_API_KEY.")
        sys.exit(1)
    
    weather_data = get_weather(city, api_key)
    display_weather(weather_data)

if __name__ == "__main__":
    main()