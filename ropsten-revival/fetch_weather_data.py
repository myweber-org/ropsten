import requests
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
    if data is None:
        print("No weather data available.")
        return
    try:
        city = data['name']
        country = data['sys']['country']
        temp = data['main']['temp']
        humidity = data['main']['humidity']
        description = data['weather'][0]['description']
        print(f"Weather in {city}, {country}:")
        print(f"  Temperature: {temp}°C")
        print(f"  Humidity: {humidity}%")
        print(f"  Conditions: {description}")
    except KeyError as e:
        print(f"Unexpected data format: missing key {e}")

if __name__ == "__main__":
    API_KEY = "your_api_key_here"
    city_name = input("Enter city name: ")
    weather_data = get_weather(API_KEY, city_name)
    display_weather(weather_data)import requests
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

def display_weather(weather_data):
    if weather_data is None:
        print("No weather data to display.")
        return

    if weather_data.get('cod') != 200:
        print(f"Error: {weather_data.get('message', 'Unknown error')}")
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
    if len(sys.argv) < 2:
        print("Usage: python fetch_weather_data.py <city_name>")
        print("Please set your OpenWeatherMap API key in the 'API_KEY' variable.")
        sys.exit(1)

    API_KEY = "YOUR_API_KEY_HERE"
    if API_KEY == "YOUR_API_KEY_HERE":
        print("Error: Please replace 'YOUR_API_KEY_HERE' with your actual OpenWeatherMap API key.")
        sys.exit(1)

    city = ' '.join(sys.argv[1:])
    weather_data = get_weather(API_KEY, city)

    if weather_data:
        display_weather(weather_data)

if __name__ == "__main__":
    main()