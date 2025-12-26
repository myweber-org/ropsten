
import requests
import json

def get_weather_data(api_key, city_name):
    base_url = "http://api.openweathermap.org/data/2.5/weather"
    params = {
        'q': city_name,
        'appid': api_key,
        'units': 'metric'
    }
    
    try:
        response = requests.get(base_url, params=params)
        response.raise_for_status()
        weather_data = response.json()
        
        if weather_data.get('cod') != 200:
            print(f"Error: {weather_data.get('message', 'Unknown error')}")
            return None
            
        return {
            'city': weather_data['name'],
            'country': weather_data['sys']['country'],
            'temperature': weather_data['main']['temp'],
            'feels_like': weather_data['main']['feels_like'],
            'humidity': weather_data['main']['humidity'],
            'pressure': weather_data['main']['pressure'],
            'weather': weather_data['weather'][0]['description'],
            'wind_speed': weather_data['wind']['speed']
        }
        
    except requests.exceptions.RequestException as e:
        print(f"Network error occurred: {e}")
        return None
    except KeyError as e:
        print(f"Unexpected data format: {e}")
        return None

def display_weather_info(weather_info):
    if weather_info:
        print(f"Weather in {weather_info['city']}, {weather_info['country']}:")
        print(f"  Temperature: {weather_info['temperature']}°C")
        print(f"  Feels like: {weather_info['feels_like']}°C")
        print(f"  Humidity: {weather_info['humidity']}%")
        print(f"  Pressure: {weather_info['pressure']} hPa")
        print(f"  Conditions: {weather_info['weather']}")
        print(f"  Wind Speed: {weather_info['wind_speed']} m/s")

if __name__ == "__main__":
    API_KEY = "your_api_key_here"
    CITY = "London"
    
    weather_info = get_weather_data(API_KEY, CITY)
    display_weather_info(weather_info)
import requests
import json

def get_weather_data(api_key, city_name):
    base_url = "http://api.openweathermap.org/data/2.5/weather"
    params = {
        'q': city_name,
        'appid': api_key,
        'units': 'metric'
    }
    
    try:
        response = requests.get(base_url, params=params)
        response.raise_for_status()
        weather_data = response.json()
        
        if weather_data.get('cod') != 200:
            print(f"Error: {weather_data.get('message', 'Unknown error')}")
            return None
            
        return {
            'city': weather_data['name'],
            'country': weather_data['sys']['country'],
            'temperature': weather_data['main']['temp'],
            'feels_like': weather_data['main']['feels_like'],
            'humidity': weather_data['main']['humidity'],
            'pressure': weather_data['main']['pressure'],
            'weather': weather_data['weather'][0]['description'],
            'wind_speed': weather_data['wind']['speed']
        }
        
    except requests.exceptions.RequestException as e:
        print(f"Network error occurred: {e}")
        return None
    except (KeyError, json.JSONDecodeError) as e:
        print(f"Data parsing error: {e}")
        return None

def display_weather(weather_info):
    if weather_info:
        print(f"Weather in {weather_info['city']}, {weather_info['country']}:")
        print(f"Temperature: {weather_info['temperature']}°C")
        print(f"Feels like: {weather_info['feels_like']}°C")
        print(f"Humidity: {weather_info['humidity']}%")
        print(f"Pressure: {weather_info['pressure']} hPa")
        print(f"Conditions: {weather_info['weather'].title()}")
        print(f"Wind Speed: {weather_info['wind_speed']} m/s")
    else:
        print("No weather data available.")

if __name__ == "__main__":
    API_KEY = "your_api_key_here"
    CITY = "London"
    
    weather = get_weather_data(API_KEY, CITY)
    display_weather(weather)