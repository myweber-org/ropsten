
import requests
import json
from datetime import datetime

class WeatherAlert:
    def __init__(self, api_key, city):
        self.api_key = api_key
        self.city = city
        self.base_url = "http://api.openweathermap.org/data/2.5/weather"
        self.alerts = []
        
    def fetch_weather_data(self):
        params = {
            'q': self.city,
            'appid': self.api_key,
            'units': 'metric'
        }
        
        try:
            response = requests.get(self.base_url, params=params)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            print(f"Error fetching weather data: {e}")
            return None
    
    def check_temperature(self, temp):
        if temp > 35:
            return "EXTREME_HEAT", f"High temperature alert: {temp}°C"
        elif temp > 30:
            return "HEAT_WARNING", f"Temperature warning: {temp}°C"
        elif temp < 0:
            return "FREEZE_WARNING", f"Freezing temperature: {temp}°C"
        return None, None
    
    def check_humidity(self, humidity):
        if humidity > 80:
            return "HIGH_HUMIDITY", f"High humidity alert: {humidity}%"
        elif humidity < 20:
            return "LOW_HUMIDITY", f"Low humidity warning: {humidity}%"
        return None, None
    
    def check_wind_speed(self, wind_speed):
        if wind_speed > 20:
            return "HIGH_WIND", f"High wind alert: {wind_speed} m/s"
        return None, None
    
    def analyze_weather(self, weather_data):
        if not weather_data:
            return
        
        main = weather_data.get('main', {})
        wind = weather_data.get('wind', {})
        
        temp = main.get('temp')
        humidity = main.get('humidity')
        wind_speed = wind.get('speed')
        
        checks = [
            self.check_temperature(temp),
            self.check_humidity(humidity),
            self.check_wind_speed(wind_speed)
        ]
        
        current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        for alert_type, message in checks:
            if alert_type and message:
                alert = {
                    'timestamp': current_time,
                    'type': alert_type,
                    'message': message,
                    'temperature': temp,
                    'humidity': humidity,
                    'wind_speed': wind_speed
                }
                self.alerts.append(alert)
                print(f"[{current_time}] ALERT: {message}")
    
    def get_recent_alerts(self, count=5):
        return self.alerts[-count:] if self.alerts else []
    
    def save_alerts_to_file(self, filename='weather_alerts.json'):
        with open(filename, 'w') as f:
            json.dump(self.alerts, f, indent=2)
        print(f"Alerts saved to {filename}")
    
    def monitor_weather(self, interval_minutes=30):
        import time
        
        print(f"Weather monitoring started for {self.city}")
        print("Press Ctrl+C to stop monitoring\n")
        
        try:
            while True:
                weather_data = self.fetch_weather_data()
                self.analyze_weather(weather_data)
                time.sleep(interval_minutes * 60)
        except KeyboardInterrupt:
            print("\nMonitoring stopped")
            if self.alerts:
                self.save_alerts_to_file()

def main():
    API_KEY = "your_api_key_here"
    CITY = "London,UK"
    
    weather_monitor = WeatherAlert(API_KEY, CITY)
    
    # Single check
    weather_data = weather_monitor.fetch_weather_data()
    weather_monitor.analyze_weather(weather_data)
    
    # Show recent alerts
    recent_alerts = weather_monitor.get_recent_alerts(3)
    if recent_alerts:
        print("\nRecent alerts:")
        for alert in recent_alerts:
            print(f"- {alert['message']}")

if __name__ == "__main__":
    main()