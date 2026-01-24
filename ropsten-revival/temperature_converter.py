
def celsius_to_fahrenheit(celsius):
    return (celsius * 9/5) + 32

def celsius_to_kelvin(celsius):
    return celsius + 273.15

def fahrenheit_to_celsius(fahrenheit):
    return (fahrenheit - 32) * 5/9

def fahrenheit_to_kelvin(fahrenheit):
    return (fahrenheit - 32) * 5/9 + 273.15

def kelvin_to_celsius(kelvin):
    return kelvin - 273.15

def kelvin_to_fahrenheit(kelvin):
    return (kelvin - 273.15) * 9/5 + 32

def convert_temperature(value, from_unit, to_unit):
    units = ['C', 'F', 'K']
    from_unit = from_unit.upper()
    to_unit = to_unit.upper()
    
    if from_unit not in units or to_unit not in units:
        raise ValueError("Invalid temperature unit. Use 'C', 'F', or 'K'")
    
    if from_unit == to_unit:
        return value
    
    conversion_map = {
        ('C', 'F'): celsius_to_fahrenheit,
        ('C', 'K'): celsius_to_kelvin,
        ('F', 'C'): fahrenheit_to_celsius,
        ('F', 'K'): fahrenheit_to_kelvin,
        ('K', 'C'): kelvin_to_celsius,
        ('K', 'F'): kelvin_to_fahrenheit,
    }
    
    if (from_unit, to_unit) in conversion_map:
        return conversion_map[(from_unit, to_unit)](value)
    
    raise ValueError(f"Conversion from {from_unit} to {to_unit} not implemented")

if __name__ == "__main__":
    print("Temperature Converter")
    print("20°C to Fahrenheit:", convert_temperature(20, 'C', 'F'))
    print("68°F to Celsius:", convert_temperature(68, 'F', 'C'))
    print("100°C to Kelvin:", convert_temperature(100, 'C', 'K'))
    print("300K to Fahrenheit:", convert_temperature(300, 'K', 'F'))