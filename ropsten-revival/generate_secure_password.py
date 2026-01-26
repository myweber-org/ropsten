import secrets
import string

def generate_password(length=12, use_digits=True, use_special=True):
    """
    Generate a secure random password with specified criteria.
    
    Args:
        length (int): Length of the password (default: 12)
        use_digits (bool): Include digits (default: True)
        use_special (bool): Include special characters (default: True)
    
    Returns:
        str: Generated password
    """
    character_pool = string.ascii_letters
    
    if use_digits:
        character_pool += string.digits
    
    if use_special:
        character_pool += string.punctuation
    
    if not character_pool:
        raise ValueError("Character pool cannot be empty")
    
    if length < 4:
        raise ValueError("Password length must be at least 4 characters")
    
    password = ''.join(secrets.choice(character_pool) for _ in range(length))
    
    return password

def validate_password_strength(password):
    """
    Validate password strength based on common criteria.
    
    Args:
        password (str): Password to validate
    
    Returns:
        tuple: (is_valid, message)
    """
    if len(password) < 8:
        return False, "Password must be at least 8 characters long"
    
    has_upper = any(c.isupper() for c in password)
    has_lower = any(c.islower() for c in password)
    has_digit = any(c.isdigit() for c in password)
    has_special = any(c in string.punctuation for c in password)
    
    if not (has_upper and has_lower):
        return False, "Password must contain both uppercase and lowercase letters"
    
    if not has_digit:
        return False, "Password must contain at least one digit"
    
    if not has_special:
        return False, "Password must contain at least one special character"
    
    return True, "Password meets strength requirements"

if __name__ == "__main__":
    # Example usage
    print("Generating secure passwords:")
    
    # Generate a standard password
    password1 = generate_password()
    is_valid1, message1 = validate_password_strength(password1)
    print(f"Standard password: {password1}")
    print(f"Validation: {message1}")
    
    # Generate a longer password with all character types
    password2 = generate_password(length=16, use_digits=True, use_special=True)
    is_valid2, message2 = validate_password_strength(password2)
    print(f"\nStrong password: {password2}")
    print(f"Validation: {message2}")
    
    # Generate a simpler password (no special characters)
    password3 = generate_password(use_special=False)
    is_valid3, message3 = validate_password_strength(password3)
    print(f"\nSimple password: {password3}")
    print(f"Validation: {message3}")