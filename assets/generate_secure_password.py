
import secrets
import string

def generate_password(length=16, use_uppercase=True, use_lowercase=True, use_digits=True, use_special=True):
    if length < 4:
        raise ValueError("Password length must be at least 4 characters")
    
    character_pool = ""
    if use_uppercase:
        character_pool += string.ascii_uppercase
    if use_lowercase:
        character_pool += string.ascii_lowercase
    if use_digits:
        character_pool += string.digits
    if use_special:
        character_pool += string.punctuation
    
    if not character_pool:
        raise ValueError("At least one character set must be selected")
    
    password_chars = []
    
    if use_uppercase:
        password_chars.append(secrets.choice(string.ascii_uppercase))
    if use_lowercase:
        password_chars.append(secrets.choice(string.ascii_lowercase))
    if use_digits:
        password_chars.append(secrets.choice(string.digits))
    if use_special:
        password_chars.append(secrets.choice(string.punctuation))
    
    remaining_length = length - len(password_chars)
    for _ in range(remaining_length):
        password_chars.append(secrets.choice(character_pool))
    
    secrets.SystemRandom().shuffle(password_chars)
    return ''.join(password_chars)

if __name__ == "__main__":
    print("Generated password:", generate_password())
    print("Short numeric PIN:", generate_password(6, use_uppercase=False, use_lowercase=False, use_special=False))
    print("Complex password:", generate_password(20, use_uppercase=True, use_lowercase=True, use_digits=True, use_special=True))