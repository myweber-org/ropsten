import sys
import os

def xor_cipher(data, key):
    """XOR cipher for encryption and decryption."""
    return bytes([b ^ key for b in data])

def encrypt_file(input_path, output_path, key):
    """Encrypt a file using XOR cipher."""
    try:
        with open(input_path, 'rb') as f:
            data = f.read()
        encrypted = xor_cipher(data, key)
        with open(output_path, 'wb') as f:
            f.write(encrypted)
        print(f"File encrypted successfully: {output_path}")
    except Exception as e:
        print(f"Error encrypting file: {e}")

def decrypt_file(input_path, output_path, key):
    """Decrypt a file using XOR cipher."""
    encrypt_file(input_path, output_path, key)  # XOR is symmetric

def main():
    if len(sys.argv) != 5:
        print("Usage: python file_encryptor.py <encrypt|decrypt> <input_file> <output_file> <key>")
        sys.exit(1)

    operation = sys.argv[1]
    input_file = sys.argv[2]
    output_file = sys.argv[3]

    try:
        key = int(sys.argv[4])
        if not 0 <= key <= 255:
            raise ValueError
    except ValueError:
        print("Key must be an integer between 0 and 255")
        sys.exit(1)

    if not os.path.exists(input_file):
        print(f"Input file does not exist: {input_file}")
        sys.exit(1)

    if operation == 'encrypt':
        encrypt_file(input_file, output_file, key)
    elif operation == 'decrypt':
        decrypt_file(input_file, output_file, key)
    else:
        print("Operation must be 'encrypt' or 'decrypt'")
        sys.exit(1)

if __name__ == "__main__":
    main()import sys
import os

def xor_cipher(data, key):
    return bytes([b ^ key for b in data])

def encrypt_file(input_path, output_path, key):
    try:
        with open(input_path, 'rb') as f:
            data = f.read()
        encrypted = xor_cipher(data, key)
        with open(output_path, 'wb') as f:
            f.write(encrypted)
        print(f"File encrypted successfully: {output_path}")
        return True
    except Exception as e:
        print(f"Error: {e}")
        return False

def decrypt_file(input_path, output_path, key):
    return encrypt_file(input_path, output_path, key)

def main():
    if len(sys.argv) < 4:
        print("Usage: python file_encryptor.py <encrypt|decrypt> <input_file> <output_file> [key]")
        print("If key not provided, default key 42 will be used.")
        sys.exit(1)

    action = sys.argv[1].lower()
    input_file = sys.argv[2]
    output_file = sys.argv[3]
    key = int(sys.argv[4]) if len(sys.argv) > 4 else 42

    if not os.path.exists(input_file):
        print(f"Input file does not exist: {input_file}")
        sys.exit(1)

    if action == 'encrypt':
        encrypt_file(input_file, output_file, key)
    elif action == 'decrypt':
        decrypt_file(input_file, output_file, key)
    else:
        print("Invalid action. Use 'encrypt' or 'decrypt'.")

if __name__ == "__main__":
    main()