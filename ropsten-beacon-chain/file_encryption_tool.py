
import os
import sys

class XORCipher:
    def __init__(self, key: str):
        self.key = key.encode('utf-8')
    
    def _xor_operation(self, data: bytes) -> bytes:
        key_length = len(self.key)
        return bytes([data[i] ^ self.key[i % key_length] for i in range(len(data))])
    
    def encrypt_file(self, input_path: str, output_path: str):
        try:
            with open(input_path, 'rb') as f:
                plaintext = f.read()
            
            ciphertext = self._xor_operation(plaintext)
            
            with open(output_path, 'wb') as f:
                f.write(ciphertext)
            
            print(f"File encrypted successfully: {output_path}")
            return True
        except Exception as e:
            print(f"Encryption failed: {e}")
            return False
    
    def decrypt_file(self, input_path: str, output_path: str):
        return self.encrypt_file(input_path, output_path)

def main():
    if len(sys.argv) < 4:
        print("Usage: python file_encryption_tool.py <encrypt|decrypt> <input_file> <output_file>")
        print("Example: python file_encryption_tool.py encrypt secret.txt secret.enc")
        sys.exit(1)
    
    operation = sys.argv[1].lower()
    input_file = sys.argv[2]
    output_file = sys.argv[3]
    
    if not os.path.exists(input_file):
        print(f"Error: Input file '{input_file}' does not exist.")
        sys.exit(1)
    
    key = input("Enter encryption key: ").strip()
    if not key:
        print("Error: Key cannot be empty.")
        sys.exit(1)
    
    cipher = XORCipher(key)
    
    if operation == 'encrypt':
        cipher.encrypt_file(input_file, output_file)
    elif operation == 'decrypt':
        cipher.decrypt_file(input_file, output_file)
    else:
        print(f"Error: Unknown operation '{operation}'. Use 'encrypt' or 'decrypt'.")

if __name__ == "__main__":
    main()
import os
import sys

def xor_cipher(data, key):
    return bytes([b ^ key for b in data])

def encrypt_file(input_path, output_path, key):
    try:
        with open(input_path, 'rb') as f:
            data = f.read()
        encrypted = xor_cipher(data, key)
        with open(output_path, 'wb') as f:
            f.write(encrypted)
        return True
    except Exception as e:
        print(f"Error during encryption: {e}")
        return False

def decrypt_file(input_path, output_path, key):
    return encrypt_file(input_path, output_path, key)

def main():
    if len(sys.argv) < 5:
        print("Usage: python file_encryption_tool.py <encrypt|decrypt> <input_file> <output_file> <key>")
        sys.exit(1)

    operation = sys.argv[1].lower()
    input_file = sys.argv[2]
    output_file = sys.argv[3]

    try:
        key = int(sys.argv[4]) % 256
    except ValueError:
        print("Key must be an integer")
        sys.exit(1)

    if not os.path.exists(input_file):
        print(f"Input file '{input_file}' does not exist")
        sys.exit(1)

    if operation == 'encrypt':
        success = encrypt_file(input_file, output_file, key)
        if success:
            print(f"File encrypted successfully: {output_file}")
    elif operation == 'decrypt':
        success = decrypt_file(input_file, output_file, key)
        if success:
            print(f"File decrypted successfully: {output_file}")
    else:
        print("Operation must be 'encrypt' or 'decrypt'")
        sys.exit(1)

if __name__ == "__main__":
    main()import os
import sys

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
        print(f"Encryption failed: {e}")
        return False

def decrypt_file(input_path, output_path, key):
    return encrypt_file(input_path, output_path, key)

def main():
    if len(sys.argv) < 5:
        print("Usage: python file_encryption_tool.py <encrypt|decrypt> <input_file> <output_file> <key>")
        sys.exit(1)

    operation = sys.argv[1].lower()
    input_file = sys.argv[2]
    output_file = sys.argv[3]
    
    try:
        key = int(sys.argv[4]) % 256
    except ValueError:
        print("Key must be an integer")
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

if __name__ == "__main__":
    main()
import os
import sys

class XORCipher:
    def __init__(self, key: str):
        self.key = key.encode('utf-8')
    
    def _xor_operation(self, data: bytes) -> bytes:
        key_length = len(self.key)
        return bytes([data[i] ^ self.key[i % key_length] for i in range(len(data))])
    
    def encrypt_file(self, input_path: str, output_path: str):
        try:
            with open(input_path, 'rb') as f:
                plaintext = f.read()
            
            ciphertext = self._xor_operation(plaintext)
            
            with open(output_path, 'wb') as f:
                f.write(ciphertext)
            
            print(f"File encrypted successfully: {output_path}")
            return True
        except Exception as e:
            print(f"Encryption failed: {e}")
            return False
    
    def decrypt_file(self, input_path: str, output_path: str):
        return self.encrypt_file(input_path, output_path)

def main():
    if len(sys.argv) != 5:
        print("Usage: python file_encryption_tool.py <encrypt|decrypt> <key> <input_file> <output_file>")
        sys.exit(1)
    
    operation = sys.argv[1].lower()
    key = sys.argv[2]
    input_file = sys.argv[3]
    output_file = sys.argv[4]
    
    if not os.path.exists(input_file):
        print(f"Input file not found: {input_file}")
        sys.exit(1)
    
    cipher = XORCipher(key)
    
    if operation == 'encrypt':
        cipher.encrypt_file(input_file, output_file)
    elif operation == 'decrypt':
        cipher.decrypt_file(input_file, output_file)
    else:
        print("Invalid operation. Use 'encrypt' or 'decrypt'.")
        sys.exit(1)

if __name__ == "__main__":
    main()import os
import sys

def xor_cipher(data, key):
    return bytes([b ^ key for b in data])

def encrypt_file(input_path, output_path, key):
    try:
        with open(input_path, 'rb') as f:
            data = f.read()
        encrypted = xor_cipher(data, key)
        with open(output_path, 'wb') as f:
            f.write(encrypted)
        return True
    except Exception as e:
        print(f"Error: {e}")
        return False

def decrypt_file(input_path, output_path, key):
    return encrypt_file(input_path, output_path, key)

def main():
    if len(sys.argv) != 5:
        print("Usage: python file_encryption_tool.py <encrypt|decrypt> <input_file> <output_file> <key>")
        sys.exit(1)

    operation = sys.argv[1]
    input_file = sys.argv[2]
    output_file = sys.argv[3]
    
    try:
        key = int(sys.argv[4]) % 256
    except ValueError:
        print("Key must be an integer")
        sys.exit(1)

    if not os.path.exists(input_file):
        print(f"Input file '{input_file}' not found")
        sys.exit(1)

    if operation == 'encrypt':
        if encrypt_file(input_file, output_file, key):
            print(f"File encrypted successfully: {output_file}")
        else:
            print("Encryption failed")
    elif operation == 'decrypt':
        if decrypt_file(input_file, output_file, key):
            print(f"File decrypted successfully: {output_file}")
        else:
            print("Decryption failed")
    else:
        print("Operation must be 'encrypt' or 'decrypt'")
        sys.exit(1)

if __name__ == "__main__":
    main()