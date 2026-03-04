import os
import base64
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.backends import default_backend

class FileEncryptor:
    def __init__(self, password: str, salt: bytes = None):
        self.password = password.encode()
        self.salt = salt or os.urandom(16)
        self.backend = default_backend()
        
    def _derive_key(self, key_length: int = 32) -> bytes:
        kdf = PBKDF2(
            algorithm=hashes.SHA256(),
            length=key_length,
            salt=self.salt,
            iterations=100000,
            backend=self.backend
        )
        return kdf.derive(self.password)
    
    def encrypt_file(self, input_path: str, output_path: str = None) -> str:
        if not output_path:
            output_path = input_path + '.enc'
        
        with open(input_path, 'rb') as f:
            plaintext = f.read()
        
        key = self._derive_key()
        iv = os.urandom(16)
        
        cipher = Cipher(
            algorithms.AES(key),
            modes.CBC(iv),
            backend=self.backend
        )
        encryptor = cipher.encryptor()
        
        pad_length = 16 - (len(plaintext) % 16)
        padded_data = plaintext + bytes([pad_length] * pad_length)
        
        ciphertext = encryptor.update(padded_data) + encryptor.finalize()
        
        result = self.salt + iv + ciphertext
        
        with open(output_path, 'wb') as f:
            f.write(result)
        
        return output_path
    
    def decrypt_file(self, input_path: str, output_path: str = None) -> str:
        if not output_path:
            if input_path.endswith('.enc'):
                output_path = input_path[:-4]
            else:
                output_path = input_path + '.dec'
        
        with open(input_path, 'rb') as f:
            data = f.read()
        
        self.salt = data[:16]
        iv = data[16:32]
        ciphertext = data[32:]
        
        key = self._derive_key()
        
        cipher = Cipher(
            algorithms.AES(key),
            modes.CBC(iv),
            backend=self.backend
        )
        decryptor = cipher.decryptor()
        
        padded_plaintext = decryptor.update(ciphertext) + decryptor.finalize()
        
        pad_length = padded_plaintext[-1]
        plaintext = padded_plaintext[:-pad_length]
        
        with open(output_path, 'wb') as f:
            f.write(plaintext)
        
        return output_path

def main():
    import sys
    
    if len(sys.argv) < 4:
        print("Usage: python file_encryption_utility.py <encrypt|decrypt> <input_file> <password> [output_file]")
        sys.exit(1)
    
    operation = sys.argv[1].lower()
    input_file = sys.argv[2]
    password = sys.argv[3]
    output_file = sys.argv[4] if len(sys.argv) > 4 else None
    
    encryptor = FileEncryptor(password)
    
    try:
        if operation == 'encrypt':
            result = encryptor.encrypt_file(input_file, output_file)
            print(f"File encrypted successfully: {result}")
        elif operation == 'decrypt':
            result = encryptor.decrypt_file(input_file, output_file)
            print(f"File decrypted successfully: {result}")
        else:
            print("Invalid operation. Use 'encrypt' or 'decrypt'.")
    except Exception as e:
        print(f"Error: {str(e)}")

if __name__ == "__main__":
    main()
import os
import sys

class XORCipher:
    def __init__(self, key: str):
        self.key = key.encode('utf-8')
    
    def _process(self, data: bytes) -> bytes:
        key_length = len(self.key)
        return bytes([data[i] ^ self.key[i % key_length] for i in range(len(data))])
    
    def encrypt_file(self, input_path: str, output_path: str) -> bool:
        try:
            with open(input_path, 'rb') as f:
                plaintext = f.read()
            ciphertext = self._process(plaintext)
            with open(output_path, 'wb') as f:
                f.write(ciphertext)
            return True
        except Exception as e:
            print(f"Encryption error: {e}", file=sys.stderr)
            return False
    
    def decrypt_file(self, input_path: str, output_path: str) -> bool:
        return self.encrypt_file(input_path, output_path)

def main():
    if len(sys.argv) != 5:
        print("Usage: python file_encryption_utility.py <encrypt|decrypt> <key> <input_file> <output_file>")
        sys.exit(1)
    
    operation = sys.argv[1].lower()
    key = sys.argv[2]
    input_file = sys.argv[3]
    output_file = sys.argv[4]
    
    if not os.path.exists(input_file):
        print(f"Error: Input file '{input_file}' not found.")
        sys.exit(1)
    
    cipher = XORCipher(key)
    
    if operation == 'encrypt':
        success = cipher.encrypt_file(input_file, output_file)
        if success:
            print(f"File encrypted successfully: {output_file}")
        else:
            print("Encryption failed.")
    elif operation == 'decrypt':
        success = cipher.decrypt_file(input_file, output_file)
        if success:
            print(f"File decrypted successfully: {output_file}")
        else:
            print("Decryption failed.")
    else:
        print("Error: Operation must be 'encrypt' or 'decrypt'.")
        sys.exit(1)

if __name__ == "__main__":
    main()
import os
import hashlib
from Crypto.Cipher import AES
from Crypto.Protocol.KDF import PBKDF2
from Crypto.Util.Padding import pad, unpad
from Crypto.Random import get_random_bytes

class FileEncryptor:
    def __init__(self, password: str, salt_length: int = 16):
        self.password = password.encode()
        self.salt_length = salt_length

    def derive_key(self, salt: bytes) -> tuple:
        key = PBKDF2(self.password, salt, dkLen=32, count=1000000)
        return key[:16], key[16:]

    def encrypt_file(self, input_path: str, output_path: str) -> bool:
        try:
            with open(input_path, 'rb') as f:
                plaintext = f.read()

            salt = get_random_bytes(self.salt_length)
            key, iv = self.derive_key(salt)

            cipher = AES.new(key, AES.MODE_CBC, iv)
            ciphertext = cipher.encrypt(pad(plaintext, AES.block_size))

            with open(output_path, 'wb') as f:
                f.write(salt + ciphertext)

            return True
        except Exception as e:
            print(f"Encryption failed: {e}")
            return False

    def decrypt_file(self, input_path: str, output_path: str) -> bool:
        try:
            with open(input_path, 'rb') as f:
                data = f.read()

            salt = data[:self.salt_length]
            ciphertext = data[self.salt_length:]

            key, iv = self.derive_key(salt)
            cipher = AES.new(key, AES.MODE_CBC, iv)

            plaintext = unpad(cipher.decrypt(ciphertext), AES.block_size)

            with open(output_path, 'wb') as f:
                f.write(plaintext)

            return True
        except Exception as e:
            print(f"Decryption failed: {e}")
            return False

    def calculate_hash(self, file_path: str) -> str:
        sha256_hash = hashlib.sha256()
        with open(file_path, 'rb') as f:
            for byte_block in iter(lambda: f.read(4096), b""):
                sha256_hash.update(byte_block)
        return sha256_hash.hexdigest()

def main():
    encryptor = FileEncryptor("secure_password_123")
    
    test_file = "test_document.txt"
    encrypted_file = "encrypted.dat"
    decrypted_file = "decrypted_document.txt"

    with open(test_file, 'w') as f:
        f.write("This is a secret document containing sensitive information.")

    print("Original file created")
    
    if encryptor.encrypt_file(test_file, encrypted_file):
        print("File encrypted successfully")
        print(f"Encrypted file hash: {encryptor.calculate_hash(encrypted_file)}")

    if encryptor.decrypt_file(encrypted_file, decrypted_file):
        print("File decrypted successfully")
        print(f"Decrypted file hash: {encryptor.calculate_hash(decrypted_file)}")

    original_hash = encryptor.calculate_hash(test_file)
    decrypted_hash = encryptor.calculate_hash(decrypted_file)

    if original_hash == decrypted_hash:
        print("Hash verification: SUCCESS - Files are identical")
    else:
        print("Hash verification: FAILED - Files differ")

    for file in [test_file, encrypted_file, decrypted_file]:
        if os.path.exists(file):
            os.remove(file)
            print(f"Cleaned up: {file}")

if __name__ == "__main__":
    main()
import os
import sys

def xor_cipher(data, key):
    """Encrypt or decrypt data using XOR cipher."""
    return bytes([b ^ key for b in data])

def encrypt_file(input_path, output_path, key):
    """Encrypt a file using XOR cipher."""
    try:
        with open(input_path, 'rb') as f:
            plaintext = f.read()
        ciphertext = xor_cipher(plaintext, key)
        with open(output_path, 'wb') as f:
            f.write(ciphertext)
        print(f"File encrypted successfully: {output_path}")
        return True
    except Exception as e:
        print(f"Error encrypting file: {e}")
        return False

def decrypt_file(input_path, output_path, key):
    """Decrypt a file using XOR cipher."""
    return encrypt_file(input_path, output_path, key)

def main():
    if len(sys.argv) < 5:
        print("Usage: python file_encryption_utility.py <encrypt|decrypt> <input_file> <output_file> <key>")
        sys.exit(1)

    operation = sys.argv[1].lower()
    input_file = sys.argv[2]
    output_file = sys.argv[3]
    
    try:
        key = int(sys.argv[4]) % 256
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

if __name__ == "__main__":
    main()import os
from cryptography.fernet import Fernet

def generate_key(key_file='secret.key'):
    """Generate and save a new encryption key."""
    key = Fernet.generate_key()
    with open(key_file, 'wb') as f:
        f.write(key)
    return key

def load_key(key_file='secret.key'):
    """Load encryption key from file."""
    return open(key_file, 'rb').read()

def encrypt_file(file_path, key):
    """Encrypt a file using Fernet encryption."""
    fernet = Fernet(key)
    
    with open(file_path, 'rb') as f:
        original_data = f.read()
    
    encrypted_data = fernet.encrypt(original_data)
    
    encrypted_file = file_path + '.encrypted'
    with open(encrypted_file, 'wb') as f:
        f.write(encrypted_data)
    
    return encrypted_file

def decrypt_file(encrypted_file, key):
    """Decrypt an encrypted file."""
    fernet = Fernet(key)
    
    with open(encrypted_file, 'rb') as f:
        encrypted_data = f.read()
    
    decrypted_data = fernet.decrypt(encrypted_data)
    
    original_file = encrypted_file.replace('.encrypted', '.decrypted')
    with open(original_file, 'wb') as f:
        f.write(decrypted_data)
    
    return original_file

def encrypt_string(text, key):
    """Encrypt a string."""
    fernet = Fernet(key)
    return fernet.encrypt(text.encode())

def decrypt_string(encrypted_text, key):
    """Decrypt an encrypted string."""
    fernet = Fernet(key)
    return fernet.decrypt(encrypted_text).decode()

def main():
    """Example usage of the encryption utility."""
    # Generate or load key
    key_file = 'my_secret.key'
    if not os.path.exists(key_file):
        key = generate_key(key_file)
        print(f"Generated new key: {key[:20]}...")
    else:
        key = load_key(key_file)
        print(f"Loaded existing key: {key[:20]}...")
    
    # Example: Encrypt a string
    secret_message = "This is a confidential message"
    encrypted = encrypt_string(secret_message, key)
    print(f"Encrypted string: {encrypted[:50]}...")
    
    # Example: Decrypt the string
    decrypted = decrypt_string(encrypted, key)
    print(f"Decrypted string: {decrypted}")
    
    # Example: File encryption
    test_file = 'test_document.txt'
    if os.path.exists(test_file):
        encrypted_file = encrypt_file(test_file, key)
        print(f"Created encrypted file: {encrypted_file}")
        
        decrypted_file = decrypt_file(encrypted_file, key)
        print(f"Created decrypted file: {decrypted_file}")

if __name__ == "__main__":
    main()
import os
import base64
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
from cryptography.hazmat.primitives import padding
from cryptography.hazmat.backends import default_backend
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
from cryptography.hazmat.primitives import hashes

def derive_key(password, salt):
    kdf = PBKDF2HMAC(
        algorithm=hashes.SHA256(),
        length=32,
        salt=salt,
        iterations=100000,
        backend=default_backend()
    )
    return kdf.derive(password.encode())

def encrypt_file(input_file, output_file, password):
    salt = os.urandom(16)
    key = derive_key(password, salt)
    iv = os.urandom(16)
    
    cipher = Cipher(algorithms.AES(key), modes.CBC(iv), backend=default_backend())
    encryptor = cipher.encryptor()
    
    padder = padding.PKCS7(algorithms.AES.block_size).padder()
    
    with open(input_file, 'rb') as f:
        plaintext = f.read()
    
    padded_data = padder.update(plaintext) + padder.finalize()
    ciphertext = encryptor.update(padded_data) + encryptor.finalize()
    
    with open(output_file, 'wb') as f:
        f.write(salt + iv + ciphertext)

def decrypt_file(input_file, output_file, password):
    with open(input_file, 'rb') as f:
        data = f.read()
    
    salt = data[:16]
    iv = data[16:32]
    ciphertext = data[32:]
    
    key = derive_key(password, salt)
    cipher = Cipher(algorithms.AES(key), modes.CBC(iv), backend=default_backend())
    decryptor = cipher.decryptor()
    
    padded_plaintext = decryptor.update(ciphertext) + decryptor.finalize()
    
    unpadder = padding.PKCS7(algorithms.AES.block_size).unpadder()
    plaintext = unpadder.update(padded_plaintext) + unpadder.finalize()
    
    with open(output_file, 'wb') as f:
        f.write(plaintext)

def main():
    action = input("Enter 'e' to encrypt or 'd' to decrypt: ").strip().lower()
    input_file = input("Enter input file path: ").strip()
    output_file = input("Enter output file path: ").strip()
    password = input("Enter password: ").strip()
    
    if action == 'e':
        encrypt_file(input_file, output_file, password)
        print("Encryption completed successfully.")
    elif action == 'd':
        decrypt_file(input_file, output_file, password)
        print("Decryption completed successfully.")
    else:
        print("Invalid action specified.")

if __name__ == "__main__":
    main()
import os
import sys

def xor_cipher(data: bytes, key: bytes) -> bytes:
    """Encrypt or decrypt data using XOR cipher."""
    return bytes([data[i] ^ key[i % len(key)] for i in range(len(data))])

def process_file(input_path: str, output_path: str, key: str):
    """Read a file, encrypt/decrypt it, and write to output."""
    try:
        with open(input_path, 'rb') as f:
            file_data = f.read()
        
        key_bytes = key.encode('utf-8')
        processed_data = xor_cipher(file_data, key_bytes)
        
        with open(output_path, 'wb') as f:
            f.write(processed_data)
        
        print(f"File processed successfully: {output_path}")
    
    except FileNotFoundError:
        print(f"Error: Input file not found: {input_path}")
        sys.exit(1)
    except Exception as e:
        print(f"Error processing file: {e}")
        sys.exit(1)

def main():
    if len(sys.argv) != 4:
        print("Usage: python file_encryption_utility.py <input_file> <output_file> <key>")
        print("Example: python file_encryption_utility.py secret.txt encrypted.txt mypassword")
        sys.exit(1)
    
    input_file = sys.argv[1]
    output_file = sys.argv[2]
    key = sys.argv[3]
    
    if not os.path.exists(input_file):
        print(f"Error: Input file does not exist: {input_file}")
        sys.exit(1)
    
    process_file(input_file, output_file, key)

if __name__ == "__main__":
    main()