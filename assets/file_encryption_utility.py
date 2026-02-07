import os
import base64
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
from cryptography.hazmat.backends import default_backend

class FileEncryptor:
    def __init__(self, password, salt_length=16):
        self.password = password.encode()
        self.salt_length = salt_length
        self.backend = default_backend()

    def derive_key(self, salt):
        kdf = PBKDF2(
            algorithm=hashes.SHA256(),
            length=32,
            salt=salt,
            iterations=100000,
            backend=self.backend
        )
        return kdf.derive(self.password)

    def encrypt_file(self, input_path, output_path):
        salt = os.urandom(self.salt_length)
        key = self.derive_key(salt)
        iv = os.urandom(16)

        cipher = Cipher(algorithms.AES(key), modes.CBC(iv), backend=self.backend)
        encryptor = cipher.encryptor()

        with open(input_path, 'rb') as f:
            plaintext = f.read()

        padding_length = 16 - (len(plaintext) % 16)
        plaintext += bytes([padding_length]) * padding_length

        ciphertext = encryptor.update(plaintext) + encryptor.finalize()

        with open(output_path, 'wb') as f:
            f.write(salt + iv + ciphertext)

        return True

    def decrypt_file(self, input_path, output_path):
        with open(input_path, 'rb') as f:
            data = f.read()

        salt = data[:self.salt_length]
        iv = data[self.salt_length:self.salt_length + 16]
        ciphertext = data[self.salt_length + 16:]

        key = self.derive_key(salt)
        cipher = Cipher(algorithms.AES(key), modes.CBC(iv), backend=self.backend)
        decryptor = cipher.decryptor()

        plaintext = decryptor.update(ciphertext) + decryptor.finalize()

        padding_length = plaintext[-1]
        plaintext = plaintext[:-padding_length]

        with open(output_path, 'wb') as f:
            f.write(plaintext)

        return True

def generate_secure_password(length=32):
    return base64.b64encode(os.urandom(length)).decode()[:length]

if __name__ == "__main__":
    encryptor = FileEncryptor("strong_password_here")
    
    test_data = b"This is a secret message that needs encryption."
    with open("test_plain.txt", "wb") as f:
        f.write(test_data)

    encryptor.encrypt_file("test_plain.txt", "test_encrypted.bin")
    encryptor.decrypt_file("test_encrypted.bin", "test_decrypted.txt")

    with open("test_decrypted.txt", "rb") as f:
        print(f.read())

    os.remove("test_plain.txt")
    os.remove("test_encrypted.bin")
    os.remove("test_decrypted.txt")
import os
import base64
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
from cryptography.hazmat.backends import default_backend

class FileEncryptor:
    def __init__(self, password: str, salt_length: int = 16):
        self.password = password.encode()
        self.salt_length = salt_length

    def derive_key(self, salt: bytes, key_length: int = 32) -> bytes:
        kdf = PBKDF2HMAC(
            algorithm=hashes.SHA256(),
            length=key_length,
            salt=salt,
            iterations=100000,
            backend=default_backend()
        )
        return kdf.derive(self.password)

    def encrypt_file(self, input_path: str, output_path: str) -> bool:
        try:
            with open(input_path, 'rb') as f:
                plaintext = f.read()

            salt = os.urandom(self.salt_length)
            key = self.derive_key(salt)
            iv = os.urandom(16)

            cipher = Cipher(algorithms.AES(key), modes.CBC(iv), backend=default_backend())
            encryptor = cipher.encryptor()

            pad_length = 16 - (len(plaintext) % 16)
            padded_data = plaintext + bytes([pad_length] * pad_length)

            ciphertext = encryptor.update(padded_data) + encryptor.finalize()

            with open(output_path, 'wb') as f:
                f.write(salt + iv + ciphertext)

            return True
        except Exception:
            return False

    def decrypt_file(self, input_path: str, output_path: str) -> bool:
        try:
            with open(input_path, 'rb') as f:
                data = f.read()

            salt = data[:self.salt_length]
            iv = data[self.salt_length:self.salt_length + 16]
            ciphertext = data[self.salt_length + 16:]

            key = self.derive_key(salt)
            cipher = Cipher(algorithms.AES(key), modes.CBC(iv), backend=default_backend())
            decryptor = cipher.decryptor()

            padded_plaintext = decryptor.update(ciphertext) + decryptor.finalize()
            pad_length = padded_plaintext[-1]
            plaintext = padded_plaintext[:-pad_length]

            with open(output_path, 'wb') as f:
                f.write(plaintext)

            return True
        except Exception:
            return False

def main():
    encryptor = FileEncryptor("secure_password_123")
    
    test_data = b"This is a secret message that needs encryption."
    with open("test_plain.txt", "wb") as f:
        f.write(test_data)

    if encryptor.encrypt_file("test_plain.txt", "test_encrypted.bin"):
        print("File encrypted successfully")
    
    if encryptor.decrypt_file("test_encrypted.bin", "test_decrypted.txt"):
        print("File decrypted successfully")

    with open("test_decrypted.txt", "rb") as f:
        decrypted = f.read()
    
    if decrypted == test_data:
        print("Encryption/decryption verified successfully")

    for fname in ["test_plain.txt", "test_encrypted.bin", "test_decrypted.txt"]:
        if os.path.exists(fname):
            os.remove(fname)

if __name__ == "__main__":
    main()
import os
import sys

class XORCipher:
    def __init__(self, key: str):
        self.key = key.encode('utf-8')
    
    def encrypt(self, data: bytes) -> bytes:
        key_length = len(self.key)
        return bytes([data[i] ^ self.key[i % key_length] for i in range(len(data))])
    
    def decrypt(self, data: bytes) -> bytes:
        return self.encrypt(data)

def process_file(input_path: str, output_path: str, key: str, mode: str):
    if not os.path.exists(input_path):
        print(f"Error: Input file '{input_path}' not found.")
        sys.exit(1)
    
    cipher = XORCipher(key)
    
    try:
        with open(input_path, 'rb') as f:
            file_data = f.read()
        
        if mode == 'encrypt':
            processed_data = cipher.encrypt(file_data)
            action = "Encrypted"
        elif mode == 'decrypt':
            processed_data = cipher.decrypt(file_data)
            action = "Decrypted"
        else:
            print("Error: Mode must be 'encrypt' or 'decrypt'.")
            sys.exit(1)
        
        with open(output_path, 'wb') as f:
            f.write(processed_data)
        
        print(f"{action} file saved to: {output_path}")
        
    except Exception as e:
        print(f"Error processing file: {e}")
        sys.exit(1)

def main():
    if len(sys.argv) != 5:
        print("Usage: python file_encryption_utility.py <input_file> <output_file> <key> <mode>")
        print("Mode: 'encrypt' or 'decrypt'")
        sys.exit(1)
    
    input_file = sys.argv[1]
    output_file = sys.argv[2]
    key = sys.argv[3]
    mode = sys.argv[4].lower()
    
    process_file(input_file, output_file, key, mode)

if __name__ == "__main__":
    main()