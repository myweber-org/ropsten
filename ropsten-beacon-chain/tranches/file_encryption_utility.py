
import os
import base64
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
from cryptography.hazmat.backends import default_backend

class SecureFileEncryptor:
    def __init__(self, password: str, salt_length: int = 16):
        self.password = password.encode()
        self.salt_length = salt_length

    def derive_key(self, salt: bytes, iterations: int = 100000) -> bytes:
        kdf = PBKDF2(
            algorithm=hashes.SHA256(),
            length=32,
            salt=salt,
            iterations=iterations,
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

            cipher = Cipher(
                algorithms.AES(key),
                modes.CBC(iv),
                backend=default_backend()
            )
            encryptor = cipher.encryptor()

            padding_length = 16 - (len(plaintext) % 16)
            padded_data = plaintext + bytes([padding_length]) * padding_length
            ciphertext = encryptor.update(padded_data) + encryptor.finalize()

            encrypted_data = salt + iv + ciphertext

            with open(output_path, 'wb') as f:
                f.write(encrypted_data)

            return True
        except Exception:
            return False

    def decrypt_file(self, input_path: str, output_path: str) -> bool:
        try:
            with open(input_path, 'rb') as f:
                encrypted_data = f.read()

            salt = encrypted_data[:self.salt_length]
            iv = encrypted_data[self.salt_length:self.salt_length + 16]
            ciphertext = encrypted_data[self.salt_length + 16:]

            key = self.derive_key(salt)

            cipher = Cipher(
                algorithms.AES(key),
                modes.CBC(iv),
                backend=default_backend()
            )
            decryptor = cipher.decryptor()

            padded_plaintext = decryptor.update(ciphertext) + decryptor.finalize()
            padding_length = padded_plaintext[-1]
            plaintext = padded_plaintext[:-padding_length]

            with open(output_path, 'wb') as f:
                f.write(plaintext)

            return True
        except Exception:
            return False

def generate_secure_password(length: int = 32) -> str:
    return base64.urlsafe_b64encode(os.urandom(length)).decode()[:length]import os
import hashlib
from base64 import b64encode, b64decode
from Crypto.Cipher import AES
from Crypto.Util.Padding import pad, unpad
from Crypto.Random import get_random_bytes

class FileEncryptor:
    def __init__(self, password):
        self.key = hashlib.sha256(password.encode()).digest()
    
    def encrypt_file(self, input_path, output_path=None):
        if not os.path.exists(input_path):
            raise FileNotFoundError(f"Input file not found: {input_path}")
        
        if output_path is None:
            output_path = input_path + '.enc'
        
        iv = get_random_bytes(AES.block_size)
        cipher = AES.new(self.key, AES.MODE_CBC, iv)
        
        with open(input_path, 'rb') as f:
            plaintext = f.read()
        
        ciphertext = cipher.encrypt(pad(plaintext, AES.block_size))
        
        with open(output_path, 'wb') as f:
            f.write(iv + ciphertext)
        
        return output_path
    
    def decrypt_file(self, input_path, output_path=None):
        if not os.path.exists(input_path):
            raise FileNotFoundError(f"Input file not found: {input_path}")
        
        if output_path is None:
            if input_path.endswith('.enc'):
                output_path = input_path[:-4]
            else:
                output_path = input_path + '.dec'
        
        with open(input_path, 'rb') as f:
            data = f.read()
        
        iv = data[:AES.block_size]
        ciphertext = data[AES.block_size:]
        
        cipher = AES.new(self.key, AES.MODE_CBC, iv)
        plaintext = unpad(cipher.decrypt(ciphertext), AES.block_size)
        
        with open(output_path, 'wb') as f:
            f.write(plaintext)
        
        return output_path
    
    @staticmethod
    def generate_random_key():
        return b64encode(get_random_bytes(32)).decode()

def example_usage():
    encryptor = FileEncryptor("secure_password_123")
    
    test_content = b"This is a secret message that needs encryption."
    test_file = "test_secret.txt"
    
    with open(test_file, 'wb') as f:
        f.write(test_content)
    
    try:
        encrypted_file = encryptor.encrypt_file(test_file)
        print(f"Encrypted file created: {encrypted_file}")
        
        decrypted_file = encryptor.decrypt_file(encrypted_file)
        print(f"Decrypted file created: {decrypted_file}")
        
        with open(decrypted_file, 'rb') as f:
            restored_content = f.read()
        
        if test_content == restored_content:
            print("Encryption/decryption successful!")
        else:
            print("Encryption/decryption failed!")
    
    finally:
        for file in [test_file, test_file + '.enc', test_file + '.dec']:
            if os.path.exists(file):
                os.remove(file)

if __name__ == "__main__":
    example_usage()