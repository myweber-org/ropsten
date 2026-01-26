
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
    return base64.urlsafe_b64encode(os.urandom(length)).decode()[:length]