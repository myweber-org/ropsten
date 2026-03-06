import os
import hashlib
from base64 import b64encode, b64decode
from Crypto.Cipher import AES
from Crypto.Util.Padding import pad, unpad
from Crypto.Random import get_random_bytes

class FileEncryptor:
    def __init__(self, password: str):
        self.key = hashlib.sha256(password.encode()).digest()
    
    def encrypt_file(self, input_path: str, output_path: str = None) -> str:
        if not os.path.exists(input_path):
            raise FileNotFoundError(f"Input file not found: {input_path}")
        
        if output_path is None:
            output_path = input_path + '.enc'
        
        iv = get_random_bytes(AES.block_size)
        cipher = AES.new(self.key, AES.MODE_CBC, iv)
        
        with open(input_path, 'rb') as f_in:
            plaintext = f_in.read()
        
        ciphertext = cipher.encrypt(pad(plaintext, AES.block_size))
        
        with open(output_path, 'wb') as f_out:
            f_out.write(iv + ciphertext)
        
        return output_path
    
    def decrypt_file(self, input_path: str, output_path: str = None) -> str:
        if not os.path.exists(input_path):
            raise FileNotFoundError(f"Input file not found: {input_path}")
        
        if output_path is None:
            if input_path.endswith('.enc'):
                output_path = input_path[:-4]
            else:
                output_path = input_path + '.dec'
        
        with open(input_path, 'rb') as f_in:
            data = f_in.read()
        
        iv = data[:AES.block_size]
        ciphertext = data[AES.block_size:]
        
        cipher = AES.new(self.key, AES.MODE_CBC, iv)
        plaintext = unpad(cipher.decrypt(ciphertext), AES.block_size)
        
        with open(output_path, 'wb') as f_out:
            f_out.write(plaintext)
        
        return output_path
    
    def encrypt_string(self, plaintext: str) -> str:
        iv = get_random_bytes(AES.block_size)
        cipher = AES.new(self.key, AES.MODE_CBC, iv)
        ciphertext = cipher.encrypt(pad(plaintext.encode(), AES.block_size))
        return b64encode(iv + ciphertext).decode()
    
    def decrypt_string(self, encrypted_data: str) -> str:
        data = b64decode(encrypted_data)
        iv = data[:AES.block_size]
        ciphertext = data[AES.block_size:]
        cipher = AES.new(self.key, AES.MODE_CBC, iv)
        plaintext = unpad(cipher.decrypt(ciphertext), AES.block_size)
        return plaintext.decode()

def generate_secure_password(length: int = 32) -> str:
    return b64encode(get_random_bytes(length)).decode()[:length]

if __name__ == "__main__":
    encryptor = FileEncryptor("secure_password_123")
    
    test_string = "Sensitive data that needs protection"
    encrypted = encryptor.encrypt_string(test_string)
    decrypted = encryptor.decrypt_string(encrypted)
    
    print(f"Original: {test_string}")
    print(f"Encrypted: {encrypted}")
    print(f"Decrypted: {decrypted}")
    print(f"Match: {test_string == decrypted}")
    
    print(f"\nSecure password: {generate_secure_password()}")
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

    def derive_key(self, salt: bytes) -> bytes:
        kdf = PBKDF2HMAC(
            algorithm=hashes.SHA256(),
            length=32,
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
        print("Encryption successful")
    
    if encryptor.decrypt_file("test_encrypted.bin", "test_decrypted.txt"):
        print("Decryption successful")

    with open("test_decrypted.txt", "rb") as f:
        decrypted = f.read()
        print(f"Decrypted content matches: {decrypted == test_data}")

    for file in ["test_plain.txt", "test_encrypted.bin", "test_decrypted.txt"]:
        if os.path.exists(file):
            os.remove(file)

if __name__ == "__main__":
    main()
import os
import base64
from Crypto.Cipher import AES
from Crypto.Protocol.KDF import PBKDF2
from Crypto.Util.Padding import pad, unpad
from Crypto.Random import get_random_bytes

class FileEncryptor:
    def __init__(self, password: str, salt_length: int = 16):
        self.password = password.encode()
        self.salt_length = salt_length

    def derive_key(self, salt: bytes) -> bytes:
        return PBKDF2(self.password, salt, dkLen=32, count=1000000)

    def encrypt_file(self, input_path: str, output_path: str) -> bool:
        try:
            with open(input_path, 'rb') as f:
                plaintext = f.read()

            salt = get_random_bytes(self.salt_length)
            key = self.derive_key(salt)
            iv = get_random_bytes(AES.block_size)

            cipher = AES.new(key, AES.MODE_CBC, iv)
            ciphertext = cipher.encrypt(pad(plaintext, AES.block_size))

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
            iv = data[self.salt_length:self.salt_length + AES.block_size]
            ciphertext = data[self.salt_length + AES.block_size:]

            key = self.derive_key(salt)
            cipher = AES.new(key, AES.MODE_CBC, iv)
            plaintext = unpad(cipher.decrypt(ciphertext), AES.block_size)

            with open(output_path, 'wb') as f:
                f.write(plaintext)

            return True
        except Exception:
            return False

    def encrypt_string(self, plaintext: str) -> str:
        salt = get_random_bytes(self.salt_length)
        key = self.derive_key(salt)
        iv = get_random_bytes(AES.block_size)

        cipher = AES.new(key, AES.MODE_CBC, iv)
        ciphertext = cipher.encrypt(pad(plaintext.encode(), AES.block_size))

        combined = salt + iv + ciphertext
        return base64.b64encode(combined).decode()

    def decrypt_string(self, encrypted_data: str) -> str:
        data = base64.b64decode(encrypted_data)
        salt = data[:self.salt_length]
        iv = data[self.salt_length:self.salt_length + AES.block_size]
        ciphertext = data[self.salt_length + AES.block_size:]

        key = self.derive_key(salt)
        cipher = AES.new(key, AES.MODE_CBC, iv)
        plaintext = unpad(cipher.decrypt(ciphertext), AES.block_size)

        return plaintext.decode()

def example_usage():
    encryptor = FileEncryptor("secure_password_123")

    # File encryption example
    with open('test.txt', 'w') as f:
        f.write("Sensitive data that needs protection")

    encryptor.encrypt_file('test.txt', 'test.enc')
    encryptor.decrypt_file('test.enc', 'test_decrypted.txt')

    # String encryption example
    secret_message = "Confidential information"
    encrypted = encryptor.encrypt_string(secret_message)
    decrypted = encryptor.decrypt_string(encrypted)

    print(f"Original: {secret_message}")
    print(f"Encrypted: {encrypted}")
    print(f"Decrypted: {decrypted}")

    # Cleanup
    for file in ['test.txt', 'test.enc', 'test_decrypted.txt']:
        if os.path.exists(file):
            os.remove(file)

if __name__ == "__main__":
    example_usage()