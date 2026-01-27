import os
import base64
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
from cryptography.hazmat.primitives import padding
from cryptography.hazmat.backends import default_backend
import hashlib

class FileEncryptor:
    def __init__(self, password: str):
        self.key = self._derive_key(password)
    
    def _derive_key(self, password: str) -> bytes:
        salt = b'static_salt_for_demo'
        key = hashlib.pbkdf2_hmac('sha256', password.encode(), salt, 100000, 32)
        return key
    
    def encrypt_file(self, input_path: str, output_path: str):
        iv = os.urandom(16)
        cipher = Cipher(algorithms.AES(self.key), modes.CBC(iv), backend=default_backend())
        encryptor = cipher.encryptor()
        
        padder = padding.PKCS7(128).padder()
        
        with open(input_path, 'rb') as f_in, open(output_path, 'wb') as f_out:
            f_out.write(iv)
            
            while True:
                chunk = f_in.read(4096)
                if not chunk:
                    break
                padded_data = padder.update(chunk)
                encrypted_chunk = encryptor.update(padded_data)
                f_out.write(encrypted_chunk)
            
            final_padded = padder.finalize()
            final_encrypted = encryptor.update(final_padded) + encryptor.finalize()
            f_out.write(final_encrypted)
    
    def decrypt_file(self, input_path: str, output_path: str):
        with open(input_path, 'rb') as f_in:
            iv = f_in.read(16)
            cipher = Cipher(algorithms.AES(self.key), modes.CBC(iv), backend=default_backend())
            decryptor = cipher.decryptor()
            unpadder = padding.PKCS7(128).unpadder()
            
            with open(output_path, 'wb') as f_out:
                while True:
                    chunk = f_in.read(4096)
                    if not chunk:
                        break
                    decrypted_chunk = decryptor.update(chunk)
                    unpadded_data = unpadder.update(decrypted_chunk)
                    f_out.write(unpadded_data)
                
                final_decrypted = decryptor.finalize()
                final_unpadded = unpadder.update(final_decrypted) + unpadder.finalize()
                f_out.write(final_unpadded)

def main():
    encryptor = FileEncryptor("secure_password123")
    
    test_content = b"This is a secret message for encryption testing."
    with open("test_plain.txt", "wb") as f:
        f.write(test_content)
    
    encryptor.encrypt_file("test_plain.txt", "test_encrypted.bin")
    encryptor.decrypt_file("test_encrypted.bin", "test_decrypted.txt")
    
    with open("test_decrypted.txt", "rb") as f:
        decrypted_content = f.read()
    
    print(f"Original: {test_content}")
    print(f"Decrypted: {decrypted_content}")
    print(f"Match: {test_content == decrypted_content}")
    
    os.remove("test_plain.txt")
    os.remove("test_encrypted.bin")
    os.remove("test_decrypted.txt")

if __name__ == "__main__":
    main()
from cryptography.fernet import Fernet
import os
import sys

class FileEncryptor:
    def __init__(self, key_file='secret.key'):
        self.key_file = key_file
        self.key = None
        self.cipher = None
        
    def generate_key(self):
        self.key = Fernet.generate_key()
        with open(self.key_file, 'wb') as f:
            f.write(self.key)
        print(f"Key generated and saved to {self.key_file}")
        return self.key
    
    def load_key(self):
        if not os.path.exists(self.key_file):
            raise FileNotFoundError(f"Key file {self.key_file} not found")
        
        with open(self.key_file, 'rb') as f:
            self.key = f.read()
        return self.key
    
    def initialize_cipher(self, key=None):
        if key:
            self.key = key
        elif not self.key:
            self.load_key()
        
        self.cipher = Fernet(self.key)
        return self.cipher
    
    def encrypt_file(self, input_file, output_file=None):
        if not self.cipher:
            self.initialize_cipher()
        
        if not os.path.exists(input_file):
            raise FileNotFoundError(f"Input file {input_file} not found")
        
        if not output_file:
            output_file = input_file + '.encrypted'
        
        with open(input_file, 'rb') as f:
            data = f.read()
        
        encrypted_data = self.cipher.encrypt(data)
        
        with open(output_file, 'wb') as f:
            f.write(encrypted_data)
        
        print(f"File encrypted: {output_file}")
        return output_file
    
    def decrypt_file(self, input_file, output_file=None):
        if not self.cipher:
            self.initialize_cipher()
        
        if not os.path.exists(input_file):
            raise FileNotFoundError(f"Input file {input_file} not found")
        
        if not output_file:
            if input_file.endswith('.encrypted'):
                output_file = input_file[:-10]
            else:
                output_file = input_file + '.decrypted'
        
        with open(input_file, 'rb') as f:
            encrypted_data = f.read()
        
        try:
            decrypted_data = self.cipher.decrypt(encrypted_data)
        except Exception as e:
            raise ValueError(f"Decryption failed: {str(e)}")
        
        with open(output_file, 'wb') as f:
            f.write(decrypted_data)
        
        print(f"File decrypted: {output_file}")
        return output_file
    
    def encrypt_string(self, text):
        if not self.cipher:
            self.initialize_cipher()
        
        if isinstance(text, str):
            text = text.encode('utf-8')
        
        encrypted_text = self.cipher.encrypt(text)
        return encrypted_text.decode('utf-8')
    
    def decrypt_string(self, encrypted_text):
        if not self.cipher:
            self.initialize_cipher()
        
        if isinstance(encrypted_text, str):
            encrypted_text = encrypted_text.encode('utf-8')
        
        decrypted_text = self.cipher.decrypt(encrypted_text)
        return decrypted_text.decode('utf-8')

def main():
    if len(sys.argv) < 3:
        print("Usage: python file_encryption_utility.py <encrypt|decrypt> <filename> [output_filename]")
        print("       python file_encryption_utility.py genkey")
        sys.exit(1)
    
    action = sys.argv[1].lower()
    encryptor = FileEncryptor()
    
    try:
        if action == 'genkey':
            encryptor.generate_key()
        elif action == 'encrypt':
            filename = sys.argv[2]
            output_file = sys.argv[3] if len(sys.argv) > 3 else None
            encryptor.encrypt_file(filename, output_file)
        elif action == 'decrypt':
            filename = sys.argv[2]
            output_file = sys.argv[3] if len(sys.argv) > 3 else None
            encryptor.decrypt_file(filename, output_file)
        else:
            print(f"Unknown action: {action}")
            sys.exit(1)
    except Exception as e:
        print(f"Error: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()