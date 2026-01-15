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