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