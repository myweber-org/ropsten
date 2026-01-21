
from PIL import Image
import sys
import os

def convert_to_grayscale(input_path, output_path=None):
    try:
        img = Image.open(input_path)
        grayscale_img = img.convert('L')
        
        if output_path is None:
            base, ext = os.path.splitext(input_path)
            output_path = f"{base}_grayscale{ext}"
        
        grayscale_img.save(output_path)
        print(f"Grayscale image saved to: {output_path}")
        return True
    except FileNotFoundError:
        print(f"Error: Input file '{input_path}' not found.")
        return False
    except Exception as e:
        print(f"Error processing image: {e}")
        return False

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python image_grayscale_converter.py <input_image> [output_image]")
        sys.exit(1)
    
    input_file = sys.argv[1]
    output_file = sys.argv[2] if len(sys.argv) > 2 else None
    
    convert_to_grayscale(input_file, output_file)