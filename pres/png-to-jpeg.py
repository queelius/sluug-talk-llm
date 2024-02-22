#!/usr/bin/env python3

# Python code to turn the convert_and_resize function into a command-line tool
from argparse import ArgumentParser
from PIL import Image

def convert_and_resize(input_path, output_path, new_width, quality=95):
    # Open the original image
    with Image.open(input_path) as img:
        # Calculate the aspect ratio
        aspect_ratio = img.width / img.height
        new_height = int(new_width / aspect_ratio)
        
        # Resize the image with antialiasing and save as JPEG
        img_resized = img.resize((new_width, new_height), Image.ANTIALIAS)
        img_resized.convert("RGB").save(output_path, "JPEG", quality=quality)

def main():
    # Initialize argument parser
    parser = ArgumentParser(description="Convert a high-quality PNG to a downsized JPEG.")
    parser.add_argument("input_path", help="Path to the input PNG image file.")
    parser.add_argument("output_path", help="Path to save the downsized output JPEG image.")
    parser.add_argument("new_width", type=int, help="New width for the downsized image.")
    parser.add_argument("--quality", type=int, default=95, help="Quality level for the JPEG image (0-100).")
    
    # Parse arguments
    args = parser.parse_args()
    
    # Perform conversion and resizing
    convert_and_resize(args.input_path, args.output_path, args.new_width, args.quality)

# This allows the script to be run from the command line
if __name__ == "__main__":
    main()

# Note: The code is not run here due to lack of access to the file system. You can run it on your local machine.
# Note: Don't forget to include the definition of the `convert_and_resize` function in the same file.
