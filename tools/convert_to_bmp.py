
from PIL import Image
import os
import sys

if len(sys.argv) < 3:
    print("Usage: python3 convert_to_bmp.py <input_dir> <output_dir> [max_images]")
    sys.exit(1)

input_dir = sys.argv[1]
output_dir = sys.argv[2]
max_images = int(sys.argv[3]) if len(sys.argv) > 3 else 5

os.makedirs(output_dir, exist_ok=True)

if not os.path.exists(input_dir):
    print(f"Error: Directory '{input_dir}' not found!")
    sys.exit(1)

all_pngs = sorted([f for f in os.listdir(input_dir) if f.endswith('.png')])
print(f"Found {len(all_pngs)} PNG files in {input_dir}")

png_files = all_pngs[:max_images]
print(f"Converting first {len(png_files)} images to BMP...\n")

for i, png_file in enumerate(png_files, 1):
    print(f"[{i}/{len(png_files)}] Processing: {png_file}")
    
    input_path = os.path.join(input_dir, png_file)
    output_path = os.path.join(output_dir, png_file.replace('.png', '.bmp'))
    
    try:
        img = Image.open(input_path)
        print(f"    Size: {img.size[0]}x{img.size[1]}, Mode: {img.mode}")
        
        if img.mode != 'RGB':
            print(f"    Converting from {img.mode} to RGB...")
            img = img.convert('RGB')
        
        img.save(output_path, 'BMP')
        
        size_mb = os.path.getsize(output_path) / (1024*1024)
        print(f"    ✓ Saved: {os.path.basename(output_path)} ({size_mb:.2f} MB)\n")
    except Exception as e:
        print(f"    ✗ Error: {e}\n")

print("="*60)
print(f"Conversion complete!")
bmp_count = len([f for f in os.listdir(output_dir) if f.endswith('.bmp')])
print(f"{bmp_count} BMP files in {output_dir}/")
print("="*60)
