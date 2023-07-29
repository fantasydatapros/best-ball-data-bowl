import os
import subprocess
import shutil

# Directory containing .ipynb files
input_dir = 'submissions'

# Directory to save .md files
output_dir = 'posts'

# Directory to save images
img_dir = os.path.join(output_dir, 'img')

# Create the image directory if it doesn't exist
os.makedirs(img_dir, exist_ok=True)

# Get list of all .ipynb files in input_dir
ipynb_files = [f for f in os.listdir(input_dir) if f.endswith('.ipynb')]

# Iterate over .ipynb files and convert each one
for ipynb_file in ipynb_files:
    # Construct full file paths
    input_file = os.path.join(input_dir, ipynb_file)
    # Remove .ipynb extension and add .md extension
    output_file = os.path.join(output_dir, ipynb_file.rsplit('.', 1)[0] + '.md')

    # Run nbconvert command
    subprocess.run(['jupyter', 'nbconvert', '--to', 'markdown', input_file, '--output-dir', output_dir])

    # Move the image folder to img
    image_folder = os.path.join(output_dir, ipynb_file.rsplit('.', 1)[0] + '_files')
    if os.path.exists(image_folder):
        new_image_folder = os.path.join(img_dir, ipynb_file.rsplit('.', 1)[0] + '_files')
        shutil.move(image_folder, new_image_folder)
