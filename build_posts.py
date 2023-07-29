import os
import subprocess

# Directory containing .ipynb files
input_dir = 'submissions'

# Directory to save .md files
output_dir = 'posts'

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

    # Move the image folder to posts
    image_folder = os.path.join(input_dir, ipynb_file.rsplit('.', 1)[0] + '_files')
    if os.path.exists(image_folder):
        new_image_folder = os.path.join(output_dir, ipynb_file.rsplit('.', 1)[0] + '_files')
        os.rename(image_folder, new_image_folder)
