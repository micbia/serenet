import os
import sys


# Root of all the output directories
root_path = sys.argv[1]

# Initialize an empty string to store the text
all_text = ''

# Walk through all subdirectories
for subdir, dirs, files in os.walk(root_path):
    for file in files:
        # Check if the current file is 'optimization.txt'
        if file == 'optimization.txt':
            # Open the file and append its content to all_text
            with open(os.path.join(subdir, file), 'r') as f:
                all_text += f.read() + '\n'

# Write all_text to a new 'optimization.txt' file in your root of all the output directories
with open(os.path.join(root_path, 'optimization.txt'), 'w') as f:
    f.write(all_text)
