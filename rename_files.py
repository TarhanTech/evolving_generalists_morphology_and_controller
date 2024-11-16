import os
import glob

# Set the folder where your files are located
folder_path = './runs'

# Use glob to find files starting with 'exp1_gen'
for file_path in glob.glob(os.path.join(folder_path, 'exp3_spec*')):
    # Extract the directory and the original filename
    directory, filename = os.path.split(file_path)
    # Replace the prefix
    new_filename = filename.replace("exp3_spec", "Spec-MorphEvo-Long", 1)
    # Get the new file path
    new_file_path = os.path.join(directory, new_filename)
    # Rename the file
    os.rename(file_path, new_file_path)
    print(f'Renamed: {file_path} -> {new_file_path}')
