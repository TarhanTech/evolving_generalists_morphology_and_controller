import os
import pickle
import statistics

# Define the base path
base_path = './runs/OurAlgo-MorphEvo-Gen'

p1_sizes = []
p2_sizes = []
p3_sizes = []
# Iterate over each folder in the base path
for folder_name in os.listdir(base_path):
    folder_path = os.path.join(base_path, folder_name)
    
    # Check if it's a directory
    if os.path.isdir(folder_path):
        # Path to the E_var.pkl file
        file_path = os.path.join(folder_path, 'E_var.pkl')
        
        # Check if the file exists
        if os.path.exists(file_path):
            # Load the .pkl file
            with open(file_path, 'rb') as file:
                e = pickle.load(file)
            
            p1_sizes.append(len(e[0]))
            p2_sizes.append(len(e[1]))
            if len(e) > 2:
                p3_sizes.append(len(e[2]))

print(statistics.mean(p1_sizes))
print(statistics.mean(p2_sizes))
print(statistics.mean(p3_sizes))
                            

