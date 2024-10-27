#!/bin/bash

# Check if the required parameter is provided
if [ -z "$1" ]; then
  echo "Error: Experiment name parameter is required."
  echo "Usage: $0 exp1"
  exit 1
fi

# Assign the parameter to a variable
exp_name="$1"
counter=0
max_jobs=4

# Determine whether to use 'generalist' or 'specialist' based on exp_name
if [[ "$exp_name" == "exp1" ]] || [[ "$exp_name" == "exp2" ]] || [[ "$exp_name" == "exp5" ]]; then
  python_arg="generalist"
elif [[ "$exp_name" == "exp3" ]] || [[ "$exp_name" == "exp4" ]]; then
  python_arg="specialist"
else
  echo "Error: Experiment name must be 'exp1', 'exp2', 'exp3', or 'exp4' to determine the Python argument."
  exit 1
fi


# Loop through all folders matching the pattern in the "runs/" directory
for dir in runs/"$exp_name"*; do
  if [ -d "$dir" ]; then
    # Run the Python script in the background
    python graphs.py "$python_arg" --run_path "$dir" &
    ((counter++))
    
    # If we reach the max_jobs limit, wait for all current jobs to finish
    if [[ $counter -ge $max_jobs ]]; then
      wait
      counter=0  # Reset the counter after the batch completes
    fi
  else
    echo "Directory $dir does not exist."
  fi
done

# Wait for any remaining jobs to finish
wait
echo "All scripts executed!"
