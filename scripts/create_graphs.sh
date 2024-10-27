#!/bin/bash

# Check if the required parameter is provided
if [ -z "$1" ]; then
  echo "Error: Experiment name parameter is required."
  echo "Usage: $0 experiment_name"
  exit 1
fi

# Assign the parameter to a variable
exp_name="$1"
counter=0
max_jobs=4

# Determine the Python argument and whether to include the --dis_morph_evo flag
dis_morph_evo_flag=""  # Initialize as empty

if [[ "$exp_name" == "exp1" ]] || [[ "$exp_name" == "exp2" ]]; then
  python_arg="generalist"
elif [[ "$exp_name" == "exp5" ]]; then
  python_arg="generalist"
  dis_morph_evo_flag="--dis_morph_evo"--dis_morp
elif [[ "$exp_name" == "exp3" ]] || [[ "$exp_name" == "exp4" ]]; then
  python_arg="specialist"
else
  echo "Error: Experiment name must be 'exp1', 'exp2', 'exp3', 'exp4', or 'exp5'."
  exit 1
fi

# Loop through all folders matching the pattern in the "runs/" directory
for dir in runs/"$exp_name"*; do
  if [ -d "$dir" ]; then
    # Run the Python script in the background
    echo $dir
    python graphs.py "$python_arg" --run_path "$dir" $dis_morph_evo_flag &
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
