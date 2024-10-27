#!/bin/bash

# Counter for tracking concurrent jobs
counter=0
max_jobs=4

# Loop through all folders starting with "exp1" in the "runs/" directory
for dir in runs/exp1_gen*; do
  if [ -d "$dir" ]; then
    # Run the Python script in the background
    python graphs.py generalist --run_path "$dir" &
    ((counter++))
    
    # If we reach the max_jobs limit, wait for all current jobs to finish
    if [[ $counter -ge $max_jobs ]]; then
      wait
      counter=0  # Reset the counter after the batch completes
    fi
  fi
done

# Wait for any remaining jobs to finish
wait
echo "All scripts executed!"