#!/bin/bash

if [ -z "$1" ]; then
  echo "Error: Experiment name parameter is required."
  echo "Usage: $0 experiment_name"
  exit 1
fi

exp_name="$1"
max_jobs=3
total_runs=12

run_experiment() {
    echo "Starting $1..."
    python main.py $1
    echo "Finished $1"
}

for ((i=1; i<=total_runs; i++)); do
    # Run the experiment in the background
    run_experiment &
    
    # If the number of running jobs reaches max_jobs, wait for one to finish
    if (( $(jobs -r | wc -l) >= max_jobs )); then
        wait -n  # Wait for the next job to finish before starting a new one
    fi
done

# Wait for any remaining background jobs to finish
wait

echo "All $total_runs runs of experiment1 have been executed."