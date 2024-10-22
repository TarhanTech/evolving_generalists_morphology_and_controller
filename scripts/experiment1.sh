#!/bin/bash

max_jobs=3
total_runs=12

run_experiment() {
    echo "Starting experiment1..."
    python main.py experiment1
    echo "Finished experiment1"
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