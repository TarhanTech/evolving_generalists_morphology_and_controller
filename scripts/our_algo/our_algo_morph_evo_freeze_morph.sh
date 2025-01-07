#!/bin/bash

exp_name="OurAlgo-MorphEvo-FreezeMorph#"
max_jobs=3
total_runs=27

run_experiment() {
    echo "Starting $exp_name..."
    python main.py our_algo --freeze_params morphology --freeze_interval 10
    echo "Finished $exp_name"
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

echo "All $total_runs runs of $exp_name have been executed."