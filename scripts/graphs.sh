#!/bin/bash

# Running Python scripts in parallel
python graphs.py generalist --run_path runs/exp1_gen_2560.225762710642_23-32-52-065334 &
python graphs.py generalist --run_path runs/exp1_gen_2617.977608658112_16-47 &
python graphs.py generalist --run_path runs/exp1_gen_2833.1786833309798_16-42 &
python graphs.py generalist --run_path runs/exp2_gen_09-24_18-59 &
# Wait for all background jobs to finish
wait
echo "All scripts executed!"