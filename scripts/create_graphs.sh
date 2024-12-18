#!/bin/bash

if [ -z "$1" ]; then
  echo "Error: Experiment name parameter is required."
  echo "Usage: $0 experiment_name"
  exit 1
fi

# Assign the parameter to a variable
exp_name="$1"
counter=0
max_jobs=3

# Determine the Python argument and whether to include the --dis_morph_evo flag
dis_morph_evo_flag=""  # Initialize as empty
default_morph_flag=""

if [[ "$exp_name" == "OurAlgo-MorphEvo-Gen" ]] || [[ "$exp_name" == "OurAlgoNoPart-MorphEvo-Gen" ]] || [[ "$exp_name" == "FullGen-MorphEvo-Gen" ]]; then
  python_arg="generalist"
elif [[ "$exp_name" == "OurAlgo-LargeMorph-Gen" ]]; then
  python_arg="generalist"
  dis_morph_evo_flag="--dis_morph_evo"
elif [[ "$exp_name" == "OurAlgo-DefaultMorph-Gen" ]] || [[ "$exp_name" == "FullGen-DefaultMorph-Gen" ]]; then
  python_arg="generalist"
  dis_morph_evo_flag="--dis_morph_evo"
  default_morph_flag="--default_morph"
elif [[ "$exp_name" == "Spec-MorphEvo" ]] || [[ "$exp_name" == "Spec-MorphEvo-Long" ]]; then
  python_arg="specialist"
elif [[ "$exp_name" == "Spec-DefaultMorph" ]]; then
  python_arg="specialist"
  dis_morph_evo_flag="--dis_morph_evo"
  default_morph_flag="--default_morph"
else
  echo "Error: Experiment name must be one of the experiments."
  exit 1
fi

# Loop through all folders matching the pattern in the "runs/" directory
for dir in runs/$exp_name/"$exp_name"*; do
  if [ -d "$dir" ]; then
    # # Skip directories that contain an underscore followed by 3 or more digits
    # if [[ "$dir" =~ _[0-9]{3,} ]]; then
    #   echo "Skipping $dir"
    #   continue
    # fi
    
    # Run the Python script in the background
    echo $dir
    python graphs.py "$python_arg" $dis_morph_evo_flag $default_morph_flag --run_path "$dir" &
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
