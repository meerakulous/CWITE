#!/bin/bash

# List of Python scripts to run
scripts=(
    "proposed.py"
)

# GPU settings: use GPUs 0 through 4
gpus=(0 1 2 3 4 5 6 7)
num_gpus=${#gpus[@]}
gpu_index=0

# CPU core settings
start_core=0
end_core=128
chunk_size=5
core=$start_core

# Index ranges
step=50
max_idx=1000

for script in "${scripts[@]}"; do
    for ((idxmin=0; idxmin<max_idx; idxmin+=step)); do
        idxmax=$((idxmin + step))

        # Assign CPU range
        range_start=$core
        range_end=$((core + chunk_size - 1))

        if [ "$range_end" -gt "$end_core" ]; then
            echo "Reached end of CPU core range. Restarting from $start_core."
            core=$start_core
            range_start=$core
            range_end=$((core + chunk_size - 1))
        fi

        # Assign GPU from list
        gpu_id=${gpus[$gpu_index]}

        # Run the script with taskset and assigned GPU and index range
        echo "Running $script [idxmin=$idxmin, idxmax=$idxmax] on cores $range_start-$range_end with GPU $gpu_id"
        taskset -c "$range_start"-"$range_end" python "$script" --gpu "$gpu_id" --idxmin "$idxmin" --idxmax "$idxmax" &

        # Advance GPU and CPU core pointers
        gpu_index=$(( (gpu_index + 1) % num_gpus ))
        core=$((core + chunk_size))
    done
done

# Wait for all background processes to finish
wait
