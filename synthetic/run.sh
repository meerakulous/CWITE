#!/bin/bash

# List of Python files to run
python_files=(
"proposed.py"
"proposed_sameweight.py"
"proposed_samecluster.py"
"IPCW.py"
"proposed_assignMAX.py"
"deephit.py"
"oracle.py"
"powells.py"
)

# Define total CPU range and chunk size
cpu_start=0
cpu_end=128
cpu_chunk_size=2

# Index range settings
full_range_min=0
full_range_max=1000
chunk_size=125

# Loop control
cpu_index=$cpu_start
gpu=0

for script in "${python_files[@]}"; do
  # Chunked idx range for all scripts
  for ((idx=$full_range_min; idx<$full_range_max; idx+=chunk_size)); do
    idxmin=$idx
    idxmax=$((idx + chunk_size))
    cpu_chunk_end=$((cpu_index + cpu_chunk_size - 1))

    echo "Launching $script with idx $idxmin-$idxmax on GPU $gpu and CPUs $cpu_index-$cpu_chunk_end"
    taskset -c $cpu_index-$cpu_chunk_end python "$script" -gpu $gpu -idxmin $idxmin -idxmax $idxmax &

    # Update CPU and GPU
    cpu_index=$((cpu_index + cpu_chunk_size))
    gpu=$(((gpu + 1) % 8))

    # Prevent exceeding CPU limit
    if [ $cpu_index -gt $cpu_end ]; then
      echo "CPU limit reached. Waiting for jobs to finish..."
      wait
      cpu_index=$cpu_start
    fi
  done
done

# Wait for all background jobs to finish
wait
echo "All tasks completed."
