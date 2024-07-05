#!/bin/bash

# Define variables
cu_file="src/test.cu"
executable="src/test"
log_file="logs/test_output.log"

# Compile the C++ file
nvcc -o "$executable" "$cu_file"

# Check if compilation was successful
if [ $? -ne 0 ]; then
    echo "Compilation failed. Exiting."
    exit 1
fi

# Clear the log file
> "$log_file"

# Execute the program 5 times and log the output
for i in {1..100}
do
    echo "Execution $i:" >> "$log_file"
    ./"$executable" >> "$log_file" 2>&1
    echo "------------------------------------" >> "$log_file"
done

echo "Finished executing and logging."