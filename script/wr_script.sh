#!/bin/bash

cu_file="src/wr_race_nodelay.cu"
executable="object/wr_race_nodelay"
log_file="logs/wr_race_nodelay.log"

nvcc -o "$executable" "$cu_file"

if [ $? -ne 0 ]; then
    echo "Compilation failed. Exiting."
    exit 1
fi

> "$log_file"

total_executions=100

draw_progress_bar() {
    local progress=$1
    local total=$2
    local width=50
    local filled=$((progress * width / total))
    local empty=$((width - filled))

    printf "\r["
    for ((i=0; i<filled; i++)); do printf "#"; done
    for ((i=0; i<empty; i++)); do printf " "; done
    printf "] %d/%d" "$progress" "$total"
}

for i in $(seq 1 "$total_executions")
do
    echo "Execution $i:" >> "$log_file"
    ./"$executable" >> "$log_file" 2>&1
    echo "------------------------------------" >> "$log_file"

    draw_progress_bar "$i" "$total_executions"
done

echo -e "\nFinished executing and logging."
