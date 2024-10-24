#!/bin/bash
## HOW TO RUN: ./reset.sh node_list.txt
## node_list.txt should be formatted like:
## type-srnnumber-mlnode,mlnode,mlnode
## type-srnnumber-mlnode (etc)
## if you are using cell, the base station is always the first node

# Get the number of lines in the file
num_lines=$(wc -l < "$1")

# Loop through each line number
for ((i=1; i<=num_lines; i++)); do
  # Read the line using sed
  line=$(sed -n "${i}p" "$1")

  # Split the line into three variables
  node_type=$(echo "$line" | cut -d'-' -f1)
  number=$(echo "$line" | cut -d'-' -f2)
  nodes=$(echo "$line" | cut -d'-' -f3)

  # Add the prefix 'genesys-' to the number portion
  prefixed_number="genesys-$number"

  case "$node_type" in
    cell)
      psswrd="scope"
    ;;
    wifi)
      psswrd="sunflower"
    ;;
    server)
      psswrd="ChangeMe"
    ;;
  esac

  echo "Resetting $prefixed_number"
  pids=$(sshpass -p "$psswrd" ssh "$prefixed_number" "ps aux | grep 'python3 inferenceNode' | grep -v grep | awk '{print \$2}'")
  if [[ $pids ]]; then
    echo "Processes are already running on $prefixed_number with PIDs: $pids. Killing them..."
    # Loop through each PID and kill it
    for pid in $pids; do
        sshpass -p "$psswrd" ssh "$prefixed_number" "kill -9 $pid"
    done
  fi
done
