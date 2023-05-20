#!/bin/bash

# Setup env
source $HOME/miniconda3/etc/profile.d/conda.sh
conda activate cs236781-hw

# Define the range of values for L and K
L_values=(2 4 8 16)
K_values=(32 64)

# Loop through the values
for K in "${K_values[@]}"; do
  for L in "${L_values[@]}"; do
    # Assemble the command with the current L and K values
    command="srun -c 2 --gres=gpu:1 --pty python -m hw2.experiments run-exp -n exp1_1 -K $K -L $L -P 2 -H 100 --early-stopping 8 -M 'cnn'"

    # Execute the command
    eval "$command"
  done
done
