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
    command="srun -c 2 --gres=gpu:1 --pty python -m hw2.experiments run-exp -n exp1_1 -K $K -L $L -P 2 -H 100 --early-stopping 8 -M 'cnn' -s 42"

    # Execute the command
    eval "$command"
  done
done

# Define the range of values for L and K
L_values=(2 4 8)
K_values=(32 64 128)

# Loop through the values
for K in "${K_values[@]}"; do
  for L in "${L_values[@]}"; do
    # Assemble the command with the current L and K values
    command="srun -c 2 --gres=gpu:1 --pty python -m hw2.experiments run-exp -n exp1_2 -K $K -L $L -P 2 -H 100 --early-stopping 8 -M 'cnn' -s 42"

    # Execute the command
    eval "$command"
  done
done

# Define the range of values for L and K
L_values=(2 3 4)
K_values=(1)

# Loop through the values
for L in "${L_values[@]}"; do
  for K in "${K_values[@]}"; do
    # Assemble the command with the current L and K values
    command="srun -c 2 --gres=gpu:1 --pty python -m hw2.experiments run-exp -n exp1_3 -K 64 128 -L $L -P 2 -H 100 --early-stopping 8 -M 'cnn' -s 42"

    # Execute the command
    eval "$command"
  done
done


# Define the range of values for L and K
L_values=(8 16 32)
K_values=(1)

# Loop through the values
for L in "${L_values[@]}"; do
  for K in "${K_values[@]}"; do
    # Assemble the command with the current L and K values
    command="srun -c 2 --gres=gpu:1 --pty python -m hw2.experiments run-exp -n exp1_4 -K 32 -L $L -P 2 -H 100 --early-stopping 8 -M 'resnet' -s 42"
    eval "$command"
  done
done
  
L_values=(2 4 8)
K_values=(1)
    
for L in "${L_values[@]}"; do
  for K in "${K_values[@]}"; do
     Assemble the command with the current L and K values
    command="srun -c 2 --gres=gpu:1 --pty python -m hw2.experiments run-exp -n exp1_4 -K 64 128 256 -L $L -P 3 -H 100 --early-stopping 8 -M 'resnet' -s 42"

   #  Execute the command
    eval "$command"
  done
done
