#!/bin/bash

# Define the list of models, datasets, and losses to loop through
# models=("MNIST" "CIFAR10" "CIFAR100")  # Replace with your models
datasets=("MNIST" "CIFAR10" "CIFAR100")  # Replace with your datasets
losses=("entmax" "sparsemax" "softmax")  # Replace with your loss functions
seeds=(23 05 19 95 42)  # Replace with your seeds
# Define the other optional parameters
epochs=100  # Default number of epochs
patience=3  # Default patience
model="cnn"

# Loop through each combination of model, dataset, and loss
for seed in "${seeds[@]}"
do
    for dataset in "${datasets[@]}"
    do
        for loss in "${losses[@]}"
        do
            # Create a unique save filename based on model, dataset, and loss
            save_filename="${model}_${dataset}_${loss}_${seed}_model.pth"

            # Run the Python script with the current arguments
            python example_usage/train.py "$model" "$dataset" "$loss" "$save_filename" --seed "$seed" --epochs "$epochs" --patience "$patience"

            # Optionally print a message after each run (for debugging/logging purposes)
            echo "Run with model=$model, dataset=$dataset, loss=$loss completed."
        done
    done
done
