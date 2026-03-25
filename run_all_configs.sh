#!/bin/bash

# Directory containing the config files
CONFIG_DIR="nature_milling/configs"

# Check if the config directory exists
if [ ! -d "$CONFIG_DIR" ]; then
  echo "Error: Directory '$CONFIG_DIR' not found."
  exit 1
fi

# Loop through all yaml files in the directory
for config_file in "$CONFIG_DIR"/*.yaml; do
  # Check if the file exists to avoid errors if the directory is empty
  if [ -f "$config_file" ]; then
    echo "--------------------------------------------------"
    echo "Running experiment with config: $config_file"
    echo "--------------------------------------------------"
    python run.py "$config_file"
    echo "--------------------------------------------------"
    echo "Finished experiment with config: $config_file"
    echo "--------------------------------------------------"
  fi
done

echo "All experiments complete."
