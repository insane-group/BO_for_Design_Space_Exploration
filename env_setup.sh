#!/bin/bash

if [ "$#" -ne 2 ]; then
    echo "Usage: $0 <environment_name> <conda or mamba>"
    exit 1
fi

echo "Setting up environment."

$2 create --name $1 --file ./requirements.txt -y || { echo "Failed wile creating the mamba environment."; exit 1; }

$2 run -n $1 pip install xgboost                                     || { echo "Failed wile installing var."; exit 1; }

echo "Finished setting up environment."
