#!/bin/bash

source .venv/bin/activate

# Set the PYTHONPATH to include the src directory
export PYTHONPATH=$(dirname "$0")

# Run uvicorn with the server module
python src/main/dmc_train.py
