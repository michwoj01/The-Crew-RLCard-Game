#!/bin/bash

# Set the PYTHONPATH to include the src directory
export PYTHONPATH=$(dirname "$0")/src

# Run uvicorn with the server module
python src/bridge/main.py
