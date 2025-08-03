#!/bin/bash

# add one arg to script
if [ $# -ne 1 ]; then
    echo "Usage: $0 <number_of_tasks>"
    exit 1
fi
NO_TASKS=$1

export PYTHONPATH=$(dirname "$0")

python3 human_train.py $NO_TASKS
