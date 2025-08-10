#!/bin/bash

export PYTHONPATH=$(dirname "$0")

python src/main/mcts_train.py
