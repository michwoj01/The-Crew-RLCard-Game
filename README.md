# The Crew RLCard Game

## Overview

The Crew RLCard Game is a simulation of The Crew card game, designed for research and experimentation in reinforcement
learning. This project provides a framework for implementing various agents and testing their performance in the game
environment.

## Project Structure

```plaintext
the-crew-rlcard-game
├── src                  # Source code for the project
├── LICENSE              # License file
├── README.md            # Project documentation
├── requirements.txt     # List of project dependencies
└── run_game.sh    # Shell script to run a human game

```

## Setup Instructions

1. After unpacking the ZIP, go to the project directory:

   ```bash
   cd the-crew-rlcard-game
   ```

2. Install the required dependencies:

   ```bash
   pip install -r requirements.txt
   ```

3. Run the game with number of tasks between 1 and 4:

   ```bash
   ./run_game.sh <number_of_tasks>
   ```

4. After game is finished, window will be closed automatically and either:
   `😞 Team lost. Tasks were not completed correctly.` or `🎉 Team won! All tasks completed successfully!` will be
   displayed in the terminal.
   If you want to play again, just run the script again with the same or different number of tasks.

## Game Rules

Game rules are based on the official rules of The Crew card game. For detailed rules, refer to
the [official rulebook](https://gramywplanszowki.pl/storage/games/668/files/zaloga-w-poszukiwaniu-dziewiatej-planety-instrukcja.pdf)

IMPORTANT: The game scenario is to take some tasks with one communication token, no other restrictions apply.

## License

This project is licensed under the Apache License 2.0. See the LICENSE file for more details.
