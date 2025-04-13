from game import CrewGame
from players import CrewPlayer
import asyncio
game = CrewGame()
asyncio.run(game.init_game(
    players=[
        CrewPlayer(0),
        CrewPlayer(1),
        CrewPlayer(2),
        CrewPlayer(3)
    ],
    no_missions=4,
    show_hands=True
))
asyncio.run(game.play_game())