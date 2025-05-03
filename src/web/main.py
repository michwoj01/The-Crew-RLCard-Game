from fastapi import FastAPI, WebSocket
from fastapi.responses import HTMLResponse
from web.game import HumanCrewGame
from web.players import CrewPlayer
from web.players import HumanCrewPlayer, IntelligentCrewPlayer
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

app = FastAPI()
game = None


@app.get("/")
async def get():
    with open("index.html") as f:
        return HTMLResponse(f.read())


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    global game

    # Receive player configuration
    config = await websocket.receive_json()
    player_types = config["player_types"]

    # Initialize players
    players = [HumanCrewPlayer(0, websocket)]
    for i, player_type in enumerate(player_types):
        if player_type == "intelligent":
            players.append(IntelligentCrewPlayer(i + 1))
        else:
            players.append(CrewPlayer(i + 1))

    # Initialize game
    game = HumanCrewGame()
    # Prepare the game
    await game.init_game(players=players, no_missions=4, show_hands=True)
    # Start the game
    await game.play_game()
