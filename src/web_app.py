import sys
from pathlib import Path

# Add the src directory to the Python path
sys.path.append(str(Path(__file__).resolve().parent))

from fastapi import FastAPI, WebSocket
from fastapi.responses import HTMLResponse
from players import HumanCrewPlayer, IntelligentCrewPlayer
from card import CrewCard, Signal

app = FastAPI()

# Initialize players and game state
human_player = HumanCrewPlayer(player_id=1)
ai_player = IntelligentCrewPlayer(player_id=2)
game_state = {"tasks": [], "current_round": [], "communication_logs": []}

@app.get("/")
async def get_index():
    """Serve the index.html file."""
    with open("index.html", "r") as file:
        return HTMLResponse(content=file.read(), media_type="text/html")

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket endpoint for real-time communication."""
    await websocket.accept()
    try:
        while True:
            data = await websocket.receive_json()

            if data["action"] == "play_card":
                card = CrewCard(data["suit"], data["rank"])
                human_player.set_selected_card(card)
                played_card = await human_player.play_card()
                game_state["current_round"].append((human_player.player_id, played_card))
                await websocket.send_json({"status": "card_played", "card": str(played_card)})
                await websocket.send_json({"status": "game_state", "state": game_state})

            elif data["action"] == "choose_task":
                task = CrewCard(data["suit"], data["rank"])
                human_player.set_selected_task(task)
                chosen_task = await human_player.choose_task(game_state["tasks"])
                game_state["tasks"].append((human_player.player_id, chosen_task))
                await websocket.send_json({"status": "task_chosen", "task": str(chosen_task)})
                await websocket.send_json({"status": "game_state", "state": game_state})

            elif data["action"] == "communicate":
                signal = (CrewCard(data["suit"], data["rank"]), Signal(data["signal"]))
                human_player.set_selected_signal(signal)
                communication = await human_player.communicate()
                game_state["communication_logs"].append((human_player.player_id, communication))
                await websocket.send_json({"status": "communicated", "signal": str(communication)})
                await websocket.send_json({"status": "game_state", "state": game_state})

            elif data["action"] == "get_game_state":
                await websocket.send_json({"status": "game_state", "state": game_state})

            elif data["action"] == "get_hand":
                hand = [str(card) for card in human_player.get_hand()]
                await websocket.send_json({"status": "hand", "hand": hand})

            elif data["action"] == "get_communication_logs":
                await websocket.send_json({"status": "communication_logs", "logs": game_state["communication_logs"]})

    except Exception as e:
        await websocket.close()
        print(f"WebSocket closed: {e}")
