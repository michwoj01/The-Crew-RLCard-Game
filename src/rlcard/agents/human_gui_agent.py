import queue
import time
import tkinter as tk
from tkinter import ttk, font

import numpy as np

from src.rlcard.envs.action_event import ActionEvent
from src.rlcard.envs.card import CrewCard


def get_suit_color(suit):
    colors = {
        'B': '#0066cc',
        'Y': '#cc9900',
        'P': '#9900cc',
        'G': '#009900',
        'R': '#cc0000'
    }
    return colors.get(suit, 'black')


def sort_actions(action_ids):
    def key_fn(action_id):
        action_str = str(ActionEvent.from_action_id(action_id))

        # Handle non-card actions (signals, skip, etc.)
        if "signal" in action_str.lower():
            return 10, 0  # Put signals at the end
        elif "skip" in action_str.lower():
            return 11, 0  # Put skip after signals

        # Handle regular cards
        suit_order = {'B': 0, 'G': 1, 'Y': 2, 'P': 3, 'R': 4}
        if len(action_str) > 1:
            suit = action_str[0]
            try:
                rank = int(action_str[1:]) if action_str[1:].isdigit() else 0
            except:
                rank = 0
        else:
            suit = ''
            rank = 0
        return suit_order.get(suit, 9), rank

    return sorted(action_ids, key=key_fn)


class CardButton(tk.Button):

    def __init__(self, parent, card_id, action_text="", command=None, **kwargs):
        self.card_id = card_id
        self.action_text = action_text

        if action_text:
            card_str = action_text
        else:
            card_str = str(ActionEvent.from_action_id(card_id))

        fg_color = "#333333"
        # Parse card info and set colors
        if "skip" in card_str.lower():
            bg_color = "#f0f0f0"
            text = "skip"
        elif 'sig' in card_str.lower():
            suit = card_str[6] if card_str else ''
            rank = card_str[7] if card_str else ''
            sig_type = card_str[11] if card_str else ''
            bg_color = get_suit_color(suit)
            text = 'sig ' + rank + ' ' + sig_type
        elif 'task' in card_str.lower():
            suit = card_str[7] if card_str else ''
            rank = card_str[8] if len(card_str) > 6 else ''
            bg_color = get_suit_color(suit)
            text = 'task ' + rank
        else:
            suit = card_str[0] if card_str else ''
            rank = card_str[1:] if len(card_str) > 1 else ''
            bg_color = get_suit_color(suit)
            text = rank

        super().__init__(
            parent,
            text=text,
            command=command,
            highlightbackground=bg_color,
            background=bg_color,
            fg=fg_color,
            font=("Arial", 9),
            width=4,
            height=2,
            relief="raised",
            **kwargs
        )

        # Hover effects
        self.bind("<Enter>", self._on_hover)
        self.bind("<Leave>", self._on_leave)

    def _on_hover(self, event):
        self.config(relief="solid")

    def _on_leave(self, event):
        self.config(relief="raised")


class CrewGameGUI:
    def __init__(self, show_log=True):
        self.signals_labels: dict[int, tk.Label] = {}
        self.tasks_labels: dict[int, tk.Label] = {}
        self.trick_labels: dict[int, tk.Label] = {}
        self.root = tk.Tk()
        self.root.title("The Crew Card Game")
        self.root.geometry("1200x400")  # Made wider to accommodate log
        self.root.configure(bg="#f5f5f5")

        self.selected_action = None
        self.action_queue = queue.Queue()
        self.current_state = None
        self.move_counter = 0

        # No separate game log window
        self.game_log = None

        self.setup_ui()

    def setup_ui(self):
        # Main container
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))

        # Configure grid weights
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        main_frame.columnconfigure(0, weight=2)  # Game area gets more space
        main_frame.columnconfigure(1, weight=1)  # Log gets less space

        # Left side - game content
        left_frame = ttk.Frame(main_frame)
        left_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S), padx=(0, 10))
        left_frame.columnconfigure(0, weight=1)

        # Game state section
        self.setup_game_state_section(left_frame, 0)

        # Hand section
        self.setup_hand_section(left_frame, 1)

        # Actions section
        self.setup_actions_section(left_frame, 2)

        # Status section
        self.status_label = ttk.Label(left_frame, text="Waiting for game state...",
                                      font=("Arial", 12))
        self.status_label.grid(row=3, column=0, pady=(0, 0), sticky=(tk.W, tk.E))

        # Right side - game log
        self.setup_game_log_section(main_frame, 0, 1)

        # Add close handler
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)

    def setup_game_log_section(self, parent, row, column):
        # Game log frame
        log_frame = ttk.LabelFrame(parent, text="🎮 Game Log", padding="10")
        log_frame.grid(row=row, column=column, sticky=(tk.W, tk.E, tk.N, tk.S))
        log_frame.columnconfigure(0, weight=1)
        log_frame.rowconfigure(0, weight=1)

        # Scrollable text area
        text_frame = ttk.Frame(log_frame)
        text_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        text_frame.columnconfigure(0, weight=1)
        text_frame.rowconfigure(0, weight=1)

        self.log_text_widget = tk.Text(text_frame, wrap=tk.WORD, font=("Consolas", 9),
                                       bg="white", fg="black", state=tk.DISABLED,
                                       width=40, height=15)  # Reduced height from default to 15 lines

        log_scrollbar = ttk.Scrollbar(text_frame, orient=tk.VERTICAL, command=self.log_text_widget.yview)
        self.log_text_widget.configure(yscrollcommand=log_scrollbar.set)

        self.log_text_widget.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        log_scrollbar.grid(row=0, column=1, sticky=(tk.N, tk.S))

        # Clear button
        clear_button = ttk.Button(log_frame, text="Clear Log", command=self.clear_log)
        clear_button.grid(row=1, column=0, pady=(10, 0))

        self.move_counter = 0

    def setup_game_state_section(self, parent, row):
        # Game state frame
        game_frame = ttk.LabelFrame(parent, text="Game State", padding="10")
        game_frame.grid(row=row, column=0, sticky=(tk.W, tk.E, tk.N, tk.S), pady=(0, 10))
        game_frame.columnconfigure(0, weight=1, minsize=200)
        game_frame.columnconfigure(1, weight=1, minsize=200)
        game_frame.columnconfigure(2, weight=1, minsize=200)

        # Current trick section
        trick_section = ttk.LabelFrame(game_frame, text="Current Trick", padding="5")
        trick_section.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S), padx=(0, 5))
        self.trick_frame = trick_section
        self.setup_trick_display()

        # Tasks section
        tasks_section = ttk.LabelFrame(game_frame, text="Tasks", padding="5")
        tasks_section.grid(row=0, column=1, sticky=(tk.W, tk.E, tk.N, tk.S), padx=(0, 5))
        self.tasks_frame = tasks_section
        self.setup_tasks_display()

        # Signals section
        signals_section = ttk.LabelFrame(game_frame, text="Signals", padding="5")
        signals_section.grid(row=0, column=2, sticky=(tk.W, tk.E, tk.N, tk.S))
        self.signals_frame = signals_section
        self.setup_signals_display()

    def setup_trick_display(self):
        self.trick_frame.columnconfigure(0, weight=1, minsize=50)
        self.trick_frame.columnconfigure(1, weight=1, minsize=50)
        self.trick_frame.columnconfigure(2, weight=1, minsize=50)
        self.trick_frame.columnconfigure(3, weight=1, minsize=50)
        self.trick_labels.clear()
        for i in range(4):
            player_frame = ttk.Frame(self.trick_frame)
            player_frame.grid(row=0, column=i, padx=5, pady=5, sticky=(tk.W, tk.E))

            player_label = ttk.Label(player_frame, text=f"Player {i}", font=("Arial", 10, "bold"))
            player_label.grid(row=0, column=0)

            card_label = tk.Label(player_frame, text="No card", width=4, height=2, relief="raised", bg="#f0f0f0",
                                  font=("Arial", 9))
            card_label.grid(row=1, column=0, pady=5)
            self.trick_labels[i] = card_label

    def setup_tasks_display(self):
        self.tasks_frame.columnconfigure(0, weight=1, minsize=180)
        self.tasks_labels.clear()
        for i in range(4):
            label = tk.Label(self.tasks_frame, text=f"Player {i}: No tasks",
                             font=("Arial", 10), wraplength=0, anchor="w", justify="left")
            label.grid(row=i, column=0, sticky=(tk.W, tk.E), pady=2)
            self.tasks_labels[i] = label

    def setup_signals_display(self):
        self.signals_frame.columnconfigure(0, weight=1, minsize=180)
        self.signals_labels = {}
        for i in range(4):
            label = tk.Label(self.signals_frame, text=f"Player {i}: No signal",
                             font=("Arial", 10), anchor="w", justify="left")
            label.grid(row=i, column=0, sticky=(tk.W, tk.E), pady=2)
            self.signals_labels[i] = label

    def setup_hand_section(self, parent, row):
        # Hand frame
        hand_frame = ttk.LabelFrame(parent, text="Your Hand", padding="10")
        hand_frame.grid(row=row, column=0, sticky=(tk.W, tk.E), pady=(0, 10))

        # Create a frame for hand cards (no scrolling needed for better display)
        self.hand_cards_frame = ttk.Frame(hand_frame)
        self.hand_cards_frame.grid(row=0, column=0, sticky=(tk.W, tk.E))

        hand_frame.columnconfigure(0, weight=1)

    def setup_actions_section(self, parent, row):
        # Actions frame
        actions_frame = ttk.LabelFrame(parent, text="Available Actions", padding="10")
        actions_frame.grid(row=row, column=0, sticky=(tk.W, tk.E), pady=(0, 10))

        # Simple frame for actions (no scrolling needed, single row)
        self.actions_cards_frame = ttk.Frame(actions_frame)
        self.actions_cards_frame.grid(row=0, column=0, sticky=(tk.W, tk.E))

        actions_frame.columnconfigure(0, weight=1)

    def update_game_state(self, state, player_id=None):
        """Update the GUI with the current game state"""
        self.current_state = state
        raw_obs = state['obs']
        raw_legal_actions = state['legal_actions']
        offset = 0

        # Clear previous displays
        self.clear_hand()
        self.clear_actions()

        # Update hand
        hand = [i for i in range(40) if raw_obs[offset + i] == 1]
        self.display_hand(hand)
        offset += 40

        # Update trick
        self.update_trick_display(raw_obs, offset)
        offset += 4 * 14

        # Update tasks
        self.update_tasks_display(raw_obs, offset)
        offset += 4 * 14 + 40 + 4

        # Update signals (if present)
        if offset + 4 * 17 <= len(raw_obs):
            self.update_signals_display(raw_obs, offset)

        # Display legal actions only if it's the human player's turn
        if player_id == 0:  # Human player
            self.display_legal_actions(raw_legal_actions)
            self.status_label.config(text=f"Your turn - {len(raw_legal_actions)} actions available")
        else:
            self.status_label.config(text=f"Player {player_id}'s turn - waiting for AI...")

        # Force GUI update
        self.root.update()

    def log_player_move(self, player_id, action_text):
        """Log a player's move to the game log"""
        self.move_counter += 1

        # Color coding based on move type
        if "task" in action_text.lower():
            prefix = "🎯"
        elif "signal" in action_text.lower():
            prefix = "📡"
        elif any(suit in action_text for suit in ['B', 'Y', 'P', 'G', 'R']):
            prefix = "🃏"
        else:
            prefix = "➤"

        timestamp = time.strftime("%H:%M:%S")

        self.log_text_widget.config(state=tk.NORMAL)

        # Insert move with formatting
        move_text = f"[{timestamp}] {prefix} Player {player_id}: {action_text}\n"
        self.log_text_widget.insert(tk.END, move_text)

        # Auto-scroll to bottom
        self.log_text_widget.see(tk.END)
        self.log_text_widget.config(state=tk.DISABLED)

    def log_phase_change(self, phase_text):
        """Log a phase change"""
        self.log_text_widget.config(state=tk.NORMAL)
        separator = "─" * 30 + "\n"
        phase_line = f"🔄 {phase_text}\n"
        self.log_text_widget.insert(tk.END, f"\n{separator}{phase_line}{separator}\n")
        self.log_text_widget.see(tk.END)
        self.log_text_widget.config(state=tk.DISABLED)

    def clear_log(self):
        """Clear the log"""
        self.log_text_widget.config(state=tk.NORMAL)
        self.log_text_widget.delete(1.0, tk.END)
        self.log_text_widget.config(state=tk.DISABLED)
        self.move_counter = 0

    def clear_hand(self):
        for widget in self.hand_cards_frame.winfo_children():
            widget.destroy()

    def clear_actions(self):
        for widget in self.actions_cards_frame.winfo_children():
            widget.destroy()

    def display_hand(self, hand_card_ids):
        for widget in self.hand_cards_frame.winfo_children():
            widget.destroy()

        sorted_cards = sort_actions(hand_card_ids)

        max_cols = 10
        for i, card_id in enumerate(sorted_cards):
            row = i // max_cols
            col = i % max_cols

            card_str = str(ActionEvent.from_action_id(card_id))
            suit = card_str[0] if card_str else ''
            rank = card_str[1:] if len(card_str) > 1 else ''

            bg_color = get_suit_color(suit)

            card_label = tk.Label(
                self.hand_cards_frame,
                text=rank,
                width=4,
                height=2,
                relief="raised",
                bg=bg_color,
                fg="white",
                font=("Arial", 9),
                anchor="center"
            )
            card_label.grid(row=row, column=col, padx=2, pady=2)

    def display_legal_actions(self, legal_actions):
        sorted_actions = sort_actions(legal_actions)

        for i, action_id in enumerate(sorted_actions):
            action_text = str(ActionEvent.from_action_id(action_id))
            card_button = CardButton(
                self.actions_cards_frame,
                action_id,
                action_text,
                command=lambda aid=action_id: self.on_action_selected(aid)
            )
            card_button.grid(row=0, column=i, padx=2, pady=2)  # Single row, column=i

    def update_trick_display(self, raw_obs, offset):
        for pid in range(4):
            trick_rep = raw_obs[offset + pid * 14: offset + (pid + 1) * 14]
            rank_part = trick_rep[:9]
            suit_part = trick_rep[9:14]

            card_found = False
            for rank_idx, r in enumerate(rank_part):
                if r == 1:
                    for suit_idx, s in enumerate(suit_part):
                        if s == 1:
                            card_id = rank_idx + 9 * suit_idx
                            card_str = str(ActionEvent.from_action_id(card_id))
                            suit = card_str[0] if card_str else ''
                            rank = card_str[1:] if len(card_str) > 1 else ''

                            bg_color = get_suit_color(suit)
                            self.trick_labels[pid].config(
                                text=rank,
                                fg="white",
                                bg=bg_color
                            )
                            card_found = True
                            break
                if card_found:
                    break

            if not card_found:
                self.trick_labels[pid].config(
                    text="None",
                    fg="gray",
                    bg="#f0f0f0"
                )

    def update_tasks_display(self, raw_obs, offset):
        for pid in range(4):
            task_rep = raw_obs[offset + pid * 14: offset + (pid + 1) * 14]
            rank_part = task_rep[:9]
            suit_part = task_rep[9:13]
            if_taken = task_rep[13]

            tasks = []
            for rank_idx, r in enumerate(rank_part):
                if r == 1:
                    for suit_idx, s in enumerate(suit_part):
                        if s == 1:
                            card_id = rank_idx + 9 * suit_idx
                            card_str = str(ActionEvent.from_action_id(card_id))
                            tasks.append(card_str)

            if tasks:
                label = self.tasks_labels[pid]
                tasks_text = f"Player {pid}: {', '.join(tasks)}"
                first_task_color = get_suit_color(tasks[0][0]) if tasks else "black"
                label.config(text=tasks_text, fg=first_task_color)
                current_font = font.Font(font=label['font'])
                current_font.configure(overstrike=if_taken == 1)
                label.configure(font=current_font)

            else:
                tasks_text = f"Player {pid}: No tasks"
                self.tasks_labels[pid].config(text=tasks_text, fg="gray")

    def update_signals_display(self, raw_obs, offset):
        signal_types = {0: 'Low', 1: 'High', 2: 'Only'}

        for pid in range(4):
            sig_rep = raw_obs[offset + pid * 17: offset + (pid + 1) * 17]
            rank_idx = np.argmax(sig_rep[:9]) if np.any(sig_rep[:9]) else None
            suit_idx = np.argmax(sig_rep[9:14]) if np.any(sig_rep[9:14]) else None
            sig_type_idx = np.argmax(sig_rep[14:17]) if np.any(sig_rep[14:17]) else None

            if rank_idx is not None and suit_idx is not None and sig_type_idx is not None:
                suit = CrewCard.suits[suit_idx] if suit_idx < len(CrewCard.suits) else '?'
                rank = rank_idx + 1
                sig_type = signal_types.get(sig_type_idx, '?')

                signal_text = f"Player {pid}: {suit}{rank} - {sig_type}"
                color = get_suit_color(suit)
                self.signals_labels[pid].config(text=signal_text, fg=color)
            else:
                signal_text = f"Player {pid}: No signal"
                self.signals_labels[pid].config(text=signal_text, fg="gray")

    def on_action_selected(self, action_id):
        self.selected_action = action_id
        self.action_queue.put(action_id)

        self.clear_actions()
        self.status_label.config(text="Action selected - waiting for other players...")
        self.root.update()

    def wait_for_action(self):
        while True:
            self.root.update()
            try:
                return self.action_queue.get_nowait()
            except queue.Empty:
                time.sleep(0.1)
                continue

    def on_closing(self):
        self.root.destroy()


class HumanAgentGUI:

    def __init__(self):
        self.use_raw = False
        self.gui = None
        self.move_buffer = []  # Buffer moves before GUI is created

    def create_gui(self):
        if not self.gui:
            self.gui = CrewGameGUI()
            # Replay buffered moves
            for player_id, action_id in self.move_buffer:
                action_text = str(ActionEvent.from_action_id(action_id))
                self.gui.log_player_move(player_id, action_text)
            self.move_buffer.clear()

    def step(self, state) -> int:
        if not self.gui:
            self.create_gui()

        self.gui.update_game_state(state, player_id=0)
        selected_action_id = self.gui.wait_for_action()
        return selected_action_id

    def update_for_ai_turn(self, state, player_id, action_taken=None):
        if self.gui:
            self.gui.update_game_state(state, player_id=player_id)

            if action_taken is not None:
                action_text = str(ActionEvent.from_action_id(action_taken))
                self.gui.log_player_move(player_id, action_text)

    def log_phase_change(self, phase_text):
        if self.gui:
            self.gui.log_phase_change(phase_text)

    def log_move(self, player_id, action_id):
        if self.gui:
            action_text = str(ActionEvent.from_action_id(action_id))
            self.gui.log_player_move(player_id, action_text)
        else:
            # Buffer the move if GUI isn't ready yet
            self.move_buffer.append((player_id, action_id))

    def update_state_only(self, state):
        if self.gui:
            self.gui.update_game_state(state, player_id=None)

    def eval_step(self, state):
        return self.step(state), {}
