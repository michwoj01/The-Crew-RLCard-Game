import os
import platform
import queue
import time
import tkinter as tk
from datetime import datetime
from tkinter import ttk, font, messagebox, filedialog

import numpy as np

from action_event import ActionEvent
from card import CrewCard
from player import CrewPlayer


def get_suit_color(suit):
    colors = {
        'B': '#0066cc',
        'Y': '#cc9900',
        'P': '#9900cc',
        'G': '#009900',
        'R': '#000000'
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

        # Parse card info and set colors
        if platform.system() == 'Darwin':
            fg_color = 'black'
        else:
            fg_color = 'white'
        if "skip" in card_str.lower():
            bg_color = "#404040"
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
            width=5,
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
    def __init__(self, show_log=True, log_file_path=None):
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

        # File logging setup
        self.log_file_path = log_file_path
        self.log_file = None
        self.auto_save_enabled = True

        if self.log_file_path:
            self.setup_log_file()

        # No separate game log window
        self.game_log = None

        self.setup_ui()

    def setup_log_file(self):
        try:
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(self.log_file_path), exist_ok=True)

            self.log_file = open(self.log_file_path, 'w', encoding='utf-8')

            # Write header
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            self.log_file.write(f"=== The Crew Game Log ===\n")
            self.log_file.write(f"Started: {timestamp}\n")
            self.log_file.write(f"{'=' * 50}\n\n")
            self.log_file.flush()

        except Exception as e:
            messagebox.showerror("Error", f"Could not create log file: {e}")
            self.log_file = None

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

    def write_to_file(self, text):
        """Write text to the log file if enabled"""
        if self.log_file and self.auto_save_enabled:
            try:
                self.log_file.write(text)
                self.log_file.flush()  # Ensure immediate write
            except Exception as e:
                print(f"Error writing to log file: {e}")

    def update_game_state(self, state, player_id=None):
        """Update the GUI with the current game state"""
        self.current_state = state
        raw_obs = state['obs']
        raw_legal_actions = state['legal_actions']
        offset = 0

        self.clear_actions()

        if player_id is not None:
            # Clear previous displays
            self.clear_hand()

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

        # Write to file
        file_text = f"[{timestamp}] Move {self.move_counter:03d} - Player {player_id}: {action_text}\n"
        self.write_to_file(file_text)

        # Auto-scroll to bottom
        self.log_text_widget.see(tk.END)
        self.log_text_widget.config(state=tk.DISABLED)

    def log_game_result(self, team_won):
        timestamp = time.strftime("%H:%M:%S")

        # Determine result text and emoji
        if team_won:
            result_emoji = "🎉"
            result_text = "TEAM VICTORY!"
            detailed_text = "All tasks completed successfully!"
        else:
            result_emoji = "😞"
            result_text = "TEAM DEFEAT"
            detailed_text = "Tasks were not completed correctly."

        self.log_text_widget.config(state=tk.NORMAL)
        separator = "═" * 40 + "\n"
        result_line = f"{result_emoji} {result_text} {result_emoji}\n"
        detail_line = f"📊 {detailed_text}\n"

        display_text = f"\n{separator}{result_line}{detail_line}"
        display_text += separator + "\n"

        self.log_text_widget.insert(tk.END, display_text)
        self.log_text_widget.see(tk.END)
        self.log_text_widget.config(state=tk.DISABLED)

        # Write to file
        file_text = f"\n[{timestamp}] ========== GAME RESULT ==========\n"
        file_text += f"[{timestamp}] {result_text}: {detailed_text}\n"
        file_text += f"[{timestamp}] Total moves in game: {self.move_counter}\n"
        file_text += f"[{timestamp}] ===============================\n"

        self.write_to_file(file_text)

    def show_end_popup(self, team_won: bool):
        bg_color = '#006600' if team_won else '#660000'
        result_text = " TEAM VICTORY! " if team_won else " TEAM DEFEAT "

        popup = tk.Toplevel(self.root)
        popup.overrideredirect(True)
        popup.attributes('-topmost', True)
        popup.configure(bg=bg_color)

        self.root.update_idletasks()
        w = 400;
        h = 200
        x = self.root.winfo_x() + (self.root.winfo_width() - w) // 2
        y = self.root.winfo_y() + (self.root.winfo_height() - h) // 2
        popup.geometry(f"{w}x{h}+{x}+{y}")

        lbl = tk.Label(
            popup,
            text=result_text,
            font=("Arial", 36, "bold"),
            fg="white",
            bg=bg_color
        )
        lbl.pack(expand=True, fill="both")

        def close_all():
            try:
                popup.destroy()
            except:
                pass
            self.root.destroy()

        popup.after(5000, close_all)

    def log_player_hands(self, players: list[CrewPlayer]):
        for player in players:
            player_id = player.player_id
            hand_cards = [str(card) for card in player.hand]
            file_text = f"Player {player_id} Hand: {', '.join(hand_cards)}\n"
            self.write_to_file(file_text)

    def clear_log(self):
        """Clear the log"""
        self.log_text_widget.config(state=tk.NORMAL)
        self.log_text_widget.delete(1.0, tk.END)
        self.log_text_widget.config(state=tk.DISABLED)
        self.move_counter = 0

    def save_log_as(self):
        """Save the current log to a file"""
        if not self.log_text_widget.get(1.0, tk.END).strip():
            messagebox.showwarning("Warning", "No log content to save!")
            return

        file_path = filedialog.asksaveasfilename(
            defaultextension=".txt",
            filetypes=[("Text files", "*.txt"), ("All files", "*.*")],
            title="Save Game Log"
        )

        if file_path:
            try:
                with open(file_path, 'w', encoding='utf-8') as f:
                    # Write header
                    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    f.write(f"=== The Crew Game Log ===\n")
                    f.write(f"Saved: {timestamp}\n")
                    f.write(f"Total Moves: {self.move_counter}\n")
                    f.write(f"{'=' * 50}\n\n")

                    # Get content from text widget (clean version for file)
                    content = self.log_text_widget.get(1.0, tk.END)
                    f.write(content)

                messagebox.showinfo("Success", f"Log saved to {file_path}")
            except Exception as e:
                messagebox.showerror("Error", f"Could not save log: {e}")

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

    def log_trick(self, trick_moves):
        """Display completed trick in GUI, wait 1.5s, clear, and log winner"""
        # Display all 4 cards in GUI
        for move in trick_moves:
            card = move.card
            player_id = move.player_id
            card_str = str(card)
            suit = card_str[0] if card_str else ''
            rank = card_str[1:] if len(card_str) > 1 else ''

            bg_color = get_suit_color(suit)
            self.trick_labels[player_id].config(
                text=rank,
                fg="white",
                bg=bg_color
            )

        # Update GUI and wait 1.5 seconds
        self.root.update()

        # Determine winner and log
        leading_card = trick_moves[0].card
        trick_winner = trick_moves[0].player_id

        for move in trick_moves[1:]:
            card = move.card
            if card.suit == leading_card.suit:
                if card.rank > leading_card.rank:
                    leading_card = card
                    trick_winner = move.player_id
            elif card.suit == CrewCard.trump_suit:
                leading_card = card
                trick_winner = move.player_id

        # Log winner
        timestamp = time.strftime("%H:%M:%S")
        self.log_text_widget.config(state=tk.NORMAL)
        display_text = f"[{timestamp}] 🏆 Player {trick_winner} wins the trick\n"
        self.log_text_widget.insert(tk.END, display_text)
        self.log_text_widget.see(tk.END)
        self.log_text_widget.config(state=tk.DISABLED)

        file_text = f"[{timestamp}] Player {trick_winner} wins the trick\n"
        self.write_to_file(file_text)

        time.sleep(1.5)

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
        # Close log file if open
        if self.log_file:
            try:
                timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                self.log_file.write(f"\n{'=' * 50}\n")
                self.log_file.write(f"Game ended: {timestamp}\n")
                self.log_file.write(f"Total moves logged: {self.move_counter}\n")
                self.log_file.close()
            except Exception as e:
                print(f"Error closing log file: {e}")
        self.root.quit()
        self.root.destroy()


class HumanAgentGUI:

    def __init__(self, log_file_path=None):
        self.use_raw = False
        self.log_file_path = log_file_path
        self.gui = CrewGameGUI(log_file_path=self.log_file_path)

    def step(self, state) -> int:
        self.gui.update_game_state(state, player_id=0)
        selected_action_id = self.gui.wait_for_action()
        return selected_action_id

    def log_move(self, player_id, action_id):
        action_text = ActionEvent.from_action_id(action_id).full_name()
        self.gui.log_player_move(player_id, action_text)

    def log_game_result(self, team_won):
        self.gui.log_game_result(team_won)
        self.gui.show_end_popup(team_won)
        self.gui.root.mainloop()

    def log_player_hands(self, players):
        self.gui.log_player_hands(players)

    def update_state_only(self, state):
        self.gui.update_game_state(state, player_id=None)

    def eval_step(self, state):
        return self.step(state), {}

    def log_trick(self, trick_moves):
        self.gui.log_trick(trick_moves)
