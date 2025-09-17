import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


class GameAnalyzer:
    def __init__(self):
        self.colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']  # Blue, orange, green, red
        self.option_order = []  # Track option order from files
        # Fixed colors for loss reasons
        self.reason_colors = {
            'R': '#ff6b6b',  # Red
            'MBR': '#66cc00',  # Teal
            'GBR': '#45b7d1',  # Blue
            'MBZ': '#f9ca24',  # Yellow
            'GBZ': '#f0932b',  # Orange
            'CZ': '#cc66ff'  # Dark Red
        }

        # Mapping of reason codes to full descriptions
        self.reason_labels = {
            'R': 'Bad card deal',
            'MBR': 'Smart bot rocket mistake',
            'GBR': 'Stupid bot rocket mistake',
            'MBZ': 'Smart bot other mistake',
            'GBZ': 'Stupid bot other mistake',
            'CZ': 'Human player mistake'
        }

    def load_and_combine_data(self, csv_files):
        """Load and combine all CSV files while preserving order"""
        all_data = []

        for file in csv_files:
            if os.path.exists(file):
                df = pd.read_csv(file, sep=';')
                print(f"Loaded {len(df)} rows from {file}")
                all_data.append(df)
                # Store option order based on file order
                if 'Option' in df.columns:
                    for option in df['Option'].unique():
                        if option not in self.option_order:
                            self.option_order.append(option)
            else:
                print(f"Warning: File {file} not found")

        if not all_data:
            raise ValueError("No valid CSV files found")

        # Combine all dataframes
        combined_df = pd.concat(all_data, ignore_index=True)
        print(f"Total combined data: {len(combined_df)} rows")

        return combined_df

    def get_ordered_options(self, data):
        """Get Option values in the order they appeared in files"""
        available_options = data['Option'].unique()
        # Keep only options that are actually in the data, in file order
        ordered_options = [option for option in self.option_order if option in available_options]
        # Add any remaining options that weren't in option_order
        remaining_options = [option for option in available_options if option not in ordered_options]
        return ordered_options + remaining_options

    def get_ordered_tasks(self, data):
        """Get Tasks values in order 1, 2, 3, 4"""
        available_tasks = sorted(data['Tasks'].unique())
        return available_tasks

    def plot_win_percentage_by_option(self, data, save_prefix=None):
        """Create a single plot with 4 subplots showing win percentage for each Option across different Tasks"""

        unique_tasks = self.get_ordered_tasks(data)
        ordered_options = self.get_ordered_options(data)

        # Define colors for each option (matching the class colors)
        option_colors = {
            ordered_options[0]: '#ff6b6b',  # Red
            ordered_options[1]: '#f9ca24',  # Yellow
            ordered_options[2]: '#45b7d1',  # Blue
            ordered_options[3]: '#2ca02c'  # Green
        }

        # Create 2x2 subplot layout
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))

        # Map options to positions
        position_map = {
            0: (0, 0),  # First option - top left
            1: (0, 1),  # Second option - top right
            2: (1, 0),  # Third option - bottom left
            3: (1, 1)  # Fourth option - bottom right
        }

        for i, option in enumerate(ordered_options[:4]):
            if i not in position_map:
                continue

            row, col = position_map[i]
            option_data = data[data['Option'] == option]

            # Calculate win percentage by Tasks for this option
            win_stats = option_data.groupby('Tasks')['Result'].apply(
                lambda x: (x == 'WIN').mean() * 100
            ).reset_index()
            win_stats.columns = ['Tasks', 'Win_Percentage']

            # Ensure all task values are present
            all_tasks_df = pd.DataFrame({'Tasks': unique_tasks})
            win_stats = all_tasks_df.merge(win_stats, on='Tasks', how='left')
            win_stats['Win_Percentage'] = win_stats['Win_Percentage'].fillna(0)

            # Sort by Tasks
            win_stats = win_stats.sort_values('Tasks')

            # Create bar plot with option-specific color
            option_color = option_colors.get(option, '#steelblue')
            bars = axes[row, col].bar(win_stats['Tasks'], win_stats['Win_Percentage'],
                                      color=option_color, alpha=0.8, width=0.6)

            # Add value labels on bars
            for bar, value in zip(bars, win_stats['Win_Percentage']):
                height = bar.get_height()
                if value > 0:  # Only show label if there's data
                    axes[row, col].text(bar.get_x() + bar.get_width() / 2., height + 1,
                                        f'{value:.1f}%', ha='center', va='bottom', fontsize=15)

            # Formatting
            axes[row, col].set_title(f'Option {option} - Win Percentage', fontsize=15, fontweight='bold', pad=10)
            axes[row, col].set_xlabel('Number of Tasks', fontsize=15)
            axes[row, col].set_ylabel('Win Percentage (%)', fontsize=15, labelpad=10)
            axes[row, col].tick_params(axis='both', which='major', labelsize=15)
            axes[row, col].set_ylim(0, 105)
            axes[row, col].grid(True, alpha=0.3, axis='y')
            axes[row, col].spines['top'].set_visible(False)
            axes[row, col].spines['right'].set_visible(False)

            # Set x-axis to show all task numbers
            axes[row, col].set_xticks(unique_tasks)
            axes[row, col].set_xticklabels(unique_tasks)

        # Hide unused subplots if less than 4 options
        for i in range(len(ordered_options), 4):
            if i in position_map:
                row, col = position_map[i]
                axes[row, col].set_visible(False)

        plt.tight_layout()
        plt.subplots_adjust(top=0.90, hspace=0.25, wspace=0.3)

        if save_prefix:
            filename = f"{save_prefix}_win_percentage_by_option_combined.pdf"
            plt.savefig(filename, bbox_inches='tight')
            print(f"Combined win percentage plot saved as: {filename}")

    def plot_win_percentage_by_option_with_bots(self, data, save_prefix=None):
        """Create a single plot with 4 subplots showing win percentage for each Option across different Tasks,
        with additional bars showing bot-only results"""

        unique_tasks = self.get_ordered_tasks(data)
        ordered_options = self.get_ordered_options(data)

        # Define colors for each option (matching the class colors)
        option_colors = {
            ordered_options[0]: '#ff6b6b',  # Red
            ordered_options[1]: '#f9ca24',  # Yellow
            ordered_options[2]: '#45b7d1',  # Blue
            ordered_options[3]: '#2ca02c'  # Green
        }

        # Bot-only win percentages (manually provided)
        bot_win_percentages = {
            ordered_options[0]: {1: 99, 2: 98, 3: 94, 4: 83},  # Red
            ordered_options[1]: {1: 100, 2: 99, 3: 94, 4: 84},  # Yellow
            ordered_options[2]: {1: 100, 2: 100, 3: 94, 4: 85},  # Blue
            ordered_options[3]: {1: 100, 2: 100, 3: 95, 4: 86}  # Green
        }

        # Create 2x2 subplot layout
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))

        # Map options to positions
        position_map = {
            0: (0, 0),  # First option - top left
            1: (0, 1),  # Second option - top right
            2: (1, 0),  # Third option - bottom left
            3: (1, 1)  # Fourth option - bottom right
        }

        for i, option in enumerate(ordered_options[:4]):
            if i not in position_map:
                continue

            row, col = position_map[i]
            option_data = data[data['Option'] == option]

            # Calculate win percentage by Tasks for this option (human + bot)
            win_stats = option_data.groupby('Tasks')['Result'].apply(
                lambda x: (x == 'WIN').mean() * 100
            ).reset_index()
            win_stats.columns = ['Tasks', 'Win_Percentage']

            # Ensure all task values are present
            all_tasks_df = pd.DataFrame({'Tasks': unique_tasks})
            win_stats = all_tasks_df.merge(win_stats, on='Tasks', how='left')
            win_stats['Win_Percentage'] = win_stats['Win_Percentage'].fillna(0)

            # Get bot-only percentages for this option
            bot_percentages = []
            for task in unique_tasks:
                bot_pct = bot_win_percentages.get(option, {}).get(task, 0)
                bot_percentages.append(bot_pct)

            # Sort by Tasks
            win_stats = win_stats.sort_values('Tasks')

            # Create grouped bar plot
            option_color = option_colors.get(option, '#steelblue')
            bot_color = self.darken_color(option_color, 0.7)  # Darker shade for bots

            x = win_stats['Tasks']
            width = 0.35  # Width of bars

            # Position bars side by side
            x1 = [pos - width / 2 for pos in x]
            x2 = [pos + width / 2 for pos in x]

            # Create bars
            bars1 = axes[row, col].bar(x1, win_stats['Win_Percentage'], width,
                                       label='Human + Bot', color=option_color, alpha=0.8)
            bars2 = axes[row, col].bar(x2, bot_percentages, width,
                                       label='Bot only', color=bot_color, alpha=0.8)

            # Add value labels on bars
            for bar, value in zip(bars1, win_stats['Win_Percentage']):
                height = bar.get_height()
                if value > 0:  # Only show label if there's data
                    axes[row, col].text(bar.get_x() + bar.get_width() / 2., height + 1,
                                        f'{value:.0f}%', ha='center', va='bottom', fontsize=12)

            for bar, value in zip(bars2, bot_percentages):
                height = bar.get_height()
                if value > 0:  # Only show label if there's data
                    axes[row, col].text(bar.get_x() + bar.get_width() / 2., height + 1,
                                        f'{value:.0f}%', ha='center', va='bottom', fontsize=12)

            # Formatting
            axes[row, col].set_title(f'Option {option} - Win Percentage', fontsize=15, fontweight='bold', pad=10)
            axes[row, col].set_xlabel('Number of Tasks', fontsize=15)
            axes[row, col].set_ylabel('Win Percentage (%)', fontsize=15, labelpad=10)
            axes[row, col].tick_params(axis='both', which='major', labelsize=15)
            axes[row, col].set_ylim(0, 105)
            axes[row, col].grid(True, alpha=0.3, axis='y')
            axes[row, col].spines['top'].set_visible(False)
            axes[row, col].spines['right'].set_visible(False)

            # Set x-axis to show all task numbers
            axes[row, col].set_xticks(unique_tasks)
            axes[row, col].set_xticklabels(unique_tasks)

        # Hide unused subplots if less than 4 options
        for i in range(len(ordered_options), 4):
            if i in position_map:
                row, col = position_map[i]
                axes[row, col].set_visible(False)

        # Create figure-level legend on the right side of the title
        from matplotlib.patches import Rectangle
        legend_elements = [
            Rectangle((0, 0), 1, 1, facecolor='#808080', alpha=0.8, label='Human + Bot'),
            Rectangle((0, 0), 1, 1, facecolor='#404040', alpha=0.8, label='Bot only')
        ]

        fig.legend(handles=legend_elements, loc='upper right', bbox_to_anchor=(0.95, 1),
                   fontsize=14)

        plt.tight_layout()
        plt.subplots_adjust(top=0.88, hspace=0.25, wspace=0.3)

        if save_prefix:
            filename = f"{save_prefix}_win_percentage_with_bots.pdf"
            plt.savefig(filename, bbox_inches='tight')
            print(f"Win percentage plot with bot comparison saved as: {filename}")

    def darken_color(self, color, factor):
        """Helper function to darken a color by a given factor"""
        import matplotlib.colors as mcolors

        # Convert color to RGB if it's a hex string
        if isinstance(color, str) and color.startswith('#'):
            rgb = mcolors.hex2color(color)
        else:
            rgb = mcolors.to_rgb(color)

        # Darken by multiplying by factor
        darkened = tuple(c * factor for c in rgb)

        return darkened

    def plot_average_rounds(self, data, save_prefix=None):
        """Create separate bar plots showing average rounds for wins vs losses by Option"""

        unique_tasks = self.get_ordered_tasks(data)

        # Create 2x2 subplot layout
        fig, axes = plt.subplots(2, 2, figsize=(20, 16))

        # Map tasks to positions: 1(0,0), 2(0,1), 3(1,0), 4(1,1)
        position_map = {
            1: (0, 0),  # 1 task - top left
            2: (0, 1),  # 2 tasks - top right
            3: (1, 0),  # 3 tasks - bottom left
            4: (1, 1)  # 4 tasks - bottom right
        }

        for tasks in unique_tasks:
            if tasks not in position_map:
                continue

            row, col = position_map[tasks]
            task_data = data[data['Tasks'] == tasks]

            # Calculate average rounds by Option and Result
            avg_rounds = task_data.groupby(['Option', 'Result'])['Rounds'].mean().unstack(fill_value=0)

            # Sort by the file order
            ordered_options = self.get_ordered_options(task_data)
            avg_rounds = avg_rounds.reindex(ordered_options)
            avg_rounds = avg_rounds.dropna()  # Remove any NaN rows

            # Create grouped bar plot
            x = np.arange(len(avg_rounds.index))
            width = 0.35

            bars1 = None
            bars2 = None

            if 'WIN' in avg_rounds.columns:
                bars1 = axes[row, col].bar(x - width / 2, avg_rounds['WIN'], width,
                                           label='Wins', color='green', alpha=0.8)
            if 'LOSE' in avg_rounds.columns:
                bars2 = axes[row, col].bar(x + width / 2, avg_rounds['LOSE'], width,
                                           label='Losses', color='red', alpha=0.8)

            # Add value labels with larger font
            for bars in [bars1, bars2]:
                if bars is not None:
                    for bar in bars:
                        if bar.get_height() > 0:
                            height = bar.get_height()
                            axes[row, col].text(bar.get_x() + bar.get_width() / 2., height + 0.1,
                                                f'{height:.1f}', ha='center', va='bottom', fontsize=20)

            # Formatting with larger fonts
            axes[row, col].set_title(f'{tasks} Tasks', fontsize=32, fontweight='bold', pad=20)
            axes[row, col].set_xticks(x)
            axes[row, col].set_xticklabels(avg_rounds.index)
            axes[row, col].tick_params(axis='both', which='major', labelsize=24)
            axes[row, col].grid(True, alpha=0.3, axis='y')
            axes[row, col].spines['top'].set_visible(False)
            axes[row, col].spines['right'].set_visible(False)

            # Only add labels to left column and bottom row
            if col == 0:  # Left column
                axes[row, col].set_ylabel('Average Rounds', fontsize=28, labelpad=20)
            if row == 1:  # Bottom row
                axes[row, col].set_xlabel('Option', fontsize=28)

        # Add single legend at the top of the figure
        handles, labels = axes[0, 0].get_legend_handles_labels()
        if handles:
            fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 0.95),
                       ncol=2, fontsize=28)

        # Hide unused subplots if less than 4 tasks
        for tasks in [1, 2, 3, 4]:
            if tasks not in unique_tasks and tasks in position_map:
                row, col = position_map[tasks]
                axes[row, col].set_visible(False)

        plt.tight_layout()
        plt.subplots_adjust(top=0.85, hspace=0.3)  # Make room for legend and title, add vertical space

        if save_prefix:
            filename = f"{save_prefix}_average_rounds.pdf"
            plt.savefig(filename, bbox_inches='tight')
            print(f"Average rounds plots saved as: {filename}")

    def plot_average_rounds_by_option(self, data, save_prefix=None):
        """Create bar plots showing average rounds for each Option across different Tasks"""

        ordered_options = self.get_ordered_options(data)

        # Create 2x2 subplot layout
        fig, axes = plt.subplots(2, 2, figsize=(20, 16))

        # Map options to positions: Red(0,0), Yellow(0,1), Blue(1,0), Green(1,1)
        position_map = {
            0: (0, 0),  # Red - top left
            1: (0, 1),  # Yellow - top right
            2: (1, 0),  # Blue - bottom left
            3: (1, 1)  # Green - bottom right
        }

        for i, option in enumerate(ordered_options[:4]):
            if i not in position_map:
                continue

            row, col = position_map[i]
            option_data = data[data['Option'] == option]

            # Calculate average rounds by Tasks and Result
            avg_rounds = option_data.groupby(['Tasks', 'Result'])['Rounds'].mean().unstack(fill_value=0)

            # Sort by Tasks
            avg_rounds = avg_rounds.sort_index()

            # Create grouped bar plot
            x = np.arange(len(avg_rounds.index))
            width = 0.35

            bars1 = None
            bars2 = None

            if 'WIN' in avg_rounds.columns:
                bars1 = axes[row, col].bar(x - width / 2, avg_rounds['WIN'], width,
                                           label='Wins', color='green', alpha=0.8)
            if 'LOSE' in avg_rounds.columns:
                bars2 = axes[row, col].bar(x + width / 2, avg_rounds['LOSE'], width,
                                           label='Losses', color='red', alpha=0.8)

            # Add value labels with larger font
            for bars in [bars1, bars2]:
                if bars is not None:
                    for bar in bars:
                        if bar.get_height() > 0:
                            height = bar.get_height()
                            axes[row, col].text(bar.get_x() + bar.get_width() / 2., height + 0.1,
                                                f'{height:.1f}', ha='center', va='bottom', fontsize=20)

            # Formatting with larger fonts
            axes[row, col].set_title(f'Option {option}', fontsize=32, fontweight='bold', pad=20)
            axes[row, col].set_xticks(x)
            axes[row, col].set_xticklabels(avg_rounds.index)
            axes[row, col].tick_params(axis='both', which='major', labelsize=24)
            axes[row, col].grid(True, alpha=0.3, axis='y')
            axes[row, col].spines['top'].set_visible(False)
            axes[row, col].spines['right'].set_visible(False)

            # Only add labels to left column and bottom row
            if col == 0:  # Left column
                axes[row, col].set_ylabel('Average Rounds', fontsize=28, labelpad=20)
            if row == 1:  # Bottom row
                axes[row, col].set_xlabel('Number of Tasks', fontsize=28)

        # Add single legend at the top of the figure
        handles, labels = axes[0, 0].get_legend_handles_labels()
        if handles:
            fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 0.95),
                       ncol=2, fontsize=28)

        # Hide unused subplots if less than 4 options
        for i in range(len(ordered_options), 4):
            if i in position_map:
                row, col = position_map[i]
                axes[row, col].set_visible(False)

        plt.tight_layout()
        plt.subplots_adjust(top=0.85, hspace=0.3)  # Make room for legend and title, add vertical space

        if save_prefix:
            filename = f"{save_prefix}_average_rounds_by_option.pdf"
            plt.savefig(filename, bbox_inches='tight')
            print(f"Average rounds by option plots saved as: {filename}")

    def plot_average_won_tasks_in_losses(self, data, save_prefix=None):
        """Create a single bar plot showing average won tasks in lost games by Option for Tasks=3 and Tasks=4"""

        # Filter for Tasks=3 and Tasks=4
        target_tasks = [3, 4]
        filtered_data = data[data['Tasks'].isin(target_tasks)]

        # Get all options across both task counts
        all_options = self.get_ordered_options(filtered_data)

        if len(all_options) == 0:
            print("No data found for Tasks=3 or Tasks=4")
            return

        # Create individual plot
        fig, ax = plt.subplots(1, 1, figsize=(12, 8))

        # Prepare data for both task counts
        stats_by_tasks = {}

        for tasks in target_tasks:
            task_losses = data[(data['Tasks'] == tasks) & (data['Result'] == 'LOSE')]

            if len(task_losses) > 0:
                won_stats = task_losses.groupby('Option')['Won'].mean().reset_index()
                won_stats.columns = ['Option', 'Avg_Won_Tasks']
            else:
                won_stats = pd.DataFrame(columns=['Option', 'Avg_Won_Tasks'])

            # Create full dataset with all options, filling missing with NaN
            full_stats = pd.DataFrame({'Option': all_options})
            full_stats = full_stats.merge(won_stats, on='Option', how='left')
            stats_by_tasks[tasks] = full_stats

        # Create grouped bar plot
        x = np.arange(len(all_options))
        width = 0.35

        # Prepare data for Tasks=3
        values_3 = []
        for option in all_options:
            option_data = stats_by_tasks[3][stats_by_tasks[3]['Option'] == option]
            if len(option_data) > 0 and not pd.isna(option_data['Avg_Won_Tasks'].iloc[0]):
                values_3.append(option_data['Avg_Won_Tasks'].iloc[0])
            else:
                values_3.append(0)

        # Prepare data for Tasks=4
        values_4 = []
        for option in all_options:
            option_data = stats_by_tasks[4][stats_by_tasks[4]['Option'] == option]
            if len(option_data) > 0 and not pd.isna(option_data['Avg_Won_Tasks'].iloc[0]):
                values_4.append(option_data['Avg_Won_Tasks'].iloc[0])
            else:
                values_4.append(0)

        # Create bars
        bars1 = ax.bar(x - width / 2, values_3, width, label='3 Tasks', color='lightcoral', alpha=0.8)
        bars2 = ax.bar(x + width / 2, values_4, width, label='4 Tasks', color='orange', alpha=0.8)

        # Add value labels on bars
        for bar, value in zip(bars1, values_3):
            if value > 0:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width() / 2., height + 0.02,
                        f'{value:.2f}', ha='center', va='bottom', fontsize=15)

        for bar, value in zip(bars2, values_4):
            if value > 0:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width() / 2., height + 0.02,
                        f'{value:.2f}', ha='center', va='bottom', fontsize=15)

        # Check for options with no losses and add annotation
        for i, option in enumerate(all_options):
            has_losses_3 = not pd.isna(
                stats_by_tasks[3][stats_by_tasks[3]['Option'] == option]['Avg_Won_Tasks'].iloc[0])
            has_losses_4 = not pd.isna(
                stats_by_tasks[4][stats_by_tasks[4]['Option'] == option]['Avg_Won_Tasks'].iloc[0])

            if not has_losses_3 and not has_losses_4:
                ax.text(i, 0.1, 'No losses\n(both)', ha='center', va='bottom',
                        fontsize=13, style='italic', color='gray')
            elif not has_losses_3:
                ax.text(i - width / 2, 0.05, 'No losses', ha='center', va='bottom',
                        fontsize=13, style='italic', color='gray', rotation=0)
            elif not has_losses_4:
                ax.text(i + width / 2, 0.05, 'No losses', ha='center', va='bottom',
                        fontsize=14, style='italic', color='gray', rotation=0)

        # Formatting
        ax.set_xlabel('Option', fontsize=18)
        ax.set_ylabel('Average Won Tasks', fontsize=16, labelpad=20)
        ax.set_xticks(x)
        ax.set_xticklabels(all_options)
        ax.tick_params(axis='both', which='major', labelsize=16)

        # Set y-axis limit
        max_val = max(max(values_3), max(values_4))
        if max_val > 0:
            ax.set_ylim(0, max_val * 1.3)
        else:
            ax.set_ylim(0, 1)

        ax.grid(True, alpha=0.3, axis='y')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

        # Add legend
        ax.legend(fontsize=12)

        plt.tight_layout()

        if save_prefix:
            filename = f"{save_prefix}_avg_won_tasks_in_losses_combined.pdf"
            plt.savefig(filename, bbox_inches='tight')
            print(f"Combined average won tasks in losses plot saved as: {filename}")

    def plot_loss_reasons_pie_charts(self, data, save_prefix=None):
        """Create pie charts showing loss reasons by Option for Tasks=4 with consistent colors and single legend"""

        # Filter for Tasks=4 and losses only
        task4_losses = data[data['Result'] == 'LOSE']

        if len(task4_losses) == 0:
            print("No losses found for Tasks=4")
            return

        # Get ordered options
        ordered_options = self.get_ordered_options(task4_losses)

        # Create 2x2 subplot layout with extra space for legend
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))

        # Map options to positions: Red(0,0), Yellow(0,1), Blue(1,0), Green(1,1)
        position_map = {
            0: (0, 0),  # Red - top left
            1: (0, 1),  # Yellow - top right
            2: (1, 0),  # Blue - bottom left
            3: (1, 1)  # Green - bottom right
        }

        # Collect all unique reasons across all options for consistent legend
        all_reasons = set()
        for option in ordered_options[:4]:
            option_losses = task4_losses[task4_losses['Option'] == option]
            if len(option_losses) > 0:
                all_reasons.update(option_losses['Reason'].unique())

        # Sort reasons for consistent order
        all_reasons = sorted(list(all_reasons))

        for i, option in enumerate(ordered_options[:4]):
            row, col = position_map[i]
            option_losses = task4_losses[task4_losses['Option'] == option]

            if len(option_losses) == 0:
                axes[row, col].text(0.5, 0.5, 'No losses', ha='center', va='center',
                                    transform=axes[row, col].transAxes, fontsize=14)
                axes[row, col].set_title(f'Option {option} - Loss Reasons',
                                         fontsize=18, fontweight='bold')
                continue

            # Count loss reasons for this specific option
            reason_counts = option_losses['Reason'].value_counts()

            # Get colors for the reasons present in this data
            colors_for_plot = [self.reason_colors.get(reason, '#gray') for reason in reason_counts.index]

            # Create pie chart without labels (they'll be in the legend)
            wedges, texts, autotexts = axes[row, col].pie(reason_counts.values,
                                                          autopct='%1.1f%%',
                                                          colors=colors_for_plot,
                                                          startangle=90,
                                                          textprops={'fontsize': 12})

            # Formatting
            axes[row, col].set_title(f'Option {option} - Loss Reasons',
                                     fontsize=18, fontweight='bold')

            # Improve text readability
            for autotext in autotexts:
                autotext.set_fontsize(15)
                autotext.set_fontweight('bold')

        # Hide unused subplots if less than 4 options
        for i in range(len(ordered_options), 4):
            row, col = position_map[i]
            axes[row, col].set_visible(False)

        # Create a single legend for all charts
        legend_labels = [self.reason_labels.get(reason, reason) for reason in all_reasons]
        legend_colors = [self.reason_colors.get(reason, '#gray') for reason in all_reasons]

        # Create legend patches
        legend_patches = [plt.Rectangle((0, 0), 1, 1, facecolor=color) for color in legend_colors]

        # Add legend to the figure (positioned on the right side)
        fig.legend(legend_patches, legend_labels, loc='lower center', bbox_to_anchor=(0.42, -0.2),
                   fontsize=16, title='Loss Reasons', title_fontsize=14)

        plt.tight_layout()

        # Adjust layout to make room for legend
        plt.subplots_adjust(right=0.82)

        if save_prefix:
            filename = f"{save_prefix}_loss_reasons.pdf"
            plt.savefig(filename, bbox_inches='tight')
            print(f"Loss reasons pie charts saved as: {filename}")

    def plot_loss_reasons_pie_charts_by_tasks(self, data, save_prefix=None):
        """Create pie charts showing loss reasons by Tasks with consistent colors and single legend"""

        # Filter for losses only
        losses_data = data[data['Result'] == 'LOSE']

        if len(losses_data) == 0:
            print("No losses found in data")
            return

        # Get unique tasks values and sort them
        unique_tasks = sorted(losses_data['Tasks'].unique())

        if len(unique_tasks) == 0:
            print("No task data found")
            return

        # Determine subplot layout based on number of unique tasks
        n_tasks = len(unique_tasks)
        if n_tasks <= 4:
            rows, cols = 2, 2
            figsize = (16, 12)
        elif n_tasks <= 6:
            rows, cols = 2, 3
            figsize = (18, 12)
        elif n_tasks <= 9:
            rows, cols = 3, 3
            figsize = (18, 14)
        else:
            # For more than 9 tasks, use 4 columns
            rows = (n_tasks + 3) // 4
            cols = 4
            figsize = (20, 4 * rows + 2)

        # Create subplot layout
        fig, axes = plt.subplots(rows, cols, figsize=figsize)

        # Handle single subplot case
        if n_tasks == 1:
            axes = [axes]
        elif rows == 1 or cols == 1:
            axes = axes.flatten() if hasattr(axes, 'flatten') else [axes]
        else:
            axes = axes.flatten()

        # Collect all unique reasons across all tasks for consistent legend
        all_reasons = set()
        for task_count in unique_tasks:
            task_losses = losses_data[losses_data['Tasks'] == task_count]
            if len(task_losses) > 0:
                all_reasons.update(task_losses['Reason'].unique())

        # Sort reasons for consistent order
        all_reasons = sorted(list(all_reasons))

        # Create pie chart for each task count
        for i, task_count in enumerate(unique_tasks):
            task_losses = losses_data[losses_data['Tasks'] == task_count]

            if len(task_losses) == 0:
                axes[i].text(0.5, 0.5, 'No losses', ha='center', va='center',
                             transform=axes[i].transAxes, fontsize=14)
                axes[i].set_title(f'Tasks = {task_count} - Loss Reasons',
                                  fontsize=18, fontweight='bold')
                continue

            # Count loss reasons for this specific task count
            reason_counts = task_losses['Reason'].value_counts()

            # Get colors for the reasons present in this data
            colors_for_plot = [self.reason_colors.get(reason, '#gray') for reason in reason_counts.index]

            # Create pie chart without labels (they'll be in the legend)
            wedges, texts, autotexts = axes[i].pie(reason_counts.values,
                                                   autopct='%1.1f%%',
                                                   colors=colors_for_plot,
                                                   startangle=90,
                                                   textprops={'fontsize': 12})

            # Formatting
            axes[i].set_title(f'Tasks = {task_count} - Loss Reasons\n(n={len(task_losses)})',
                              fontsize=16, fontweight='bold')

            # Improve text readability
            for autotext in autotexts:
                autotext.set_fontsize(13)
                autotext.set_fontweight('bold')

        # Hide unused subplots if any
        for i in range(n_tasks, len(axes)):
            axes[i].set_visible(False)

        # Create a single legend for all charts
        legend_labels = [self.reason_labels.get(reason, reason) for reason in all_reasons]
        legend_colors = [self.reason_colors.get(reason, '#gray') for reason in all_reasons]

        # Create legend patches
        legend_patches = [plt.Rectangle((0, 0), 1, 1, facecolor=color) for color in legend_colors]

        # Add legend to the figure (positioned below the charts)
        legend_y_position = -0.15 if rows <= 2 else -0.10
        fig.legend(legend_patches, legend_labels, loc='lower center',
                   bbox_to_anchor=(0.5, legend_y_position),
                   fontsize=14, title='Loss Reasons', title_fontsize=16,
                   ncol=min(len(all_reasons), 4))

        plt.tight_layout()

        # Adjust layout to make room for legend
        plt.subplots_adjust(bottom=0.15 if rows <= 2 else 0.10)

        if save_prefix:
            filename = f"{save_prefix}_loss_reasons_by_tasks.pdf"
            plt.savefig(filename, bbox_inches='tight', dpi=300)
            print(f"Loss reasons by tasks pie charts saved as: {filename}")

    def plot_loss_reasons_by_option_and_tasks(self, data, save_prefix=None):
        """Create pie charts showing loss reasons by Option and Tasks combinations"""

        # Filter for losses only
        losses_data = data[data['Result'] == 'LOSE']

        if len(losses_data) == 0:
            print("No losses found in data")
            return

        # Get unique combinations of Option and Tasks
        combinations = losses_data[['Option', 'Tasks']].drop_duplicates().sort_values(['Option', 'Tasks'])

        if len(combinations) == 0:
            print("No Option-Tasks combinations found")
            return

        # Get unique Options and Tasks for organization
        unique_options = sorted(losses_data['Option'].unique())
        unique_tasks = sorted(losses_data['Tasks'].unique())

        print(f"Found {len(combinations)} Option-Tasks combinations:")
        print(f"Options: {unique_options}")
        print(f"Tasks: {unique_tasks}")

        # Determine subplot layout
        n_combinations = len(combinations)
        if n_combinations <= 4:
            rows, cols = 2, 2
            figsize = (16, 12)
        elif n_combinations <= 6:
            rows, cols = 2, 3
            figsize = (18, 12)
        elif n_combinations <= 9:
            rows, cols = 3, 3
            figsize = (18, 16)
        elif n_combinations <= 12:
            rows, cols = 3, 4
            figsize = (20, 16)
        else:
            # For more combinations, use 4 columns
            rows = (n_combinations + 3) // 4
            cols = 4
            figsize = (20, 4 * rows + 2)

        # Create subplot layout
        fig, axes = plt.subplots(rows, cols, figsize=figsize)

        # Handle different subplot cases
        if n_combinations == 1:
            axes = [axes]
        elif rows == 1 or cols == 1:
            axes = axes.flatten() if hasattr(axes, 'flatten') else [axes]
        else:
            axes = axes.flatten()

        # Collect all unique reasons for consistent legend
        all_reasons = set()
        combination_data = {}

        for idx, (_, row) in enumerate(combinations.iterrows()):
            option, task = row['Option'], row['Tasks']
            combo_losses = losses_data[(losses_data['Option'] == option) &
                                       (losses_data['Tasks'] == task)]

            if len(combo_losses) > 0:
                all_reasons.update(combo_losses['Reason'].unique())
                combination_data[(option, task)] = combo_losses

        # Sort reasons for consistent order
        all_reasons = sorted(list(all_reasons))

        # Create pie chart for each Option-Tasks combination
        plot_idx = 0
        for idx, (_, row) in enumerate(combinations.iterrows()):
            option, task = row['Option'], row['Tasks']
            combo_key = (option, task)

            if combo_key not in combination_data:
                # Skip combinations with no losses
                continue

            combo_losses = combination_data[combo_key]

            if len(combo_losses) == 0:
                axes[plot_idx].text(0.5, 0.5, 'No losses', ha='center', va='center',
                                    transform=axes[plot_idx].transAxes, fontsize=12)
                axes[plot_idx].set_title(f'Option {option}, Tasks {task}\nNo losses',
                                         fontsize=14, fontweight='bold')
                plot_idx += 1
                continue

            # Count loss reasons for this combination
            reason_counts = combo_losses['Reason'].value_counts()

            # Get colors for the reasons present in this data
            colors_for_plot = [self.reason_colors.get(reason, '#gray') for reason in reason_counts.index]

            # Create pie chart
            wedges, texts, autotexts = axes[plot_idx].pie(reason_counts.values,
                                                          autopct='%1.1f%%',
                                                          colors=colors_for_plot,
                                                          startangle=90,
                                                          textprops={'fontsize': 10})

            # Formatting
            axes[plot_idx].set_title(f'Option {option}, Tasks {task}\n(n={len(combo_losses)})',
                                     fontsize=14, fontweight='bold')

            # Improve text readability
            for autotext in autotexts:
                autotext.set_fontsize(11)
                autotext.set_fontweight('bold')

            plot_idx += 1

        # Hide unused subplots
        for i in range(plot_idx, len(axes)):
            axes[i].set_visible(False)

        # Create a single legend for all charts
        legend_labels = [self.reason_labels.get(reason, reason) for reason in all_reasons]
        legend_colors = [self.reason_colors.get(reason, '#gray') for reason in all_reasons]

        # Create legend patches
        legend_patches = [plt.Rectangle((0, 0), 1, 1, facecolor=color) for color in legend_colors]

        # Position legend based on number of rows
        if rows <= 2:
            legend_y = -0.12
            bottom_adjust = 0.15
        elif rows <= 3:
            legend_y = -0.08
            bottom_adjust = 0.12
        else:
            legend_y = -0.05
            bottom_adjust = 0.08

        # Add legend
        fig.legend(legend_patches, legend_labels, loc='lower center',
                   bbox_to_anchor=(0.5, legend_y),
                   fontsize=12, title='Loss Reasons', title_fontsize=14,
                   ncol=min(len(all_reasons), 5))

        plt.tight_layout()
        plt.subplots_adjust(bottom=bottom_adjust)

        if save_prefix:
            filename = f"{save_prefix}_loss_reasons_option_tasks.pdf"
            plt.savefig(filename, bbox_inches='tight', dpi=300)
            print(f"Loss reasons by Option-Tasks combinations saved as: {filename}")

        # Print summary statistics
        print("\nSummary of combinations:")
        for (option, task), combo_losses in combination_data.items():
            reason_counts = combo_losses['Reason'].value_counts()
            print(f"Option {option}, Tasks {task}: {len(combo_losses)} losses")
            for reason, count in reason_counts.items():
                percentage = (count / len(combo_losses)) * 100
                print(f"  {reason}: {count} ({percentage:.1f}%)")

    def generate_all_plots(self, csv_files, save_prefix=None):
        """Generate all requested plots"""

        # Load data
        data = self.load_and_combine_data(csv_files)

        print(f"\nData overview:")
        print(f"Option order from files: {self.option_order}")
        print(f"Unique Tasks: {sorted(data['Tasks'].unique())}")
        print(f"Result distribution:")
        print(data['Result'].value_counts())
        print(f"Loss reasons (when Result=LOSE):")
        print(data[data['Result'] == 'LOSE']['Reason'].value_counts())

        # Generate plots
        print("\n1. Creating win percentage plots...")
        self.plot_win_percentage_by_option(data, save_prefix)

        print("\n2. Creating average rounds plots (by tasks)...")
        self.plot_average_rounds(data, save_prefix)

        print("\n3. Creating average rounds plots (by option)...")
        self.plot_average_rounds_by_option(data, save_prefix)

        print("\n4. Creating average won tasks in losses plots...")
        self.plot_average_won_tasks_in_losses(data, save_prefix)

        print("\n5. Creating loss reasons pie charts...")
        self.plot_loss_reasons_pie_charts(data, save_prefix)

        self.plot_loss_reasons_by_option_and_tasks(data, save_prefix)

        self.plot_win_percentage_by_option_with_bots(data, save_prefix)


def main():
    parser = argparse.ArgumentParser(description='Analyze game data and create plots')
    parser.add_argument('files', nargs='+', help='CSV file paths (up to 4 files)')
    parser.add_argument('--save-prefix', help='Prefix for saved plot files')

    args = parser.parse_args()

    analyzer = GameAnalyzer()
    analyzer.generate_all_plots(args.files, args.save_prefix)


if __name__ == "__main__":
    main()

# Example usage:
# python game_analysis.py red.csv yellow.csv blue.csv green.csv --save-prefix results
#
# This will create separate files:
# - results_win_percentage_tasks1.pdf
# - results_win_percentage_tasks2.pdf
# - results_win_percentage_tasks3.pdf
# - results_win_percentage_tasks4.pdf
# - results_average_rounds.pdf
# - results_average_rounds_by_option.pdf
# - results_avg_won_tasks_in_losses.pdf
# - results_loss_reasons_tasks4.pdf
