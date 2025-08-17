import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.ndimage import uniform_filter1d


class CSVSmoother:
    def __init__(self):
        self.colors = ['#1f77b4', '#ff7f0e', '#2ca02c']  # Default matplotlib colors

    def load_csv(self, filepath):
        """Load CSV file and check if it contains required columns"""
        try:
            df = pd.read_csv(filepath)

            # Check for required columns
            if 'episode' not in df.columns or 'reward' not in df.columns:
                raise ValueError(f"File {filepath} must contain 'episode' and 'reward' columns")

            # Clean data and sort
            df = df.dropna(subset=['episode', 'reward'])
            df = df.sort_values('episode')

            print(f"Loaded {len(df)} points from file: {os.path.basename(filepath)}")
            return df

        except Exception as e:
            print(f"Error loading {filepath}: {e}")
            return None

    def moving_average(self, data, window_size):
        """Moving average smoothing"""
        if len(data) < window_size:
            window_size = len(data)

        smoothed = uniform_filter1d(data.astype(float), size=window_size, mode='nearest')

        # Calculate standard deviation for each window
        std_devs = []
        half_window = window_size // 2

        for i in range(len(data)):
            start_idx = max(0, i - half_window)
            end_idx = min(len(data), i + half_window + 1)
            window_data = data[start_idx:end_idx]
            std_devs.append(np.std(window_data))

        std_devs = np.array(std_devs)

        return smoothed, std_devs

    def apply_smoothing(self, df, window_size=10):
        """Apply moving average smoothing to data"""
        episodes = df['episode'].values
        rewards = df['reward'].values

        # Convert to percentage
        rewards_percent = rewards * 100

        smoothed_rewards, std_devs = self.moving_average(rewards_percent, window_size)

        return episodes, rewards_percent, smoothed_rewards, std_devs

    def plot_single_scale(self, ax, datasets, window_size=10, show_confidence=True,
                          show_original=False, ylim=None, title=""):
        """Plot data on a single axis"""

        for i, (df, name) in enumerate(datasets):
            if df is None:
                continue

            episodes, original_rewards, smoothed_rewards, std_devs = self.apply_smoothing(
                df, window_size
            )

            color = self.colors[i % len(self.colors)]

            # Plot smoothed data
            ax.plot(episodes, smoothed_rewards, color=color, linewidth=2.5,
                    label=name, alpha=0.9)

            # Confidence interval
            if show_confidence:
                ax.fill_between(episodes,
                                smoothed_rewards - std_devs,
                                smoothed_rewards + std_devs,
                                color=color, alpha=0.2, label=f'{name} ±σ')

            # Original data (optional)
            if show_original:
                ax.plot(episodes, original_rewards, color=color, alpha=0.3,
                        linewidth=0.8, linestyle='--', label=f'{name} (original)')

        ax.set_xlabel('Episode', fontsize=20)
        ax.set_ylabel('Accuracy (%)', fontsize=20)
        ax.tick_params(axis='both', which='major', labelsize=18)
        ax.grid(True, alpha=0.3)

        if ylim:
            ax.set_ylim(ylim)

        if title:
            ax.set_title(title, fontsize=22, pad=30)

    def plot_data(self, datasets, window_size=10, show_confidence=True,
                  show_original=False, figsize=(12, 12), zoom_ylim=None):
        """Create dual-scale plot with full (0-100%) and auto-scaled accuracy ranges"""

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=figsize)
        ax1.yaxis.set_label_coords(-0.05, 0.5)
        ax2.yaxis.set_label_coords(-0.05, 0.5)

        plt.style.use('seaborn-v0_8' if 'seaborn-v0_8' in plt.style.available else 'default')

        # Full scale plot (0-100%)
        self.plot_single_scale(ax1, datasets, window_size, show_confidence,
                               show_original, ylim=(0, 100), title="Full Scale (0-100%)")

        # Auto-scaled plot (let matplotlib determine the best range)
        self.plot_single_scale(ax2, datasets, window_size, show_confidence,
                               show_original, ylim=None, title="Auto-scaled")

        # Single legend for both plots
        handles, labels = ax1.get_legend_handles_labels()
        fig.legend(handles, labels, loc='lower center', bbox_to_anchor=(0.5, -0.08),
                   ncol=len(labels) // 2 if len(labels) > 3 else 3, fontsize=18)

        # Add more space between subplots - must be called before tight_layout
        plt.tight_layout()
        plt.subplots_adjust(hspace=0.4)
        return fig

    def plot_data_single(self, datasets, window_size=10, show_confidence=True,
                         show_original=False, figsize=(12, 8)):
        """Create single plot (original behavior)"""

        plt.figure(figsize=figsize)
        plt.style.use('seaborn-v0_8' if 'seaborn-v0_8' in plt.style.available else 'default')

        for i, (df, name) in enumerate(datasets):
            if df is None:
                continue

            episodes, original_rewards, smoothed_rewards, std_devs = self.apply_smoothing(
                df, window_size
            )

            color = self.colors[i % len(self.colors)]

            # Plot smoothed data
            plt.plot(episodes, smoothed_rewards, color=color, linewidth=2.5,
                     label=name, alpha=0.9)

            # Confidence interval
            if show_confidence:
                plt.fill_between(episodes,
                                 smoothed_rewards - std_devs,
                                 smoothed_rewards + std_devs,
                                 color=color, alpha=0.2, label=f'{name} ±σ')

            # Original data (optional)
            if show_original:
                plt.plot(episodes, original_rewards, color=color, alpha=0.3,
                         linewidth=0.8, linestyle='--', label=f'{name} (original)')

        plt.xlabel('Episode', fontsize=18)
        plt.ylabel('Accuracy (%)', fontsize=18)
        plt.tick_params(axis='both', which='major', labelsize=16)
        plt.legend(loc='lower center', bbox_to_anchor=(0.5, -0.25), ncol=3, fontsize=16)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()

        return plt


def main():
    parser = argparse.ArgumentParser(description='Smooth and visualize data from CSV files')
    parser.add_argument('files', nargs='*', help='Paths to CSV files (maximum 3)')
    parser.add_argument('--window', type=int, default=10, help='Window size for moving average')
    parser.add_argument('--no-confidence', action='store_true', help='Do not show confidence intervals')
    parser.add_argument('--show-original', action='store_true', help='Also show original data')
    parser.add_argument('--save', help='Save plot to file (e.g. plot.png)')
    parser.add_argument('--dual-scale', action='store_true', help='Create dual-scale plot (full + zoomed)')
    parser.add_argument('--zoom-max', type=float, help='Maximum value for zoomed plot (auto if not specified)')

    args = parser.parse_args()

    smoother = CSVSmoother()

    # Example files if no arguments provided
    if not args.files:
        print("Example usage:")
        print("python script.py file1.csv file2.csv file3.csv --window 15 --dual-scale")
        print("python script.py file1.csv file2.csv file3.csv --dual-scale --zoom-max 20")
        print("\nThe script applies moving average smoothing to CSV files.")
        print("CSV files must contain 'episode' and 'reward' columns.")
        print("Use --dual-scale to create plots with full and zoomed accuracy ranges.")
        return

    # Load files
    datasets = []
    series_labels = ['normal', 'fixed tasks', 'disabled signaling']

    for i, filepath in enumerate(args.files[:3]):  # Maximum 3 files
        df = smoother.load_csv(filepath)
        # Use predefined labels for series
        label = series_labels[i] if i < len(series_labels) else f'series_{i + 1}'
        datasets.append((df, label))

    # Check if any data was loaded
    if not any(df is not None for df, _ in datasets):
        print("Failed to load any data!")
        return

    # Create plot based on mode
    if args.dual_scale:
        zoom_ylim = (0, args.zoom_max) if args.zoom_max else None
        fig = smoother.plot_data(
            datasets,
            window_size=args.window,
            show_confidence=not args.no_confidence,
            show_original=args.show_original,
            zoom_ylim=zoom_ylim
        )
        plot_obj = fig
    else:
        plot_obj = smoother.plot_data_single(
            datasets,
            window_size=args.window,
            show_confidence=not args.no_confidence,
            show_original=args.show_original
        )

    if args.save:
        # Determine file format from extension, default to PDF for LaTeX
        if args.save.lower().endswith('.png'):
            plot_obj.savefig(args.save, dpi=300, bbox_inches='tight', format='png')
        elif args.save.lower().endswith('.pdf'):
            plot_obj.savefig(args.save, bbox_inches='tight', format='pdf')
        else:
            # Default to PDF for LaTeX compatibility
            pdf_filename = args.save + '.pdf'
            plot_obj.savefig(pdf_filename, bbox_inches='tight', format='pdf')
            print(f"No extension specified, saved as PDF: {pdf_filename}")
            return
        print(f"Plot saved as: {args.save}")

if __name__ == "__main__":
    main()

# Example of programmatic usage:
"""
# Example 1: Dual-scale plot
smoother = CSVSmoother()
df1 = smoother.load_csv('experiment_1.csv')
df2 = smoother.load_csv('experiment_2.csv')
df3 = smoother.load_csv('experiment_3.csv')

datasets = [(df1, 'normal'), (df2, 'fixed tasks'), (df3, 'disabled signaling')]
fig = smoother.plot_data(datasets, window_size=15, zoom_ylim=(0, 20))
plt.show()

# Example 2: Single plot (original behavior)
plt_obj = smoother.plot_data_single(datasets, window_size=20, show_confidence=False)
plt.show()
"""