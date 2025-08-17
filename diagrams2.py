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

    def plot_subplot(self, ax, datasets, window_size=10, show_confidence=True,
                     show_original=False, subplot_title=None):
        """Create a single subplot with smoothed data"""

        handles = []  # For legend
        labels = []

        for i, (df, name) in enumerate(datasets):
            if df is None:
                continue

            episodes, original_rewards, smoothed_rewards, std_devs = self.apply_smoothing(
                df, window_size
            )

            color = self.colors[i % len(self.colors)]

            # Plot smoothed data
            line, = ax.plot(episodes, smoothed_rewards, color=color, linewidth=2.5,
                            label=name, alpha=0.9)
            handles.append(line)
            labels.append(name)

            # Confidence interval
            if show_confidence:
                ax.fill_between(episodes,
                                smoothed_rewards - std_devs,
                                smoothed_rewards + std_devs,
                                color=color, alpha=0.2)

            # Original data (optional)
            if show_original:
                ax.plot(episodes, original_rewards, color=color, alpha=0.3,
                        linewidth=0.8, linestyle='--')

        ax.set_xlabel('Episode', fontsize=16)
        ax.set_ylabel('Accuracy (%)', fontsize=16)
        ax.tick_params(axis='both', which='major', labelsize=14)
        ax.grid(True, alpha=0.3)

        if subplot_title:
            ax.set_title(subplot_title, fontsize=16, fontweight='bold')

        return handles, labels

    def plot_two_subplots(self, datasets_top, datasets_bottom, window_size=10,
                          show_confidence=True, show_original=False, figsize=(12, 10),
                          top_title=None, bottom_title=None):
        """Create two subplots with shared legend"""

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=figsize, sharex=True)
        plt.style.use('seaborn-v0_8' if 'seaborn-v0_8' in plt.style.available else 'default')

        # Plot top subplot
        handles_top, labels_top = self.plot_subplot(
            ax1, datasets_top, window_size, show_confidence, show_original, top_title
        )

        # Plot bottom subplot
        handles_bottom, labels_bottom = self.plot_subplot(
            ax2, datasets_bottom, window_size, show_confidence, show_original, bottom_title
        )

        # Create shared legend (using labels from first subplot)
        fig.legend(handles_top, labels_top, loc='lower center',
                   bbox_to_anchor=(0.5, -0.05), ncol=3, fontsize=14)

        plt.tight_layout()
        plt.subplots_adjust(bottom=0.15)  # Make room for legend

        return fig

    def plot_data(self, datasets, window_size=10, show_confidence=True,
                  show_original=False, figsize=(12, 8)):
        """Create plot with smoothed data (original single plot method)"""

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

        plt.xlabel('Episode', fontsize=16)
        plt.ylabel('Accuracy (%)', fontsize=16)
        plt.tick_params(axis='both', which='major', labelsize=14)
        plt.legend(loc='lower center', bbox_to_anchor=(0.5, -0.25), ncol=3, fontsize=14)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()

        return plt


def main():
    parser = argparse.ArgumentParser(description='Smooth and visualize data from CSV files')
    parser.add_argument('files', nargs='*', help='Paths to CSV files (6 files for dual plot, 3 for single)')
    parser.add_argument('--window', type=int, default=10, help='Window size for moving average')
    parser.add_argument('--no-confidence', action='store_true', help='Do not show confidence intervals')
    parser.add_argument('--show-original', action='store_true', help='Also show original data')
    parser.add_argument('--dual-plot', action='store_true', help='Create two subplots (requires 6 files)')
    parser.add_argument('--top-title', help='Title for top subplot')
    parser.add_argument('--bottom-title', help='Title for bottom subplot')
    parser.add_argument('--save', help='Save plot to file (e.g. plot.pdf)')

    args = parser.parse_args()

    smoother = CSVSmoother()

    # Example files if no arguments provided
    if not args.files:
        print("Example usage:")
        print("Single plot: python script.py file1.csv file2.csv file3.csv --window 15")
        print("Dual plot: python script.py file1.csv file2.csv file3.csv file4.csv file5.csv file6.csv --dual-plot")
        print("\nThe script applies moving average smoothing to CSV files.")
        print("CSV files must contain 'episode' and 'reward' columns.")
        print("For dual plot: first 3 files = top plot, next 3 files = bottom plot")
        return

    series_labels = ['normal', 'fixed tasks', 'disabled signaling']

    if args.dual_plot:
        # Dual subplot mode
        if len(args.files) < 6:
            print("Dual plot requires 6 files (3 for each subplot)")
            return

        # Load files for top subplot (first 3)
        datasets_top = []
        for i, filepath in enumerate(args.files[:3]):
            df = smoother.load_csv(filepath)
            label = series_labels[i] if i < len(series_labels) else f'series_{i + 1}'
            datasets_top.append((df, label))

        # Load files for bottom subplot (next 3)
        datasets_bottom = []
        for i, filepath in enumerate(args.files[3:6]):
            df = smoother.load_csv(filepath)
            label = series_labels[i] if i < len(series_labels) else f'series_{i + 1}'
            datasets_bottom.append((df, label))

        # Check if any data was loaded
        if not any(df is not None for df, _ in datasets_top + datasets_bottom):
            print("Failed to load any data!")
            return

        # Create dual plot
        fig = smoother.plot_two_subplots(
            datasets_top, datasets_bottom,
            window_size=args.window,
            show_confidence=not args.no_confidence,
            show_original=args.show_original,
            top_title=args.top_title,
            bottom_title=args.bottom_title
        )

        if args.save:
            # Determine file format from extension, default to PDF for LaTeX
            if args.save.lower().endswith('.png'):
                fig.savefig(args.save, dpi=300, bbox_inches='tight', format='png')
            elif args.save.lower().endswith('.pdf'):
                fig.savefig(args.save, bbox_inches='tight', format='pdf')
            else:
                # Default to PDF for LaTeX compatibility
                pdf_filename = args.save + '.pdf'
                fig.savefig(pdf_filename, bbox_inches='tight', format='pdf')
                print(f"No extension specified, saved as PDF: {pdf_filename}")
                plt.show()
                return
            print(f"Dual plot saved as: {args.save}")

    else:
        # Single plot mode
        datasets = []
        for i, filepath in enumerate(args.files[:3]):  # Maximum 3 files
            df = smoother.load_csv(filepath)
            # Use predefined labels for series
            label = series_labels[i] if i < len(series_labels) else f'series_{i + 1}'
            datasets.append((df, label))

        # Check if any data was loaded
        if not any(df is not None for df, _ in datasets):
            print("Failed to load any data!")
            return

        # Create single plot
        plt_obj = smoother.plot_data(
            datasets,
            window_size=args.window,
            show_confidence=not args.no_confidence,
            show_original=args.show_original
        )

        if args.save:
            # Determine file format from extension, default to PDF for LaTeX
            if args.save.lower().endswith('.png'):
                plt_obj.savefig(args.save, dpi=300, bbox_inches='tight', format='png')
            elif args.save.lower().endswith('.pdf'):
                plt_obj.savefig(args.save, bbox_inches='tight', format='pdf')
            else:
                # Default to PDF for LaTeX compatibility
                pdf_filename = args.save + '.pdf'
                plt_obj.savefig(pdf_filename, bbox_inches='tight', format='pdf')
                print(f"No extension specified, saved as PDF: {pdf_filename}")
                plt.show()
                return
            print(f"Plot saved as: {args.save}")

    plt.show()


if __name__ == "__main__":
    main()

# Example of programmatic usage:
"""
# Example 1: Single plot
smoother = CSVSmoother()
df1 = smoother.load_csv('exp1.csv')
df2 = smoother.load_csv('exp2.csv') 
df3 = smoother.load_csv('exp3.csv')

datasets = [(df1, 'normal'), (df2, 'fixed tasks'), (df3, 'disabled signaling')]
smoother.plot_data(datasets, window_size=15)
plt.show()

# Example 2: Dual plot
datasets_top = [(df1, 'normal'), (df2, 'fixed tasks'), (df3, 'disabled signaling')]
datasets_bottom = [(df4, 'normal'), (df5, 'fixed tasks'), (df6, 'disabled signaling')]

fig = smoother.plot_two_subplots(datasets_top, datasets_bottom, 
                                top_title='Training Accuracy', 
                                bottom_title='Validation Accuracy')
plt.show()
"""
