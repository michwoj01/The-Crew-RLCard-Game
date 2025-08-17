import argparse
import os

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
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
            if '# _tick' not in df.columns or 'mean_episode_return_0' not in df.columns:
                raise ValueError(f"File {filepath} must contain '# _tick' and 'mean_episode_return_0' columns")

            # Clean data and sort
            df = df.dropna(subset=['# _tick', 'mean_episode_return_0'])
            df = df.sort_values('# _tick')

            # Apply transformation (x + 1)/2 to mean_episode_return_0
            df['transformed_return'] = (df['mean_episode_return_0'] + 1) / 2

            # Scale tick values to range [0, 5_000_000]
            tick_min = df['# _tick'].min()
            tick_max = df['# _tick'].max()
            df['scaled_tick'] = (df['# _tick'] - tick_min) / (tick_max - tick_min) * 5_000_000

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
        ticks = df['scaled_tick'].values
        returns = df['transformed_return'].values

        # Convert to percentage
        returns_percent = returns * 100

        smoothed_returns, std_devs = self.moving_average(returns_percent, window_size)

        return ticks, returns_percent, smoothed_returns, std_devs

    def format_x_axis(self, ax):
        """Format X-axis to show millions properly"""

        # Create custom formatter for millions
        def millions_formatter(x, pos):
            return f'{x / 1e6:.0f}M'

        ax.xaxis.set_major_formatter(ticker.FuncFormatter(millions_formatter))
        ax.xaxis.set_major_locator(ticker.MaxNLocator(nbins=6))

    def plot_single_scale(self, ax, datasets, window_size=10, show_confidence=True,
                          show_original=False, ylim=None, title=""):
        """Plot data on a single axis"""

        for i, (df, name) in enumerate(datasets):
            if df is None:
                continue

            ticks, original_returns, smoothed_returns, std_devs = self.apply_smoothing(
                df, window_size
            )

            color = self.colors[i % len(self.colors)]

            # Plot smoothed data
            ax.plot(ticks, smoothed_returns, color=color, linewidth=2.5,
                    label=name, alpha=0.9)

            # Confidence interval
            if show_confidence:
                ax.fill_between(ticks,
                                smoothed_returns - std_devs,
                                smoothed_returns + std_devs,
                                color=color, alpha=0.2, label=f'{name} ±σ')

            # Original data (optional)
            if show_original:
                ax.plot(ticks, original_returns, color=color, alpha=0.3,
                        linewidth=0.8, linestyle='--', label=f'{name} (original)')

        ax.set_xlabel('Frames', fontsize=20)
        ax.set_ylabel('Accuracy (%)', fontsize=20)
        ax.tick_params(axis='both', which='major', labelsize=18)
        ax.grid(True, alpha=0.3)

        # Format X-axis for millions
        self.format_x_axis(ax)

        if ylim:
            ax.set_ylim(ylim)

        if title:
            ax.set_title(title, fontsize=22, pad=20)

    def plot_data(self, datasets, window_size=10, show_confidence=True,
                  show_original=False, figsize=(12, 12)):
        """Create dual-scale plot with full (0-100%) and auto-scaled accuracy ranges"""

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=figsize)
        plt.style.use('seaborn-v0_8' if 'seaborn-v0_8' in plt.style.available else 'default')

        # Full scale plot (0-100%)
        self.plot_single_scale(ax1, datasets, window_size, show_confidence,
                               show_original, ylim=(0, 100), title="Full Scale (0-100%)")

        # Auto-scaled plot (let matplotlib determine the best range)
        self.plot_single_scale(ax2, datasets, window_size, show_confidence,
                               show_original, ylim=None, title="Auto-scaled")

        # Force consistent Y-axis label positioning
        ax1.yaxis.set_label_coords(-0.05, 0.5)
        ax2.yaxis.set_label_coords(-0.05, 0.5)

        # Single legend for both plots
        handles, labels = ax1.get_legend_handles_labels()
        fig.legend(handles, labels, loc='lower center', bbox_to_anchor=(0.5, -0.08),
                   ncol=len(labels) // 2 if len(labels) > 3 else 3, fontsize=18)

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

            ticks, original_returns, smoothed_returns, std_devs = self.apply_smoothing(
                df, window_size
            )

            color = self.colors[i % len(self.colors)]

            # Plot smoothed data
            plt.plot(ticks, smoothed_returns, color=color, linewidth=2.5,
                     label=name, alpha=0.9)

            # Confidence interval
            if show_confidence:
                plt.fill_between(ticks,
                                 smoothed_returns - std_devs,
                                 smoothed_returns + std_devs,
                                 color=color, alpha=0.2, label=f'{name} ±σ')

            # Original data (optional)
            if show_original:
                plt.plot(ticks, original_returns, color=color, alpha=0.3,
                         linewidth=0.8, linestyle='--', label=f'{name} (original)')

        plt.xlabel('Frames', fontsize=18)
        plt.ylabel('Accuracy (%)', fontsize=18)
        plt.tick_params(axis='both', which='major', labelsize=16)
        plt.legend(loc='lower center', bbox_to_anchor=(0.5, -0.25), ncol=3, fontsize=16)
        plt.grid(True, alpha=0.3)

        # Format X-axis for millions
        ax = plt.gca()
        self.format_x_axis(ax)

        plt.tight_layout()

        return plt


def main():
    parser = argparse.ArgumentParser(description='Smooth and visualize data from CSV files')
    parser.add_argument('files', nargs='*', help='Paths to CSV files (maximum 3)')
    parser.add_argument('--window', type=int, default=10, help='Window size for moving average')
    parser.add_argument('--no-confidence', action='store_true', help='Do not show confidence intervals')
    parser.add_argument('--show-original', action='store_true', help='Also show original data')
    parser.add_argument('--save', help='Save plot to file (e.g. plot.png)')
    parser.add_argument('--dual-scale', action='store_true', help='Create dual-scale plot (full + auto-scaled)')

    args = parser.parse_args()

    smoother = CSVSmoother()

    # Example files if no arguments provided
    if not args.files:
        print("Example usage:")
        print("python script.py file1.csv file2.csv file3.csv --window 15 --dual-scale")
        print("\nThe script applies moving average smoothing to CSV files.")
        print("CSV files must contain '# _tick' and 'mean_episode_return_0' columns.")
        print("The transformation (x + 1)/2 is applied to mean_episode_return_0.")
        print("Use --dual-scale to create plots with full and auto-scaled accuracy ranges.")
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
        plot_obj = smoother.plot_data(
            datasets,
            window_size=args.window,
            show_confidence=not args.no_confidence,
            show_original=args.show_original
        )
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

    plt.show()


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
fig = smoother.plot_data(datasets, window_size=15)
plt.show()

# Example 2: Single plot (original behavior)
plt_obj = smoother.plot_data_single(datasets, window_size=20, show_confidence=False)
plt.show()
"""