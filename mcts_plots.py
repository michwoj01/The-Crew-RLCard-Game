import argparse
import os

import matplotlib.pyplot as plt
import pandas as pd


def create_multiline_plot(csv_files, x_col, y_col, save_file=None):
    """Create a line plot with 4 series from CSV files"""

    labels = ['1 task', '2 tasks', '3 tasks', '4 tasks']
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']  # Blue, orange, green, red

    # Create plot with consistent styling matching the smoother script
    plt.figure(figsize=(12, 8))
    plt.style.use('seaborn-v0_8' if 'seaborn-v0_8' in plt.style.available else 'default')

    # Set background to match smoother script
    plt.gca().set_facecolor('white')
    plt.gcf().patch.set_facecolor('white')

    # Plot each series
    for i, csv_file in enumerate(csv_files[:4]):  # Maximum 4 files
        if not os.path.exists(csv_file):
            print(f"Warning: File {csv_file} not found, skipping...")
            continue

        # Load data
        data = pd.read_csv(csv_file)
        print(f"Loaded {len(data)} rows from {csv_file}")

        # Convert accuracy to percentage
        data[y_col] = data[y_col] * 100

        # Sort by x_col to ensure proper line connection
        data = data.sort_values(x_col)

        # Plot line with consistent styling matching the smoother script
        plt.plot(data[x_col], data[y_col],
                 color=colors[i], linewidth=2.5,
                 label=labels[i], alpha=0.9)
    # Labels and formatting - exactly matching the smoother script
    plt.xlabel('Simulations', fontsize=20)
    plt.ylabel('Accuracy (%)', fontsize=20)  # Added labelpad like in smoother
    plt.tick_params(axis='both', which='major', labelsize=18)

    # Legend - exactly matching the smoother script positioning and styling
    plt.legend(loc='lower center', bbox_to_anchor=(0.5, 0), ncol=4, fontsize=18)

    # Grid - exactly matching the smoother script
    plt.grid(True, alpha=0.3, color='gray')

    # Add frame/spines to match smoother script
    ax = plt.gca()
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_color('black')
        spine.set_linewidth(0.8)

    plt.tight_layout()

    # Save if requested - exactly matching the smoother script logic
    if save_file:
        # Determine file format from extension, default to PDF for LaTeX
        if save_file.lower().endswith('.png'):
            plt.savefig(save_file, dpi=300, bbox_inches='tight', format='png')
        elif save_file.lower().endswith('.pdf'):
            plt.savefig(save_file, bbox_inches='tight', format='pdf')
        else:
            # Default to PDF for LaTeX compatibility
            pdf_filename = save_file + '.pdf'
            plt.savefig(pdf_filename, bbox_inches='tight', format='pdf')
            print(f"No extension specified, saved as PDF: {pdf_filename}")
            return
        print(f"Plot saved as: {save_file}")

    plt.show()


def main():
    parser = argparse.ArgumentParser(description='Multi-series line plot from CSV files')
    parser.add_argument('files', nargs='+', help='CSV file paths (up to 4 files)')
    parser.add_argument('--x-col', required=True, help='X column name')
    parser.add_argument('--y-col', required=True, help='Y column name')
    parser.add_argument('--save', help='Output file name')

    args = parser.parse_args()

    if len(args.files) > 4:
        print("Warning: Only first 4 files will be used")

    create_multiline_plot(args.files, args.x_col, args.y_col, args.save)


if __name__ == "__main__":
    main()

# Example usage:
# python multiplot.py file1.csv file2.csv file3.csv file4.csv --x-col "simulation" --y-col "accuracy" --save results.pdf