import os
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pearsonr
from scipy.signal import find_peaks
from scipy.interpolate import UnivariateSpline

# Optional: GUI folder selection
try:
    import tkinter as tk
    from tkinter import filedialog
    GUI_AVAILABLE = True
except ImportError:
    GUI_AVAILABLE = False

# --- Helper Functions ---

def get_data_folder():
    """Prompt user to select or enter data folder path."""
    if GUI_AVAILABLE:
        root = tk.Tk()
        root.withdraw()
        folder = filedialog.askdirectory(title='Select Data Folder')
        if folder:
            return Path(folder)
    folder = input("Enter the path to your data folder: ").strip()
    return Path(folder)

def validate_folder(folder_path):
    if not folder_path.exists() or not folder_path.is_dir():
        raise FileNotFoundError(f"Folder '{folder_path}' does not exist or is not a directory.")
    return True

def find_data_files(folder_path, patterns):
    """Find files matching any of the provided patterns."""
    files = []
    for pattern in patterns:
        files.extend(list(folder_path.glob(pattern)))
    if not files:
        raise FileNotFoundError(f"No files matching {patterns} found in '{folder_path}'.")
    return files

def load_dataframes(folder, base_names, extensions=['.xlsx', '.csv', '.txt']):
    """Load all required dataframes for each base name."""
    dfs = {}
    for base in base_names:
        for ext in extensions:
            file = folder / f"{base}{ext}"
            if file.exists():
                if ext == '.xlsx':
                    dfs[base] = pd.read_excel(file)
                elif ext == '.csv':
                    dfs[base] = pd.read_csv(file)
                elif ext == '.txt':
                    dfs[base] = pd.read_csv(file, sep='\t')
                break
    return dfs

def plot_histogram(data, column, bins, xlabel, ylabel, title, outpath):
    plt.figure(figsize=(6,4))
    sns.histplot(data[column], bins=bins, color='skyblue')
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(outpath)
    plt.close()

def plot_scatter(x, y, xlabel, ylabel, title, outpath, color='brown', marker='o', jitter=False):
    plt.figure(figsize=(6,4))
    if jitter:
        x = x + np.random.normal(0, 0.02, size=len(x))
        y = y + np.random.normal(0, 0.02, size=len(y))
    plt.scatter(x, y, c=color, marker=marker, alpha=0.7)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(outpath)
    plt.close()

def plot_density(data, column, xlabel, ylabel, title, outpath):
    plt.figure(figsize=(6,4))
    sns.kdeplot(data[column], fill=True, color='skyblue')
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(outpath)
    plt.close()

def plot_smooth(x, y, xlabel, ylabel, title, outpath, color='brown'):
    plt.figure(figsize=(6,4))
    sns.regplot(x=x, y=y, lowess=True, scatter_kws={'s':10, 'alpha':0.5}, line_kws={'color':color})
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(outpath)
    plt.close()

def print_stats(data, column, label):
    mean = np.round(data[column].mean(), 2)
    std = np.round(data[column].std(), 2)
    print(f"{label} Mean: {mean}, STD: {std}")
    return mean, std

def correlation(x, y):
    corr, _ = pearsonr(x, y)
    print(f"Correlation: {corr:.3f}")
    return corr

def fwhm(x, y):
    """Full width at half maximum for a density curve."""
    y = np.array(y)
    x = np.array(x)
    half_max = np.max(y) / 2.0
    indices = np.where(y >= half_max)[0]
    if len(indices) < 2:
        return np.nan
    return x[indices[-1]] - x[indices[0]]

# --- Main Analysis Functions for Each Figure ---

def figure_3B_KMT_distribution(dfs, outdir):
    # Combine all KMT_No datasets
    df = pd.concat([dfs['Data_1_KMT_No'], dfs['Data_2_KMT_No'], dfs['Data_3_KMT_No']], ignore_index=True)
    plot_histogram(df, 'KMTs_per_kinetochore', bins=12,
                   xlabel='No. of KMTs per kinetochore', ylabel='Frequency',
                   title='Distribution of KMTs per Kinetochore',
                   outpath=outdir/'Figure_3B_KMT_distribution.png')
    print_stats(df, 'KMTs_per_kinetochore', 'KMTs per kinetochore')

def figure_3C_IKD_vs_KMT(dfs, outdir):
    # Combine all IKD_KMT_No datasets
    df = pd.concat([dfs['Data_1_IKD_KMT_No'], dfs['Data_2_IKD_KMT_No'], dfs['Data_3_IKD_KMT_No']], ignore_index=True)
    plot_scatter(df['Inter-kinetochore distance'], df['KMTs no.'],
                 xlabel='Inter-kinetochore distance', ylabel='KMTs number',
                 title='IKD vs KMT Number',
                 outpath=outdir/'Figure_3C_IKD_vs_KMT.png')
    correlation(df['Inter-kinetochore distance'], df['KMTs no.'])

def figure_3D_IKD_vs_Delta(dfs, outdir):
    df = pd.concat([dfs['Data_1_IKD_KMT_Delta'], dfs['Data_2_IKD_KMT_Delta'], dfs['Data_3_IKD_KMT_Delta']], ignore_index=True)
    plot_scatter(df['Inter-kinetochore distance'], df['Delta of KMTs'],
                 xlabel='Inter-kinetochore distance', ylabel='Delta of KMTs',
                 title='IKD vs Delta of KMTs',
                 outpath=outdir/'Figure_3D_IKD_vs_Delta.png')
    correlation(df['Inter-kinetochore distance'], df['Delta of KMTs'])

def figure_4B_SMT_density(dfs, outdir):
    # Density plot and FWHM calculation
    df = pd.concat([dfs['Data_1_SMT_Ends'], dfs['Data_2_SMT_Ends'], dfs['Data_3_SMT_Ends']], ignore_index=True)
    plot_density(df, 'Distance_to_Pole', xlabel='Distance to Pole', ylabel='Density',
                 title='SMT Ends Density Distribution',
                 outpath=outdir/'Figure_4B_SMT_density.png')
    # FWHM calculation
    density = sns.kdeplot(df['Distance_to_Pole'], fill=True).get_lines()[0].get_data()
    plt.close()
    width = fwhm(density[0], density[1])
    print(f"FWHM (μm): {width:.2f}")

def figure_4C_length_distribution(dfs, outdir):
    df = pd.concat([dfs['Data_1_LD'], dfs['Data_2_LD'], dfs['Data_3_LD']], ignore_index=True)
    plot_histogram(df, 'length', bins=30,
                   xlabel='Length [μm]', ylabel='No. of KMTs',
                   title='KMT Length Distribution',
                   outpath=outdir/'Figure_4C_length_distribution.png')
    print_stats(df, 'length', 'KMT Length')

def figure_4D_minus_ends_distance(dfs, outdir):
    df = pd.concat([dfs['Data_1_LD'], dfs['Data_2_LD'], dfs['Data_3_LD']], ignore_index=True)
    plot_histogram(df, 'minus_dist_to_pole', bins=30,
                   xlabel='Minus ends distance to the pole [μm]', ylabel='No. of KMTs',
                   title='Minus Ends Distance to Pole',
                   outpath=outdir/'Figure_4D_minus_ends_distance.png')
    print_stats(df, 'minus_dist_to_pole', 'Minus End Distance')

def figure_4E_relative_positions(dfs, outdir):
    df = pd.concat([dfs['Data_1_KMT_Minus_End_0'], dfs['Data_2_KMT_Minus_End_0'], dfs['Data_3_KMT_Minus_End_0']], ignore_index=True)
    plot_histogram(df, 'Relative_position', bins=30,
                   xlabel='Minus ends distance to the pole [μm]', ylabel='No. of KMTs',
                   title='KMT Minus End Relative Positions',
                   outpath=outdir/'Figure_4E_relative_positions.png')
    print_stats(df, 'Relative_position', 'Relative Position')

def figure_4F_SMT_length_distribution(dfs, outdir):
    df = pd.concat([dfs['Data_1_SMT_Ends'], dfs['Data_2_SMT_Ends'], dfs['Data_3_SMT_Ends']], ignore_index=True)
    plot_histogram(df, 'Length', bins=30,
                   xlabel='Length [μm]', ylabel='No. of SMTs',
                   title='Non-KMT Length Distribution',
                   outpath=outdir/'Figure_4F_SMT_length_distribution.png')
    print_stats(df, 'Length', 'SMT Length')

def figure_4G_SMT_distance_to_pole(dfs, outdir):
    df = pd.concat([dfs['Data_1_SMT_Ends'], dfs['Data_2_SMT_Ends'], dfs['Data_3_SMT_Ends']], ignore_index=True)
    plot_histogram(df, 'Distance_to_Pole', bins=30,
                   xlabel='Distance to Pole [μm]', ylabel='No. of SMTs',
                   title='SMT Distance to Pole',
                   outpath=outdir/'Figure_4G_SMT_distance_to_pole.png')
    print_stats(df, 'Distance_to_Pole', 'SMT Distance to Pole')

def figure_4H_SMT_relative_positions(dfs, outdir):
    df = pd.concat([dfs['Data_1_SMT_Ends'], dfs['Data_2_SMT_Ends'], dfs['Data_3_SMT_Ends']], ignore_index=True)
    plot_histogram(df, 'Relativ_Position', bins=30,
                   xlabel='Relative Position', ylabel='No. of SMTs',
                   title='SMT Relative Positions',
                   outpath=outdir/'Figure_4H_SMT_relative_positions.png')
    print_stats(df, 'Relativ_Position', 'SMT Relative Position')

def figure_5F_KMT_curvature_distribution(dfs, outdir):
    df = pd.concat([dfs['Data_1_KMT_Total_Curv'], dfs['Data_2_KMT_Total_Curv'], dfs['Data_3_KMT_Total_Curv']], ignore_index=True)
    plot_histogram(df, 'Curvature', bins=30,
                   xlabel='Tortuosity of KMTs', ylabel='No. of KMTs',
                   title='KMT Total Curvature Distribution',
                   outpath=outdir/'Figure_5F_KMT_curvature_distribution.png')
    print_stats(df, 'Curvature', 'KMT Curvature')

def figure_5G_length_vs_curvature(dfs, outdir):
    df = pd.concat([dfs['Data_1_KMT_Total_Curv'], dfs['Data_2_KMT_Total_Curv'], dfs['Data_3_KMT_Total_Curv']], ignore_index=True)
    plot_scatter(df['KMTs length'], df['Curvature'],
                 xlabel='KMTs length', ylabel='Curvature',
                 title='KMT Length vs Curvature',
                 outpath=outdir/'Figure_5G_length_vs_curvature.png', jitter=True)
    correlation(df['KMTs length'], df['Curvature'])

def figure_5I_relative_position_vs_local_curvature(dfs, outdir):
    df = pd.concat([dfs['Data_1_KMT_Local_Curv'], dfs['Data_2_KMT_Local_Curv'], dfs['Data_3_KMT_Local_Curv']], ignore_index=True)
    plot_scatter(df['Relative_Position'], df['Curvature'],
                 xlabel='Relative Position', ylabel='Curvature',
                 title='Relative Position vs Local Curvature',
                 outpath=outdir/'Figure_5I_relative_position_vs_local_curvature.png', jitter=True)
    correlation(df['Relative_Position'], df['Curvature'])

def figure_6B_fiber_area(dfs, outdir):
    df = pd.concat([dfs['Data_1_Fiber_Area'], dfs['Data_2_Fiber_Area'], dfs['Data_3_Fiber_Area']], ignore_index=True)
    plot_smooth(df['Relative_position'], df['Alpha_area'],
                xlabel='Relative Position', ylabel='Alpha Area',
                title='Fiber Area vs Relative Position',
                outpath=outdir/'Figure_6B_fiber_area.png')

def figure_6D_KMT_density(dfs, outdir):
    df = pd.concat([dfs['Data_1_N_Density'], dfs['Data_2_N_Density'], dfs['Data_3_N_Density']], ignore_index=True)
    plot_smooth(df['Relative_position'], df['Focused KMTs %'],
                xlabel='Relative Position', ylabel='Focused KMTs %',
                title='KMT Density vs Relative Position',
                outpath=outdir/'Figure_6D_KMT_density.png')

# --- Main Workflow ---

def main():
    print("Select or enter the path to your data folder.")
    data_folder = get_data_folder()
    try:
        validate_folder(data_folder)
    except Exception as e:
        print(f"Folder validation error: {e}")
        return

    # Output directory for figures
    outdir = data_folder / "figures"
    outdir.mkdir(exist_ok=True)

    # List all required base data names (as in R script)
    base_names = [
        'Data_1_KMT_No', 'Data_2_KMT_No', 'Data_3_KMT_No',
        'Data_1_IKD_KMT_No', 'Data_2_IKD_KMT_No', 'Data_3_IKD_KMT_No',
        'Data_1_IKD_KMT_Delta', 'Data_2_IKD_KMT_Delta', 'Data_3_IKD_KMT_Delta',
        'Data_1_SMT_Ends', 'Data_2_SMT_Ends', 'Data_3_SMT_Ends',
        'Data_1_LD', 'Data_2_LD', 'Data_3_LD',
        'Data_1_KMT_Minus_End_0', 'Data_2_KMT_Minus_End_0', 'Data_3_KMT_Minus_End_0',
        'Data_1_KMT_Total_Curv', 'Data_2_KMT_Total_Curv', 'Data_3_KMT_Total_Curv',
        'Data_1_KMT_Local_Curv', 'Data_2_KMT_Local_Curv', 'Data_3_KMT_Local_Curv',
        'Data_1_Fiber_Area', 'Data_2_Fiber_Area', 'Data_3_Fiber_Area',
        'Data_1_N_Density', 'Data_2_N_Density', 'Data_3_N_Density'
    ]
    # Load all dataframes
    dfs = load_dataframes(data_folder, base_names)

        # --- Generate all figures ---
    print("\nGenerating all figures and statistics...\n")
    figure_functions = [
        figure_3B_KMT_distribution,
        figure_3C_IKD_vs_KMT,
        figure_3D_IKD_vs_Delta,
        figure_4B_SMT_density,
        figure_4C_length_distribution,
        figure_4D_minus_ends_distance,
        figure_4E_relative_positions,
        figure_4F_SMT_length_distribution,
        figure_4G_SMT_distance_to_pole,
        figure_4H_SMT_relative_positions,
        figure_5F_KMT_curvature_distribution,
        figure_5G_length_vs_curvature,
        figure_5I_relative_position_vs_local_curvature,
        figure_6B_fiber_area,
        figure_6D_KMT_density
    ]
    for func in figure_functions:
        try:
            func(dfs, outdir)
        except KeyError as e:
            print(f"Skipping {func.__name__}: missing dataset {e}")
        except Exception as e:
            print(f"Error in {func.__name__}: {e}")
    print(f"\nAll figures saved to: {outdir}")

if __name__ == "__main__":
    main()
 
