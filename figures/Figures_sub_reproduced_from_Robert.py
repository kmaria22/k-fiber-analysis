###############################################################################
# Plots for eLife paper                                                       #
#                                                                             #
# (c) 2019-2021 Kiewisz                                                       #
# Converted to Python 2026                                                    #
# Author: Robert Kiewisz (original), [Your Name] (conversion)                 #
# License: GPL V3.0                                                           #
###############################################################################

import os
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pearsonr, linregress
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

def save_figure(fig, outdir, name):
    fig.savefig(outdir / f"{name}.png", bbox_inches='tight', dpi=300)
    plt.close(fig)

def euclidean_distance(row):
    """Compute 3D Euclidean distance between two points in a row."""
    p1 = np.array([row['X_Coord_P1'], row['Y_Coord_P1'], row['Z_Coord_P1']])
    p2 = np.array([row['X_Coord_P2'], row['Y_Coord_P2'], row['Z_Coord_P2']])
    return np.linalg.norm(p1 - p2)

def assign_mt_type(df):
    """Assign microtubule type based on custom logic."""
    # Placeholder: Replace with actual logic from R's Assign_MT_Type
    # For demonstration, assign randomly
    df['Type'] = np.where(df['SomeColumn'] > 0, 'KMT', 'SMT')
    return df

def fwhm(x, y):
    """Calculate Full Width at Half Maximum (FWHM) for a curve."""
    half_max = np.max(y) / 2.0
    spline = UnivariateSpline(x, y - half_max, s=0)
    roots = spline.roots()
    if len(roots) >= 2:
        return abs(roots[-1] - roots[0])
    return np.nan

# --- Main Analysis Functions ---

def figure_pole_to_pole_distance(outdir):
    # Data as in R script (hardcoded, divided by 10000)
    data = {
        'X_Coord_P1': [51385.63281, 52571.19531, 80278.8125],
        'Y_Coord_P1': [18100.44141, 113530.8203, 81406.53906],
        'Z_Coord_P1': [25617.02148, 28686.98242, 17718.85156],
        'X_Coord_P2': [51282.79297, 52565.24609, 80279.78906],
        'Y_Coord_P2': [89695.32031, 9660.630859, -13390.78125],
        'Z_Coord_P2': [24732.59766, 28693.30469, 17719.81055]
    }
    df = pd.DataFrame(data) / 10000
    df['distance'] = df.apply(euclidean_distance, axis=1)
    # Plot
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.bar(['Metaphase #1', 'Metaphase #2', 'Metaphase #3'], df['distance'], color='brown')
    ax.set_ylabel('Pole-to-Pole Distance (μm)')
    ax.set_title('Pole-to-Pole Distance')
    sns.despine()
    save_figure(fig, outdir, 'Figure_Supp_2B_Pole_to_Pole_Distance')

def figure_kmt_number_distribution(dfs, outdir):
    # Example: KMT number per kinetochore
    # Replace 'Data_1_KMT_No' etc. with actual base names
    kmt_counts = []
    for key in dfs:
        if 'KMT_No' in key:
            kmt_counts.extend(dfs[key]['KMT_No'].values)
    fig, ax = plt.subplots(figsize=(5, 4))
    sns.histplot(kmt_counts, bins=20, color='dodgerblue', ax=ax)
    ax.set_xlabel('KMTs per Kinetochore')
    ax.set_ylabel('Count')
    ax.set_title('Distribution of KMTs per Kinetochore')
    sns.despine()
    save_figure(fig, outdir, 'Figure_Supp_KMT_Number_Distribution')

def figure_ik_distance_vs_kmt_number(dfs, outdir):
    # Example: Inter-kinetochore distance vs. KMT number
    # Replace with actual data columns
    ik_dist = []
    kmt_no = []
    for key in dfs:
        if 'KMT_No' in key:
            ik_dist.extend(dfs[key]['IK_Distance'].values)
            kmt_no.extend(dfs[key]['KMT_No'].values)
    fig, ax = plt.subplots(figsize=(5, 4))
    sns.regplot(x=kmt_no, y=ik_dist, ax=ax, scatter_kws={'color':'purple'}, line_kws={'color':'black'})
    ax.set_xlabel('KMTs per Kinetochore')
    ax.set_ylabel('Inter-Kinetochore Distance (μm)')
    ax.set_title('IK Distance vs. KMT Number')
    # Pearson correlation
    r, p = pearsonr(kmt_no, ik_dist)
    ax.annotate(f"r = {r:.2f}, p = {p:.3g}", xy=(0.05, 0.95), xycoords='axes fraction', ha='left', va='top')
    sns.despine()
    save_figure(fig, outdir, 'Figure_Supp_IK_Distance_vs_KMT_Number')

def figure_smt_end_density(dfs, outdir):
    # Example: SMT end positions density
    # Replace with actual data columns
    smt_ends = []
    for key in dfs:
        if 'SMT' in key:
            smt_ends.extend(dfs[key]['End_Position'].values)
    fig, ax = plt.subplots(figsize=(5, 4))
    sns.kdeplot(smt_ends, fill=True, color='orange', ax=ax)
    ax.set_xlabel('SMT End Position (μm)')
    ax.set_ylabel('Density')
    ax.set_title('SMT End Position Density')
    sns.despine()
    save_figure(fig, outdir, 'Figure_Supp_SMT_End_Density')

def figure_kmt_length_distribution(dfs, outdir):
    # Example: KMT length distribution
    kmt_lengths = []
    for key in dfs:
        if 'KMT' in key and 'Length' in dfs[key].columns:
            kmt_lengths.extend(dfs[key]['Length'].values)
    fig, ax = plt.subplots(figsize=(5, 4))
    sns.histplot(kmt_lengths, bins=30, color='seagreen', ax=ax)
    ax.set_xlabel('KMT Length (μm)')
    ax.set_ylabel('Count')
    ax.set_title('KMT Length Distribution')
    sns.despine()
    save_figure(fig, outdir, 'Figure_Supp_KMT_Length_Distribution')

# --- Main Workflow ---

def main():
    print("Select your data folder containing the ASGA output files.")
    data_folder = get_data_folder()
    validate_folder(data_folder)
    # Define base names as in your ASGA output (update as needed)
    base_names = [
        'Data_1_KMT_No', 'Data_2_KMT_No', 'Data_3_KMT_No',
        'Data_1_SMT', 'Data_2_SMT', 'Data_3_SMT',
        'Data_1_KMT', 'Data_2_KMT', 'Data_3_KMT'
    ]
    dfs = load_dataframes(data_folder, base_names)
    outdir = data_folder / "figures"
    outdir.mkdir(exist_ok=True)
    # Generate all figures
    figure_pole_to_pole_distance(outdir)
    figure_kmt_number_distribution(dfs, outdir)
    figure_ik_distance_vs_kmt_number(dfs, outdir)
    figure_smt_end_density(dfs, outdir)
    figure_kmt_length_distribution(dfs, outdir)
    print(f"All figures saved in: {outdir}")

if __name__ == "__main__":
    main()
