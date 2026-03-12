import numpy as np
import matplotlib.pyplot as plt
import tkinter as tk
from tkinter import filedialog
from load_data_TARDIS import ImportDataFromAmira
from poles import poles
from scipy.interpolate import make_interp_spline

# select dataset
root = tk.Tk()
root.withdraw()
am_path = filedialog.askopenfilename(
    title="Select your Amira Spatial Graph (.am)",
    filetypes=[("Amira files", "*.am")]
)
if not am_path:
    exit()

# load data
data = ImportDataFromAmira(src_am=am_path)
points = data.get_segmented_points()
labels = data.get_labels()

# change fiber here
fiber_name = "Pole2_44"
if fiber_name not in labels:
    raise ValueError(f"Fiber '{fiber_name}' not found. Available fibers: {list(labels.keys())[:10]}")

segment_ids = labels[fiber_name]
print(f"K-fiber {fiber_name} with {len(segment_ids)} KMTs")

# Spindle poles from extra .py file
spindle_pole = np.array(poles["pole1"]).flatten()

# collect and orient KMTs by distance to pole
kmt_coords = []
for seg_id in segment_ids:
    idx = np.where(points[:,0] == seg_id)[0]
    if len(idx) < 2:
        continue
    xyz = points[idx,1:4].astype(float)
    dists = np.linalg.norm(xyz - spindle_pole, axis=1)
    sorted_indices = np.argsort(-dists)  # plus end first
    xyz_sorted = xyz[sorted_indices]
    kmt_coords.append(xyz_sorted)

if len(kmt_coords) == 0:
    raise ValueError("No valid KMTs found")

# Determine number of points along average fiber
avg_len = int(np.round(np.mean([kmt.shape[0] for kmt in kmt_coords])))
num_points = max(avg_len, 2)
fractions = np.linspace(0, 1, num_points)

# compute mean and median fibers
central_mean = []
central_median = []

for f in fractions:
    points_at_fraction = []
    for kmt in kmt_coords:
        L = kmt.shape[0]
        if L < 2:
            continue
        idx = f*(L-1)
        low = int(np.floor(idx))
        high = min(int(np.ceil(idx)), L-1)
        if low == high:
            pt = kmt[low]
        else:
            alpha = idx - low
            pt = (1-alpha)*kmt[low] + alpha*kmt[high]
        points_at_fraction.append(pt)
    if points_at_fraction:
        points_array = np.array(points_at_fraction)
        central_mean.append(np.mean(points_array, axis=0))
        central_median.append(np.median(points_array, axis=0))

central_mean = np.array(central_mean)
central_median = np.array(central_median)

# (Optional: smooth spline)
def smooth_line(coords):
    t = np.linspace(0,1,len(coords))
    spline = make_interp_spline(t, coords, k=3)
    t_smooth = np.linspace(0,1,len(coords)*10)
    return spline(t_smooth)

central_mean_smooth = smooth_line(central_mean)
central_median_smooth = smooth_line(central_median)

# computr fiber length
def compute_length(coords):
    diffs = np.diff(coords, axis=0)
    dists = np.linalg.norm(diffs, axis=1)
    return np.sum(dists) / 10000  

length_mean = compute_length(central_mean_smooth)
length_median = compute_length(central_median_smooth)

print(f"Length of mean fiber (red): {length_mean:.2f} µm")
print(f"Length of median fiber (yellow): {length_median:.2f} µm")

# plot
fig = plt.figure(figsize=(10,8))
ax = fig.add_subplot(111, projection='3d')

# KMTs blue
for kmt in kmt_coords:
    ax.plot(kmt[:,0], kmt[:,1], kmt[:,2], color='blue', alpha=0.7)

# Mman line red
ax.plot(central_mean_smooth[:,0], central_mean_smooth[:,1], central_mean_smooth[:,2],
        color='red', linewidth=3, label=f'Mean fiber ({length_mean:.2f} µm)')

# Median line yellow
ax.plot(central_median_smooth[:,0], central_median_smooth[:,1], central_median_smooth[:,2],
        color='yellow', linewidth=3, label=f'Median fiber ({length_median:.2f} µm)')

ax.set_xlabel('X'); ax.set_ylabel('Y'); ax.set_zlabel('Z')
ax.set_title(f"K-fiber {fiber_name} with mean (red) and median (yellow) central lines")
ax.legend()
plt.show()