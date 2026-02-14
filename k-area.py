import numpy as np
import pandas as pd
from scipy.spatial import distance_matrix

def kinetochore_area_analysis(kinetochore_points_dict):
    results = []
    neighbor_dists_dict = {}
    for kin_id, points in kinetochore_points_dict.items():
        points = np.asarray(points)
        if points.shape[0] < 2:
            area = np.nan
            kmt_density = np.nan
            neighbor_mean = np.nan
            neighbor_std = np.nan
            fiber_radius = np.nan
            neighbor_dists = np.array([])
        else:
            center = np.mean(points, axis=0)
            radial_dists = np.linalg.norm(points - center, axis=1)
            fiber_radius = np.max(radial_dists)
            area = np.pi * fiber_radius**2
            kmt_density = points.shape[0] / area if area > 0 else np.nan
            dists = distance_matrix(points, points)
            np.fill_diagonal(dists, np.nan)
            nearest_neighbors = np.nanmin(dists, axis=1)
            neighbor_mean = np.mean(nearest_neighbors)
            neighbor_std = np.std(nearest_neighbors)
            neighbor_dists = nearest_neighbors
        results.append({
            "Kinetochore_ID": kin_id,
            "Kinetochore_area": area,
            "KMT_no": points.shape[0],
            "Fiber_radius": fiber_radius,
            "KMT_density": kmt_density,
            "Neighbor_mean": neighbor_mean,
            "Neighbor_std": neighbor_std
        })
        neighbor_dists_dict[kin_id] = neighbor_dists
    summary_df = pd.DataFrame(results)
    return summary_df, neighbor_dists_dict

def summarize_kinetochore_area_stats(summary_df):
    stats = {
        "area_mean": summary_df["Kinetochore_area"].mean(),
        "area_median": summary_df["Kinetochore_area"].median(),
        "area_std": summary_df["Kinetochore_area"].std(),
        "area_min": summary_df["Kinetochore_area"].min(),
        "area_max": summary_df["Kinetochore_area"].max(),
        "density_mean": summary_df["KMT_density"].mean(),
        "neighbor_mean": summary_df["Neighbor_mean"].mean(),
        "neighbor_std": summary_df["Neighbor_mean"].std()
    }
    return stats

def export_kinetochore_area_to_excel(summary_df, neighbor_dists_dict, stats, filename="Kinetochore_Area_Results.xlsx"):
    with pd.ExcelWriter(filename, engine='openpyxl') as writer:
        summary_df.to_excel(writer, sheet_name="Area_Summary", index=False)
        pd.DataFrame([stats]).to_excel(writer, sheet_name="Summary_Statistics", index=False)
        # Each kinetochore's neighbor distances in a separate sheet
        for kin_id, dists in neighbor_dists_dict.items():
            df = pd.DataFrame({"Neighbor_Distance": dists})
            df.to_excel(writer, sheet_name=f"{kin_id[:28]}_Neighbors", index=False)
    print(f"Results exported to {filename}")

# Example usage
if __name__ == "__main__":
    np.random.seed(0)
    kinetochore_points_dict = {
        f"Kinetochore_{i+1}": np.random.normal(loc=0, scale=0.2, size=(np.random.randint(6, 15), 3))
        for i in range(5)
    }
    summary_df, neighbor_dists_dict = kinetochore_area_analysis(kinetochore_points_dict)
    stats = summarize_kinetochore_area_stats(summary_df)
    export_kinetochore_area_to_excel(summary_df, neighbor_dists_dict, stats)
