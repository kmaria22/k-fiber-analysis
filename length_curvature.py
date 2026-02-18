import numpy as np
import pandas as pd
from openpyxl import Workbook

def fiber_length(points):
    diffs = np.diff(points, axis=0)
    seg_lengths = np.linalg.norm(diffs, axis=1)
    return np.sum(seg_lengths)

def menger_curvature(p0, p1, p2):
    a = np.linalg.norm(p0 - p1)
    b = np.linalg.norm(p1 - p2)
    c = np.linalg.norm(p2 - p0)
    s = (a + b + c) / 2
    area = np.sqrt(max(s * (s - a) * (s - b) * (s - c), 0))
    denom = a * b * c
    if denom == 0:
        return 0.0
    return 4 * area / denom

def total_curvature(points):
    path_length = fiber_length(points)
    straight_dist = np.linalg.norm(points[0] - points[-1])
    tortuosity = path_length / straight_dist if straight_dist > 0 else np.nan
    mid_idx = int(np.round(len(points) / 2))
    p0, p1, p2 = points[0], points[mid_idx], points[-1]
    menger_curv = menger_curvature(p0, p1, p2)
    return {'tortuosity': tortuosity, 'menger_curvature': menger_curv}

def local_curvature(points):
    curvatures = []
    for i in range(len(points) - 2):
        kappa = menger_curvature(points[i], points[i+1], points[i+2])
        curvatures.append(kappa)
    return np.array(curvatures)

def classify_ellipse_region(x, z, x0, z0, rx25, rz25, rx50, rz50, rx100, rz100):
    r25 = ((x - x0)**2 / rx25**2) + ((z - z0)**2 / rz25**2) <= 1
    r50 = ((x - x0)**2 / rx50**2) + ((z - z0)**2 / rz50**2) <= 1
    r100 = ((x - x0)**2 / rx100**2) + ((z - z0)**2 / rz100**2) <= 1
    if r25 and r50 and r100:
        return "25%"
    elif r50 and r100:
        return "50%"
    elif r100:
        return "100%"
    else:
        return "100%"

def analyze_fiber(points, pole_coords, kinetochore_coords, ellipse_params, fiber_name="Fiber"):
    length = fiber_length(points)
    total_curv = total_curvature(points)
    local_curv = local_curvature(points)
    mean_local_curv = np.mean(local_curv) if len(local_curv) > 0 else np.nan
    plus_end = np.median(points, axis=0)
    plus_to_kinetochore = np.linalg.norm(plus_end[[0,2]] - kinetochore_coords[[0,2]])
    plus_to_pole = np.linalg.norm(plus_end - pole_coords)
    minus_to_pole = np.linalg.norm(points[0] - pole_coords)
    ellipse_region = classify_ellipse_region(
        plus_end[0], plus_end[2],
        kinetochore_coords[0], kinetochore_coords[2],
        ellipse_params['rx25'], ellipse_params['rz25'],
        ellipse_params['rx50'], ellipse_params['rz50'],
        ellipse_params['rx100'], ellipse_params['rz100']
    )
    return {
        "Fiber_Name": fiber_name,
        "Length": length,
        "Tortuosity": total_curv['tortuosity'],
        "Menger_Curvature": total_curv['menger_curvature'],
        "Mean_Local_Curvature": mean_local_curv,
        "Plus_to_Kinetochore": plus_to_kinetochore,
        "Plus_to_Pole": plus_to_pole,
        "Minus_to_Pole": minus_to_pole,
        "Ellipse_Region": ellipse_region,
        "Local_Curvature_Profile": local_curv
    }

def analyze_all_fibers(fibers_dict, pole_coords, kinetochore_coords, ellipse_params):
    results = []
    local_curvatures = {}
    for name, points in fibers_dict.items():
        res = analyze_fiber(points, pole_coords, kinetochore_coords, ellipse_params, fiber_name=name)
        results.append({k: v for k, v in res.items() if k != "Local_Curvature_Profile"})
        local_curvatures[name] = res["Local_Curvature_Profile"]
    summary_df = pd.DataFrame(results)
    return summary_df, local_curvatures

def export_length_curvature_to_excel(summary_df, local_curvatures, filename="Length_Curvature_Results.xlsx"):
    with pd.ExcelWriter(filename, engine='openpyxl') as writer:
        summary_df.to_excel(writer, sheet_name="Summary", index=False)
        # Local curvature profiles: one sheet per fiber
        for fiber, curv in local_curvatures.items():
            df = pd.DataFrame({"Local_Curvature": curv})
            df.to_excel(writer, sheet_name=f"{fiber[:28]}_Curv", index=False)
    print(f"Results exported to {filename}")

# Example usage
if __name__ == "__main__":
    fibers = {
        "Fiber1": np.random.rand(20, 3) * 10,
        "Fiber2": np.random.rand(25, 3) * 10
    }
    pole = np.array([0, 0, 0])
    kinetochore = np.array([5, 5, 5])
    ellipse_params = {
        'rx25': 2, 'rz25': 2,
        'rx50': 4, 'rz50': 4,
        'rx100': 8, 'rz100': 8
    }
    summary, local_curv_profiles = analyze_all_fibers(fibers, pole, kinetochore, ellipse_params)
    export_length_curvature_to_excel(summary, local_curv_profiles)
