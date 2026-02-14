import numpy as np
import pandas as pd
import os

class MicrotubuleTorqueAnalyzer:
    def __init__(self):
        self.fiber_results = []
        self.helicity_results = []
        self.angle_profiles = {}

    @staticmethod
    def kmts_torque_in_fiber(fiber_points, center_coords):
        """
        Calculate angular displacement (torque) of fiber segments relative to their polygon center.
        :param fiber_points: (N, 3) array of fiber segment coordinates
        :param center_coords: (N, 2) array of center coordinates for each segment (X, Z)
        :return: normalized angle profile (degrees), raw angle profile (degrees)
        """
        x = fiber_points[:, 0]
        z = fiber_points[:, 2]
        x_c = center_coords[:, 0]
        z_c = center_coords[:, 1]
        angles = np.degrees(np.arctan2(z - z_c, x - x_c))
        norm_angles = angles - angles[0]
        return norm_angles, angles

    @staticmethod
    def fiber_torque_around_center(fiber_points, global_center):
        """
        Calculate angular position of fiber points around a global center.
        :param fiber_points: (N, 3) array
        :param global_center: (2,) array (X, Z)
        :return: normalized angle profile (degrees), raw angle profile (degrees)
        """
        x = fiber_points[:, 0]
        z = fiber_points[:, 2]
        x_c, z_c = global_center
        angles = np.degrees(np.arctan2(z - z_c, x - x_c))
        norm_angles = angles - angles[0]
        return norm_angles, angles

    @staticmethod
    def helicity_of_fiber(pole_pos, kinetochore_pos):
        """
        Calculate helicity: net rotation from pole to kinetochore, normalized by Y-axis height.
        :param pole_pos: (3,) array
        :param kinetochore_pos: (3,) array
        :return: helicity (deg/um), angle (deg), height difference (um), 3D distance (um)
        """
        dx = kinetochore_pos[0] - pole_pos[0]
        dz = kinetochore_pos[2] - pole_pos[2]
        dy = kinetochore_pos[1] - pole_pos[1]
        angle = np.degrees(np.arctan2(dz, dx))
        height_diff = dy if dy != 0 else np.nan
        helicity = angle / height_diff if height_diff != 0 else np.nan
        dist_3d = np.linalg.norm(kinetochore_pos - pole_pos)
        return helicity, angle, height_diff, dist_3d

    def analyze_fiber(self, fiber_id, fiber_points, center_coords, global_center, pole_pos, kinetochore_pos):
        # Fiber torque (segment-wise, relative to local center)
        norm_angles, raw_angles = self.kmts_torque_in_fiber(fiber_points, center_coords)
        mean_angle = np.mean(norm_angles)
        std_angle = np.std(norm_angles)
        min_angle = np.min(norm_angles)
        max_angle = np.max(norm_angles)
        angle_range = max_angle - min_angle
        median_angle = np.median(norm_angles)
        q1 = np.percentile(norm_angles, 25)
        q3 = np.percentile(norm_angles, 75)
        iqr = q3 - q1

        self.fiber_results.append({
            "Fiber_ID": fiber_id,
            "Analysis_Type": "Fiber_Torque",
            "Mean_Angle_deg": mean_angle,
            "Std_Angle_deg": std_angle,
            "Points_Count": len(norm_angles),
            "Min_Angle_deg": min_angle,
            "Max_Angle_deg": max_angle,
            "Angle_Range_deg": angle_range,
            "Median_Angle_deg": median_angle,
            "Q1_Angle_deg": q1,
            "Q3_Angle_deg": q3,
            "IQR_Angle_deg": iqr
        })

        # Helicity
        helicity, angle, height_diff, dist_3d = self.helicity_of_fiber(pole_pos, kinetochore_pos)
        self.helicity_results.append({
            "Fiber_ID": fiber_id,
            "Helicity_deg_per_um": helicity,
            "Angle_deg": angle,
            "Height_Difference_um": height_diff,
            "Distance_3D_um": dist_3d,
            "Abs_Helicity_deg_per_um": abs(helicity) if helicity is not None else np.nan,
            "Helicity_Category": "Positive" if helicity > 0 else "Negative"
        })

        # Store angle profiles for export
        self.angle_profiles[fiber_id] = {
            "Normalized_Angle_deg": norm_angles,
            "Raw_Angle_deg": raw_angles
        }

    def batch_analyze(self, fibers_data):
        for fiber in fibers_data:
            self.analyze_fiber(
                fiber_id=fiber['fiber_id'],
                fiber_points=fiber['fiber_points'],
                center_coords=fiber['center_coords'],
                global_center=fiber['global_center'],
                pole_pos=fiber['pole_pos'],
                kinetochore_pos=fiber['kinetochore_pos']
            )

    def statistical_summary(self):
        df = pd.DataFrame(self.fiber_results)
        helicity_df = pd.DataFrame(self.helicity_results)
        summary = {
            "Fiber_Torque_N_Fibers": len(df),
            "Mean_Angle_Overall_deg": df["Mean_Angle_deg"].mean(),
            "Std_Angle_Overall_deg": df["Mean_Angle_deg"].std(),
            "Mean_Std_Within_Fiber_deg": df["Std_Angle_deg"].mean(),
            "Angle_Min_deg": df["Mean_Angle_deg"].min(),
            "Angle_Max_deg": df["Mean_Angle_deg"].max(),
            "Helicity_N_Fibers": len(helicity_df),
            "Mean_Helicity_deg_per_um": helicity_df["Helicity_deg_per_um"].mean(),
            "Std_Helicity_deg_per_um": helicity_df["Helicity_deg_per_um"].std(),
            "Helicity_Min_deg_per_um": helicity_df["Helicity_deg_per_um"].min(),
            "Helicity_Max_deg_per_um": helicity_df["Helicity_deg_per_um"].max(),
            "Mean_Height_Diff_um": helicity_df["Height_Difference_um"].mean()
        }
        return pd.DataFrame(list(summary.items()), columns=["Metric", "Value"])

    def export_to_excel(self, filename="microtubule_torque_analysis_results.xlsx"):
        with pd.ExcelWriter(filename, engine='openpyxl') as writer:
            # Summary results
            pd.DataFrame(self.fiber_results).to_excel(writer, sheet_name="Summary_Results", index=False)
            # Fiber torque details
            pd.DataFrame(self.fiber_results).to_excel(writer, sheet_name="Fiber_Torque_Details", index=False)
            # Helicity results
            pd.DataFrame(self.helicity_results).to_excel(writer, sheet_name="Helicity_Results", index=False)
            # Statistical summary
            self.statistical_summary().to_excel(writer, sheet_name="Statistical_Summary", index=False)
            # Angle profiles (all fibers)
            angle_df = pd.DataFrame()
            max_len = max(len(v["Normalized_Angle_deg"]) for v in self.angle_profiles.values())
            angle_df["Position_Index"] = np.arange(max_len)
            for fiber_id, profile in self.angle_profiles.items():
                pad_norm = np.pad(profile["Normalized_Angle_deg"], (0, max_len - len(profile["Normalized_Angle_deg"])), constant_values=np.nan)
                pad_raw = np.pad(profile["Raw_Angle_deg"], (0, max_len - len(profile["Raw_Angle_deg"])), constant_values=np.nan)
                angle_df[f"{fiber_id}_Normalized_Angle_deg"] = pad_norm
                angle_df[f"{fiber_id}_Raw_Angle_deg"] = pad_raw
            angle_df.to_excel(writer, sheet_name="Angle_Profiles", index=False)
            # Individual fiber detail sheets
            for fiber_id, profile in self.angle_profiles.items():
                df = pd.DataFrame({
                    "Position_Index": np.arange(len(profile["Normalized_Angle_deg"])),
                    "Raw_Angle_deg": profile["Raw_Angle_deg"],
                    "Normalized_Angle_deg": profile["Normalized_Angle_deg"],
                    "Cumulative_Rotation_deg": np.cumsum(profile["Normalized_Angle_deg"])
                })
                df.to_excel(writer, sheet_name=f"{fiber_id[:28]}_Detail", index=False)
        print(f"Results exported to {filename}")

# Example usage with synthetic data
if __name__ == "__main__":
    np.random.seed(42)
    n_fibers = 8
    n_points = 25
    fibers_data = []
    for i in range(n_fibers):
        # Simulate a twisted fiber in 3D
        y = np.linspace(0, 10, n_points)
        twist = (i - 4) * 0.1  # Vary twist per fiber
        x = np.cos(y * twist) + np.random.normal(0, 0.05, n_points)
        z = np.sin(y * twist) + np.random.normal(0, 0.05, n_points)
        fiber_points = np.stack([x, y, z], axis=1)
        # Simulate local center for each segment (here, just mean X/Z with small noise)
        center_coords = np.stack([np.mean(x) + np.random.normal(0, 0.01, n_points),
                                  np.mean(z) + np.random.normal(0, 0.01, n_points)], axis=1)
        # Global center (mean X/Z)
        global_center = np.array([np.mean(x), np.mean(z)])
        # Pole and kinetochore positions
        pole_pos = np.array([x[0], y[0], z[0]])
        kinetochore_pos = np.array([x[-1], y[-1], z[-1]])
        fibers_data.append({
            "fiber_id": f"fiber_{i+1:02d}",
            "fiber_points": fiber_points,
            "center_coords": center_coords,
            "global_center": global_center,
            "pole_pos": pole_pos,
            "kinetochore_pos": kinetochore_pos
        })
    analyzer = MicrotubuleTorqueAnalyzer()
    analyzer.batch_analyze(fibers_data)
    analyzer.export_to_excel("microtubule_torque_analysis_results.xlsx")
