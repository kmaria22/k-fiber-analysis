import re
import numpy as np
import pandas as pd

class MicrotubulePreAnalysis:
    """
    Pre-analysis tools for microtubule and kinetochore data.
    All functions are direct Python equivalents of the provided R scripts.
    """

    def __init__(self, segments, points, pole1, pole2, fiber_prefix="Pole"):
        """
        segments: pandas DataFrame with segment information (from Amira)
        points: pandas DataFrame with all point coordinates (must include 'Point_ID', 'X', 'Y', 'Z')
        pole1, pole2: numpy arrays or DataFrames with pole coordinates (shape: (1,3) or (3,))
        fiber_prefix: prefix for fiber columns (default: 'Pole')
        """
        self.segments = segments
        self.points = points
        self.pole1 = np.array(pole1).flatten()
        self.pole2 = np.array(pole2).flatten()
        self.fiber_prefix = fiber_prefix

    # --- 1. Select Points ---
    @staticmethod
    def select_points(row_idx, df):
        """
        Extracts Point_IDs from a string in the second column of df at row row_idx.
        Returns a DataFrame with a single column 'Point_ID'.
        """
        raw_string = str(df.iloc[row_idx, 1])
        cleaned = re.sub(r'\D', ',', raw_string)
        split_ids = [s for s in cleaned.split(',') if s]
        return pd.DataFrame({'Point_ID': split_ids})

    # --- 2. Find XYZ ---
    def find_xyz(self, points_df):
        """
        Joins a DataFrame of Point_IDs with the master points table to assign XYZ coordinates.
        Returns a DataFrame with Point_ID and XYZ columns, all numeric.
        """
        points_df['Point_ID'] = pd.to_numeric(points_df['Point_ID'])
        merged = pd.merge(points_df, self.points, on='Point_ID', how='left')
        return merged.apply(pd.to_numeric, errors='coerce')

    # --- 3. Sort by Fiber ---
    def sort_by_fiber(self, fiber_col):
        """
        Filters segments for rows where any column starting with fiber_col has value >= 1.
        Returns DataFrame with [first column, 'Point IDs', 'length'].
        """
        fiber_cols = [col for col in self.segments.columns if col.startswith(fiber_col)]
        filtered = self.segments[self.segments[fiber_cols].ge(1).any(axis=1)]
        cols = [self.segments.columns[0], "Point IDs", "length"]
        return filtered.loc[:, cols]

    # --- 4. Sort All Points to Start From the Kinetochore ---
    def sort_by_distance_to_pole(self, df, pole):
        """
        Sorts points in df so that the point closest to the pole is first.
        If the first point is closer to the pole than the last, reverse the order.
        """
        if df.shape[0] < 2:
            return df
        first = df.iloc[0][['X', 'Y', 'Z']].values.astype(float)
        last = df.iloc[-1][['X', 'Y', 'Z']].values.astype(float)
        pole = np.array(pole).flatten().astype(float)
        dist_first = np.linalg.norm(first - pole)
        dist_last = np.linalg.norm(last - pole)
        if dist_first < dist_last:
            return df.iloc[::-1].reset_index(drop=True)
        else:
            return df.reset_index(drop=True)

    def sort_by_distance_to_pole1(self, df):
        return self.sort_by_distance_to_pole(df, self.pole1)

    def sort_by_distance_to_pole2(self, df):
        return self.sort_by_distance_to_pole(df, self.pole2)

    # --- 5. Relative Position Calculation ---
    def relative_pos(self, x, y, pole):
        """
        Calculates the relative position of points in y between kinetochore and pole.
        x: DataFrame with reference values (e.g., fiber info)
        y: DataFrame with points (must have 'Y' column)
        pole: numpy array or list with pole coordinates
        Returns y with an added 'Relative_Position' column.
        """
        delta_y = y['Y'].astype(float) - pole[1]
        denom = float(x.iloc[0, 2]) - pole[1]
        y = y.copy()
        y['Relative_Position'] = np.round(delta_y / denom, 2)
        return y

    def relative_pos_1(self, x, y):
        return self.relative_pos(x, y, self.pole1)

    def relative_pos_2(self, x, y):
        return self.relative_pos(x, y, self.pole2)

    # --- 6. Kinetochore Position ---
    def kinetochore_position(self):
        """
        Calculates the mean position of all kinetochores and the mean pole position.
        Returns two dicts: {'X':..., 'Y':..., 'Z':...} for kinetochores and poles.
        """
        plus_ends = []
        for fiber_col in [col for col in self.segments.columns if col.startswith(self.fiber_prefix)]:
            try:
                fiber = self.sort_by_fiber(fiber_col)
                for idx in range(fiber.shape[0]):
                    points = self.select_points(idx, fiber)
                    xyz = self.find_xyz(points)
                    if xyz.shape[0] > 0:
                        plus_ends.append(xyz.iloc[0][['X', 'Y', 'Z']].values.astype(float))
            except Exception:
                continue
        if not plus_ends:
            return None, None
        plus_ends = np.array(plus_ends)
        kin_median = np.median(plus_ends, axis=0)
        kin_mean = np.mean(plus_ends, axis=0)
        pole_avg = np.mean(np.vstack([self.pole1, self.pole2]), axis=0)
        return {'X': kin_mean[0], 'Y': kin_mean[1], 'Z': kin_mean[2]}, \
               {'X': pole_avg[0], 'Y': pole_avg[1], 'Z': pole_avg[2]}

    # --- 7. Length Distribution Analysis ---
    def analyse_ld(self, x, y, kinetochore_projected, rx25, rz25, rx50, rz50, rx100, rz100):
        """
        Analyses length and end distances for a fiber.
        x: index of fiber in segments
        y: DataFrame with pole coordinates (1 row, 3 columns)
        kinetochore_projected: array-like, projected kinetochore position
        rx25, rz25, rx50, rz50, rx100, rz100: ellipse radii
        Returns DataFrame with analysis results.
        """
        fiber = self.sort_by_fiber(self.segments.columns[x])
        plus_ends = []
        for idx in range(fiber.shape[0]):
            points = self.select_points(idx, fiber)
            xyz = self.find_xyz(points)
            if xyz.shape[0] > 0:
                plus_ends.append(xyz.iloc[0][['X', 'Y', 'Z']].values.astype(float))
        if not plus_ends:
            return pd.DataFrame()
        plus_ends = np.array(plus_ends)
        plus_median = np.median(plus_ends, axis=0)
        # Distance calculations
        plus_dist_to_kin = np.sqrt((plus_median[0] - kinetochore_projected[0])**2 +
                                   (plus_median[2] - kinetochore_projected[2])**2)
        plus_dist_to_pole = np.linalg.norm(plus_median - y.values.flatten())
        # Ellipse checks
        r25 = ((plus_median[0] - kinetochore_projected[0])**2 / rx25**2 +
               (plus_median[2] - kinetochore_projected[2])**2 / rz25**2) <= 1
        r50 = ((plus_median[0] - kinetochore_projected[0])**2 / rx50**2 +
               (plus_median[2] - kinetochore_projected[2])**2 / rz50**2) <= 1
        r100 = ((plus_median[0] - kinetochore_projected[0])**2 / rx100**2 +
                (plus_median[2] - kinetochore_projected[2])**2 / rz100**2) <= 1
        ellipse = 0
        if r25:
            ellipse = 25
        elif r50:
            ellipse = 50
        elif r100:
            ellipse = 100
        # Distance from minus end to pole
        minus_end = plus_ends[-1]
        minus_dist_to_pole = np.linalg.norm(minus_end - y.values.flatten())
        # Aggregate results
        result = pd.DataFrame({
            'Plus_Dist_to_Kinetochore': [plus_dist_to_kin],
            'Plus_Dist_to_Pole': [plus_dist_to_pole],
            'Ellipse': [ellipse],
            'Minus_Dist_to_Pole': [minus_dist_to_pole]
        })
        return result

    # --- 8. Main Orchestrator ---
    def pre_analysis(self):
        """
        Main orchestrator: runs the full pre-analysis pipeline after data loading.
        Returns a dictionary with all results.
        """
        results = {}
        fiber_cols = [col for col in self.segments.columns if col.startswith(self.fiber_prefix)]
        for fiber_col in fiber_cols:
            try:
                fiber = self.sort_by_fiber(fiber_col)
                fiber_results = []
                for idx in range(fiber.shape[0]):
                    try:
                        points = self.select_points(idx, fiber)
                        xyz = self.find_xyz(points)
                        xyz1 = self.sort_by_distance_to_pole1(xyz)
                        xyz2 = self.sort_by_distance_to_pole2(xyz)
                        # Add more analysis as needed, e.g., length distribution, relative position
                        fiber_results.append({
                            'xyz': xyz,
                            'xyz_sorted_pole1': xyz1,
                            'xyz_sorted_pole2': xyz2
                        })
                    except Exception as e:
                        print(f"Error processing MT {idx} in {fiber_col}: {e}")
                        continue
                results[fiber_col] = fiber_results
            except Exception as e:
                print(f"Error processing fiber {fiber_col}: {e}")
                continue
        return results
    
if __name__ == "__main__":
    print("microtubule_pre_analysis.py started...")

    try:
        # Import data loaded earlier
        from load_data_TARDIS import segments, points, pole1, pole2
        print("Data imported successfully.")
    except Exception as e:
        print("Failed to import data from load_data_TARDIS.py:", e)
        raise SystemExit(1)

    # Run the pre-analysis
    analysis = MicrotubulePreAnalysis(segments, points, pole1, pole2)
    results = analysis.pre_analysis()

    print("Pre-analysis finished successfully.")
    print("Number of fibers processed:", len(results))
    print("Fiber names:", list(results.keys()))
    print("microtubule_pre_analysis.py done.")

