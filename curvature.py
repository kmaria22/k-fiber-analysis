import pandas as pd
import numpy as np

# 1. Data Loader
class DataLoader:
    def __init__(self, amira_path):
        self.amira_path = amira_path

    def load(self):
        # Replace with your actual loading logic from load_data_TARDIS.py
        # For example:
        # data = ImportDataFromAmira(self.amira_path)
        # return data.segments, data.points, data.pole1, data.pole2
        segments = ...  # DataFrame
        points = ...    # DataFrame
        pole1 = ...     # Array or DataFrame
        pole2 = ...     # Array or DataFrame
        return segments, points, pole1, pole2

# 2. Pre-Analysis
from microtubule_pre_analysis import MicrotubulePreAnalysis

def run_pre_analysis(segments, points, pole1, pole2):
    pre = MicrotubulePreAnalysis(segments, points, pole1, pole2)
    pre_analysis_output = pre.pre_analysis()
    print("Pre-analysis complete.")
    return pre_analysis_output

# 3. Main Analysis (Length & Curvature)
class FiberLengthCurvatureAnalysis:
    def __init__(self, pre_analysis_output, pole1, pole2):
        self.pre_analysis_output = pre_analysis_output
        self.pole1 = np.array(pole1)
        self.pole2 = np.array(pole2)

    def analyze(self):
        results = []
        for fiber, kmt_list in self.pre_analysis_output.items():
            for idx, kmt in enumerate(kmt_list):
                coords = kmt['xyz'][['X', 'Y', 'Z']].values.astype(float)
                length = np.sum(np.linalg.norm(np.diff(coords, axis=0), axis=1)) if len(coords) > 1 else 0.0
                end_to_end = np.linalg.norm(coords[0] - coords[-1]) if len(coords) > 1 else 0.0
                total_curv = length / end_to_end if end_to_end > 0 else np.nan
                results.append({
                    'Fiber': fiber,
                    'KMT_Index': idx,
                    'Length': length,
                    'Total_Curvature': total_curv,
                    'Num_Points': len(coords)
                })
        return pd.DataFrame(results)

    def export_to_excel(self, df, filename="fiber_kmt_length_curvature.xlsx"):
        df.to_excel(filename, index=False)
        print(f"Analysis results exported to {filename}")

# 4. Pipeline Orchestrator
class InMemoryMicrotubulePipeline:
    def __init__(self, amira_path):
        self.amira_path = amira_path

    def run(self):
        print("Microtubule Analysis Pipeline Started")
        # Step 1: Load data
        loader = DataLoader(self.amira_path)
        segments, points, pole1, pole2 = loader.load()
        print("✅ Data loaded.")

        # Step 2: Pre-analysis
        pre_analysis_output = run_pre_analysis(segments, points, pole1, pole2)

        # Step 3: Main analysis
        analysis = FiberLengthCurvatureAnalysis(pre_analysis_output, pole1, pole2)
        results_df = analysis.analyze()
        analysis.export_to_excel(results_df)
        print("Pipeline finished. All results are saved.")

# 5. Run the pipeline
if __name__ == "__main__":
    amira_path = "your_amira_file.am"  # Or prompt the user for a file
    pipeline = InMemoryMicrotubulePipeline(amira_path)
    pipeline.run()
