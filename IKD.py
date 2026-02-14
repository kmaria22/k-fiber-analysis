import numpy as np
import pandas as pd

class InterKinetochoreDistanceAnalyzer:
    @staticmethod
    def euclidean_distance(p1, p2):
        return np.linalg.norm(np.array(p1) - np.array(p2))

    @staticmethod
    def median_position(coords):
        coords = np.asarray(coords)
        return np.median(coords, axis=0)

    def calculate_inter_kinetochore_distance(self, kin1_coords, kin2_coords):
        pos1 = self.median_position(kin1_coords)
        pos2 = self.median_position(kin2_coords)
        return self.euclidean_distance(pos1, pos2)

    def count_kmts_per_kinetochore(self, kmt_list):
        return len(kmt_list)

    def compare_sister_kmt_numbers(self, kmt1, kmt2):
        n1 = self.count_kmts_per_kinetochore(kmt1)
        n2 = self.count_kmts_per_kinetochore(kmt2)
        delta = abs(n1 - n2)
        ratio = n1 / n2 if n2 > 0 else np.nan
        return {
            "pole1_kmts": n1,
            "pole2_kmts": n2,
            "total_kmts": n1 + n2,
            "kmt_delta": delta,
            "kmt_ratio": ratio
        }

    def analyze_kinetochore_pair(self, pair_id, kin1_coords, kin2_coords, kmt1, kmt2):
        distance = self.calculate_inter_kinetochore_distance(kin1_coords, kin2_coords)
        kmt_stats = self.compare_sister_kmt_numbers(kmt1, kmt2)
        kin1_median = self.median_position(kin1_coords)
        kin2_median = self.median_position(kin2_coords)
        result = {
            "pair_id": pair_id,
            "inter_kinetochore_distance": distance,
            "pole1_median_x": kin1_median[0],
            "pole1_median_y": kin1_median[1],
            "pole1_median_z": kin1_median[2],
            "pole2_median_x": kin2_median[0],
            "pole2_median_y": kin2_median[1],
            "pole2_median_z": kin2_median[2],
        }
        result.update(kmt_stats)
        return result

    def batch_analyze_kinetochore_pairs(self, pairs_data):
        results = []
        for pair in pairs_data:
            res = self.analyze_kinetochore_pair(
                pair_id=pair['pair_id'],
                kin1_coords=pair['kin1_coords'],
                kin2_coords=pair['kin2_coords'],
                kmt1=pair['kmt1'],
                kmt2=pair['kmt2']
            )
            results.append(res)
        return pd.DataFrame(results)

    @staticmethod
    def summarize_statistics(results_df):
        stats = {
            "distance_mean": results_df["inter_kinetochore_distance"].mean(),
            "distance_median": results_df["inter_kinetochore_distance"].median(),
            "distance_std": results_df["inter_kinetochore_distance"].std(),
            "distance_min": results_df["inter_kinetochore_distance"].min(),
            "distance_max": results_df["inter_kinetochore_distance"].max(),
            "kmt_total_mean": results_df["total_kmts"].mean(),
            "kmt_delta_mean": results_df["kmt_delta"].mean(),
            "kmt_delta_max": results_df["kmt_delta"].max(),
            "kmt_delta_min": results_df["kmt_delta"].min(),
        }
        return stats

    def export_to_excel(self, results_df, stats, filename="InterKinetochore_Results.xlsx"):
        with pd.ExcelWriter(filename, engine='openpyxl') as writer:
            results_df.to_excel(writer, sheet_name="Pairwise_Results", index=False)
            pd.DataFrame([stats]).to_excel(writer, sheet_name="Summary_Statistics", index=False)
        print(f"Results exported to {filename}")

# Example usage
if __name__ == "__main__":
    np.random.seed(42)
    pairs_data = []
    for i in range(1, 6):
        kin1_coords = np.random.normal(loc=0, scale=2, size=(10, 3))
        kin2_coords = np.random.normal(loc=2, scale=2, size=(10, 3))
        kmt1 = [f"KMT_{j}" for j in range(np.random.randint(5, 15))]
        kmt2 = [f"KMT_{j}" for j in range(np.random.randint(5, 15))]
        pairs_data.append({
            "pair_id": f"kinetochore_pair_{i:02d}",
            "kin1_coords": kin1_coords,
            "kin2_coords": kin2_coords,
            "kmt1": kmt1,
            "kmt2": kmt2
        })
    analyzer = InterKinetochoreDistanceAnalyzer()
    results_df = analyzer.batch_analyze_kinetochore_pairs(pairs_data)
    stats = analyzer.summarize_statistics(results_df)
    analyzer.export_to_excel(results_df, stats)
