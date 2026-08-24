"""
Consolidate Master 8-Way Benchmark Table from Exact CSV Artifacts
=================================================================
Reads data/lgcf_results/lgcf_summary.csv, data/solution_v2_results/solution_v2_summary.csv,
and builds the canonical master_ablation_8way.csv with exact per-horizon numbers.
"""

import pandas as pd
import numpy as np
from pathlib import Path

ROOT = Path(r"e:\politics and economy")
DATA_DIR = ROOT / "data"

lgcf_summary = pd.read_csv(DATA_DIR / "lgcf_results" / "lgcf_summary.csv")
solution_summary = pd.read_csv(DATA_DIR / "solution_v2_results" / "solution_v2_summary.csv")

# Extract per-horizon MAE for each configuration
records = []

# Config A: Eco Only
eco_h1 = lgcf_summary[(lgcf_summary["horizon"] == 1) & (lgcf_summary["config"] == "A_EcoOnly")]["mae_mean"].values[0]
eco_h3 = lgcf_summary[(lgcf_summary["horizon"] == 3) & (lgcf_summary["config"] == "A_EcoOnly")]["mae_mean"].values[0]
eco_h5 = lgcf_summary[(lgcf_summary["horizon"] == 5) & (lgcf_summary["config"] == "A_EcoOnly")]["mae_mean"].values[0]

configs_map = {
    "A_EcoOnly": "A. Economy-Only Baseline (Ridge + LGBM)",
    "B_Uniform": "B. Uniform Cross-Domain Mixture",
    "C_RandomGate": "C. Random Dirichlet Gating",
    "D_HeuristicGate": "D. Heuristic Rule-Based Gating",
    "E_LLMGate": "E. Zero-Shot LLM Gate (DeepSeek-V4)",
    "F_OracleGate": "H. Oracle Dynamic Gating (Upper Bound)",
}

for cfg, label in configs_map.items():
    if cfg == "F_OracleGate":
        continue
    h1 = lgcf_summary[(lgcf_summary["horizon"] == 1) & (lgcf_summary["config"] == cfg)]["mae_mean"].values[0]
    h3 = lgcf_summary[(lgcf_summary["horizon"] == 3) & (lgcf_summary["config"] == cfg)]["mae_mean"].values[0]
    h5 = lgcf_summary[(lgcf_summary["horizon"] == 5) & (lgcf_summary["config"] == cfg)]["mae_mean"].values[0]
    
    lift_h1 = (eco_h1 - h1) / eco_h1 * 100.0
    lift_h3 = (eco_h3 - h3) / eco_h3 * 100.0
    lift_h5 = (eco_h5 - h5) / eco_h5 * 100.0
    avg_lift = (lift_h1 + lift_h3 + lift_h5) / 3.0
    
    records.append({
        "Config_ID": cfg[0],
        "Architecture / Strategy": label,
        "h1_MAE": round(float(h1), 5),
        "h3_MAE": round(float(h3), 5),
        "h5_MAE": round(float(h5), 5),
        "Lift_h1_pct": round(float(lift_h1), 2),
        "Lift_h3_pct": round(float(lift_h3), 2),
        "Lift_h5_pct": round(float(lift_h5), 2),
        "Avg_Lift_pct": round(float(avg_lift), 2),
        "Source_Artifact": "data/lgcf_results/lgcf_summary.csv"
    })

# Add G: Solution V2 (Conformal Specialist Router) from solution_v2_summary.csv
sol_h1 = solution_summary[solution_summary["horizon"] == 1]["mae_solution_v2"].values[0]
sol_h3 = solution_summary[solution_summary["horizon"] == 3]["mae_solution_v2"].values[0]
sol_h5 = solution_summary[solution_summary["horizon"] == 5]["mae_solution_v2"].values[0]

eco_sol_h1 = solution_summary[solution_summary["horizon"] == 1]["mae_eco"].values[0]
eco_sol_h3 = solution_summary[solution_summary["horizon"] == 3]["mae_eco"].values[0]
eco_sol_h5 = solution_summary[solution_summary["horizon"] == 5]["mae_eco"].values[0]

sol_lift_h1 = (eco_sol_h1 - sol_h1) / eco_sol_h1 * 100.0
sol_lift_h3 = (eco_sol_h3 - sol_h3) / eco_sol_h3 * 100.0
sol_lift_h5 = (eco_sol_h5 - sol_h5) / eco_sol_h5 * 100.0
sol_avg_lift = (sol_lift_h1 + sol_lift_h3 + sol_lift_h5) / 3.0

records.append({
    "Config_ID": "G",
    "Architecture / Strategy": "G. LGCF-v2 (Conformal Uncertainty Router)",
    "h1_MAE": round(float(sol_h1), 5),
    "h3_MAE": round(float(sol_h3), 5),
    "h5_MAE": round(float(sol_h5), 5),
    "Lift_h1_pct": round(float(sol_lift_h1), 2),
    "Lift_h3_pct": round(float(sol_lift_h3), 2),
    "Lift_h5_pct": round(float(sol_lift_h5), 2),
    "Avg_Lift_pct": round(float(sol_avg_lift), 2),
    "Source_Artifact": "data/solution_v2_results/solution_v2_summary.csv"
})

# Add H: Oracle from lgcf_summary
orc_h1 = lgcf_summary[(lgcf_summary["horizon"] == 1) & (lgcf_summary["config"] == "F_OracleGate")]["mae_mean"].values[0]
orc_h3 = lgcf_summary[(lgcf_summary["horizon"] == 3) & (lgcf_summary["config"] == "F_OracleGate")]["mae_mean"].values[0]
orc_h5 = lgcf_summary[(lgcf_summary["horizon"] == 5) & (lgcf_summary["config"] == "F_OracleGate")]["mae_mean"].values[0]

orc_lift_h1 = (eco_h1 - orc_h1) / eco_h1 * 100.0
orc_lift_h3 = (eco_h3 - orc_h3) / eco_h3 * 100.0
orc_lift_h5 = (eco_h5 - orc_h5) / eco_h5 * 100.0
orc_avg_lift = (orc_lift_h1 + orc_lift_h3 + orc_lift_h5) / 3.0

records.append({
    "Config_ID": "H",
    "Architecture / Strategy": "H. Oracle Dynamic Gating (Upper Bound)",
    "h1_MAE": round(float(orc_h1), 5),
    "h3_MAE": round(float(orc_h3), 5),
    "h5_MAE": round(float(orc_h5), 5),
    "Lift_h1_pct": round(float(orc_lift_h1), 2),
    "Lift_h3_pct": round(float(orc_lift_h3), 2),
    "Lift_h5_pct": round(float(orc_lift_h5), 2),
    "Avg_Lift_pct": round(float(orc_avg_lift), 2),
    "Source_Artifact": "data/lgcf_results/lgcf_summary.csv"
})

out_df = pd.DataFrame(records)
out_csv = DATA_DIR / "benchmarks" / "master_ablation_8way.csv"
out_df.to_csv(out_csv, index=False)
print("Saved Master 8-Way Ablation Benchmark:")
print(out_df.to_string())
