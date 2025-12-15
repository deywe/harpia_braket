# ───────────────────────────────────────────────────────────────
# File: sphy_harpia_ghz_braket_sha256_multicore_v1_analyzer_en.py
# Purpose: HARPIA SPHY Benchmarking and Visualization Tool
# Author: deywe@QLZ | Modified by Gemini (Nov/2025)
# ───────────────────────────────────────────────────────────────
import warnings
import subprocess
import re

# Ignore specific warnings
warnings.filterwarnings("ignore", category=UserWarning)

import matplotlib
# Configure font for compatibility
matplotlib.rcParams['font.family'] = 'DejaVu Sans'

import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import sys

# 🔧 Configure output directory
ANALYZER_LOG_DIR = "logs_harpia_braket_analyzer"
os.makedirs(ANALYZER_LOG_DIR, exist_ok=True)

def run_analyzer():
    """
    Runs the CSV analyzer, calculates metrics, and generates the plot.
    """
    print("=" * 60)
    print("      🧪 HARPIA SPHY BENCHMARK ANALYZER v1.0")
    print("=" * 60)
    
    # 1. Request the file path
    csv_path = input("📁 Enter the full path of the CSV file for analysis: ")
    
    if not os.path.exists(csv_path):
        print(f"❌ Error: File not found at path: {csv_path}")
        return

    try:
        # 2. Data Reading and Preparation
        df = pd.read_csv(csv_path)
        
        # Ensure essential columns exist
        required_cols = ["SPHY (%)", "Accepted", "Frame"]
        if not all(col in df.columns for col in required_cols):
            print("❌ Error: The CSV file does not contain the required columns: SPHY (%), Accepted, Frame.")
            return

        # Basic cleanup (can be useful if data is incomplete)
        df.dropna(subset=["SPHY (%)"], inplace=True)
        
        # Converts the SPHY column to a numpy array
        coherence_values = df["SPHY (%)"].values
        total_frames = len(df)
        
        # 3. Calculation of Stability Metrics
        mean_stability_index = np.mean(coherence_values) 
        stability_variance_index = np.var(coherence_values)
        
        # Calculation of Acceptance Rate
        valid_count = df["Accepted"].eq("✅").sum()
        acceptance_rate = 100 * (valid_count / total_frames) if total_frames > 0 else 0
        
        # 4. Final Log Printing (Report)
        
        # Attempt to extract the file name for the title
        file_name_base = os.path.basename(csv_path).replace(".csv", "")
        
        print("\n" + "=" * 40)
        print(f"🧾 SPHY ANALYSIS REPORT - {file_name_base}")
        print("=" * 40)
        print(f"🔢 Total Frames Analyzed: {total_frames:,}")
        print(f"✅ Accepted States: {valid_count}/{total_frames} | {acceptance_rate:.2f}%")
        print("\n--- SPHY Stability Metrics ---")
        print(f"📊 Mean Stability Index (MSI): {mean_stability_index:.6f}")
        print(f"📊 Stability Variance Index (SVI): {stability_variance_index:.6f}")
        print("-" * 40)

        # 5. Graph Generation and Saving
        
        # Define the image file name
        img_name = os.path.join(ANALYZER_LOG_DIR, f"analyzed_{file_name_base}_graph.png")
        
        plt.figure(figsize=(12, 5))
        plt.plot(df["Frame"], coherence_values, color="darkcyan", label="⧉ SPHY Coherence")
        
        # Determine color based on acceptance
        acceptance_colors = ['green' if status == "✅" else 'red' for status in df["Accepted"]]
        
        plt.scatter(
            df["Frame"], coherence_values,
            c=acceptance_colors,
            s=8, alpha=0.6
        )
        
        # If the Qubits count is in the filename, use it in the title
        num_qubits = "N/A"
        if "num_qubits" in file_name_base:
            match = re.search(r'_(\d+)q_', file_name_base)
            if match:
                num_qubits = match.group(1)
        
        plt.axhline(90, color='gray', linestyle="dotted", linewidth=1, label="Threshold")
        plt.title(f"📡 HARPIA SPHY Analysis • Qubits: {num_qubits} • Frames: {total_frames:,}\nMSI: {mean_stability_index:.6f} | SVI: {stability_variance_index:.6f}")
        plt.xlabel("Frames")
        plt.ylabel("SPHY Coherence (%)")
        plt.grid(True, linestyle="--", alpha=0.3)
        plt.legend()
        plt.tight_layout()
        
        plt.savefig(img_name, dpi=300)
        print(f"📊 Plot saved as: {img_name}")
        plt.show()

    except Exception as e:
        print(f"❌ An error occurred during analysis: {e}")
        
if __name__ == "__main__":
    # Ensures the pandas library is installed (if not already)
    try:
        import pandas as pd
    except ImportError:
        print("The 'pandas' library is required. Installing...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "pandas"])
        import pandas as pd
        
    # Also ensures 're' is imported for regex match in title
    try:
        import re
    except ImportError:
        # 're' is usually built-in, but checking just in case
        pass
        
    run_analyzer()