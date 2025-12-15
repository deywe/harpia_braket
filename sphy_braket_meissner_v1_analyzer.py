# -*- coding: utf-8 -*-
# ───────────────────────────────────────────────────────────────
# File        : sphy_braket_meissner_v1_analyzer.py
# Purpose     : Analysis and Benchmark Generation from CSV Data
# Author      : Gemini AI (Translated by Gemini AI)
# Last Update : 2025-11-26
# ───────────────────────────────────────────────────────────────

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import sys

# Define success criteria and Key Performance Indicators (KPIs)
BENCHMARK_KPIS = {
    'Acceptance_Rate': "Valid GHZ State Acceptance Rate (%)",
    'Mean_SPHY': "Mean SPHY Stability Index (%)",
    'Stability_Variance': "SPHY Stability Variance (Lower is better)",
    'Mean_Boost': "Mean Correction Boost (Meissner AI)",
}

def get_csv_path():
    """
    Prompts the user for the complete path to the CSV file.
    """
    if len(sys.argv) > 1:
        csv_path = sys.argv[1]
    else:
        print("\n" + "=" * 60)
        print(" 📊 SPHY/MEISSNER BENCHMARK ANALYZER ")
        print("=" * 60)
        csv_path = input("➡️ Enter the FULL path to the CSV file for analysis: ")
    
    if not os.path.exists(csv_path):
        print(f"\n❌ Error: File not found at: {csv_path}")
        sys.exit(1)
        
    return csv_path

def load_and_preprocess_data(csv_path):
    """
    Loads the CSV and prepares the data for analysis.
    """
    print(f"\n📂 Loading data from: {csv_path}")
    try:
        # Load data with standard UTF-8 encoding
        df = pd.read_csv(csv_path, encoding='utf-8')
        
        # Rename columns for easier handling (assuming input columns are in Portuguese/mixed)
        df.rename(columns={'SPHY (%)': 'SPHY_Coherence', 'Accepted': 'Accepted_Status'}, inplace=True)
        
        # Convert 'Accepted_Status' column to boolean (True/False)
        # '✅' -> True, '❌' -> False
        df['Accepted_Numeric'] = df['Accepted_Status'].apply(lambda x: 1 if x == '✅' else 0)
        
        print(f"✅ Data loaded. Total Frames: {len(df):,}")
        return df
    except Exception as e:
        print(f"❌ Error processing CSV: {e}")
        sys.exit(1)

def generate_benchmarks(df):
    """
    Calculates the main benchmark KPIs.
    """
    total_frames = len(df)
    
    # 1. Acceptance Rate
    acceptance_rate = (df['Accepted_Numeric'].sum() / total_frames) * 100
    
    # 2. Mean and Variance of SPHY (Stabilized Coherence)
    mean_sphy = df['SPHY_Coherence'].mean()
    stability_variance = df['SPHY_Coherence'].var()
    
    # 3. Mean Correction Boost (Meissner AI Metric)
    mean_boost = df['Boost'].mean()
    
    results = {
        'Acceptance_Rate': acceptance_rate,
        'Mean_SPHY': mean_sphy,
        'Stability_Variance': stability_variance,
        'Mean_Boost': mean_boost,
        'Total_Frames': total_frames
    }
    return results

def plot_benchmarks(df, benchmarks, csv_path):
    """
    Generates benchmark visualization plots with improved layout.
    """
    print("\n📈 Generating Benchmark Plots...")
    
    # --- Figure and Subplots Configuration (2x2 layout for better distribution) ---
    fig, axes = plt.subplots(2, 2, figsize=(16, 12)) 
    
    # Adjust spacing between subplots to avoid overlap
    plt.subplots_adjust(hspace=0.4, wspace=0.3) 
    
    # Main figure title, adjusted to avoid overlap
    fig.suptitle(f"GHZ + Meissner Correction Benchmark Analysis • Frames: {benchmarks['Total_Frames']:,}", fontsize=18, y=1.02)
    
    # --- Plot 1: SPHY Coherence Evolution (%) ---
    axes[0, 0].plot(df['Frame'], df['SPHY_Coherence'], label='SPHY Coherence (%)', color='#0077b6', alpha=0.7)
    axes[0, 0].axhline(benchmarks['Mean_SPHY'], color='#00b4d8', linestyle='--', label=f"Mean SPHY: {benchmarks['Mean_SPHY']:.2f}%")
    axes[0, 0].set_title('Evolution of SPHY Stability Index', fontsize=12)
    axes[0, 0].set_xlabel('Simulation Frame')
    axes[0, 0].set_ylabel('SPHY Coherence (%)')
    axes[0, 0].legend()
    axes[0, 0].grid(True, linestyle=':', alpha=0.6)
    
    # --- Plot 2: Correction Boost (Meissner AI) ---
    axes[0, 1].plot(df['Frame'], df['Boost'], label='Correction Boost', color='#2a9d8f', alpha=0.6)
    axes[0, 1].axhline(benchmarks['Mean_Boost'], color='#e76f51', linestyle='--', label=f"Mean Boost: {benchmarks['Mean_Boost']:.4f}")
    axes[0, 1].set_title('Meissner AI Correction Boost Intensity', fontsize=12)
    axes[0, 1].set_xlabel('Simulation Frame')
    axes[0, 1].set_ylabel('Boost (Coherence Delta)')
    axes[0, 1].legend()
    axes[0, 1].grid(True, linestyle=':', alpha=0.6)
    
    # --- Plot 3: Acceptance Distribution (Pie Chart) ---
    acceptance_data = df.groupby('Accepted_Status').size()
    total = acceptance_data.sum()
    labels = [f'{k} ({v/total*100:.2f}%)' for k, v in acceptance_data.items()]
    
    # Plotting pie chart in axes[1,0]
    axes[1, 0].pie(acceptance_data, labels=labels, autopct='', startangle=90, colors=['#ef233c', '#4a90e2'])
    axes[1, 0].set_title(f"Valid GHZ State Acceptance Distribution\nRate: {benchmarks['Acceptance_Rate']:.2f}%", fontsize=12)
    axes[1, 0].axis('equal') 
    
    # --- Plot 4: SPHY Coherence Distribution (Histogram) ---
    # Shows the frequency distribution of stability levels
    axes[1, 1].hist(df['SPHY_Coherence'], bins=20, color='#f4a261', edgecolor='black', alpha=0.7)
    axes[1, 1].axvline(benchmarks['Mean_SPHY'], color='#e76f51', linestyle='--', label=f"Mean SPHY: {benchmarks['Mean_SPHY']:.2f}%")
    axes[1, 1].set_title('SPHY Coherence Distribution', fontsize=12)
    axes[1, 1].set_xlabel('SPHY Coherence (%)')
    axes[1, 1].set_ylabel('Frequency')
    axes[1, 1].legend()
    axes[1, 1].grid(True, linestyle=':', alpha=0.6)
    
    plt.tight_layout(rect=[0, 0.03, 1, 0.95]) # Adjusts layout to prevent overlap
    
    # Save and show the plot
    output_dir = os.path.dirname(csv_path) or '.' 
    filename_base = os.path.basename(csv_path).replace('.csv', '')
    fig_filename = os.path.join(output_dir, f"{filename_base}_BENCHMARK.png")
    plt.savefig(fig_filename, dpi=300)
    
    print(f"\n🖼️ Benchmark Plot saved as: {fig_filename}")
    plt.show()

def display_summary(benchmarks):
    """
    Displays the final benchmark results summary.
    """
    print("\n" + "#" * 70)
    print(" 🏆 FINAL SPHY/MEISSNER QUANTUM CORRECTION BENCHMARK RESULTS ")
    print("#" * 70)
    
    print(f"Total Number of Frames Analyzed: {benchmarks['Total_Frames']:,}")
    print("-" * 35)

    print(f"1. {BENCHMARK_KPIS['Acceptance_Rate']}:")
    print(f"   >>> {benchmarks['Acceptance_Rate']:.2f}% (Is the stabilization goal being met?)")
    
    print("-" * 35)

    print(f"2. {BENCHMARK_KPIS['Mean_SPHY']}:")
    print(f"   >>> {benchmarks['Mean_SPHY']:.4f}% (Mean Coherence Maintained)")
    
    print("-" * 35)

    print(f"3. {BENCHMARK_KPIS['Stability_Variance']}:")
    print(f"   >>> {benchmarks['Stability_Variance']:.6f} (Variance. The closer to zero, the more stable the system is)")
    
    print("-" * 35)

    print(f"4. {BENCHMARK_KPIS['Mean_Boost']}:")
    print(f"   >>> {benchmarks['Mean_Boost']:.4f} (On average, how much coherence the AI needed to 'inject' per frame)")
    
    print("#" * 70)

if __name__ == "__main__":
    csv_file = get_csv_path()
    
    # 1. Load and Pre-process
    data_frame = load_and_preprocess_data(csv_file)
    
    if data_frame.empty:
        print("Insufficient data to generate benchmarks.")
    else:
        # 2. Generate KPIs
        kpis = generate_benchmarks(data_frame)
        
        # 3. Display Summary and Generate Plots
        display_summary(kpis)
        plot_benchmarks(data_frame, kpis, csv_file)