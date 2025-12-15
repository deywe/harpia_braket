# -*- coding: utf-8 -*-
# ───────────────────────────────────────────────────────────────
# File        : sphy_braket_qiskit_v2_URBAN.py
# Purpose     : GHZ + HARPIA (Braket Local) + Adaptive Coherence + Meissner IA
#               [FINAL: ALL FIXES, MULTIPROCESSING, AND OPTIMIZED GRAPH]
# Author      : deywe@QLZ | Final Consolidated Code by Gemini
# Last Update : 2025-09-25
# ───────────────────────────────────────────────────────────────

# ─────────────────────────────── IMPORTS ───────────────────────────────

# The 'simbiotic_ai_meissner_core' module must be accessible.
from meissner_core import meissner_correction_step 

from braket.circuits import Circuit
from braket.devices import LocalSimulator
import numpy as np
import matplotlib.pyplot as plt
import csv
from datetime import datetime
import os, random, sys, time, hashlib
from tqdm import tqdm
from multiprocessing import Pool, Manager
from scipy.interpolate import interp1d

LOG_DIR = "logs_harpia_braket_traffic_urban"
os.makedirs(LOG_DIR, exist_ok=True)

# -----------------------------------------------------------------------------
#                   CONFIGURATION AND GLOBAL VARIABLES
# -----------------------------------------------------------------------------

OPTIMAL_CONGESTION_THRESHOLD = 30 

# Global variables to be initialized in each Pool process (PICKLING FIX)
shared_sphy_coherence = None
shared_mode = None

def initializer_pool(sphy_coherence_ref, mode_ref):
    """Initializes shared global variables in the context of each process."""
    global shared_sphy_coherence, shared_mode
    shared_sphy_coherence = sphy_coherence_ref
    shared_mode = mode_ref

def get_user_parameters():
    """Collects simulation mode and number of frames from the user."""
    print("=" * 60)
    print("       📡 HARPIA SPHY MULTICORE SIMULATOR")
    print("=" * 60)
    print("Select Simulation Mode:")
    print("1: GHZ Quantum Stabilization (Original)")
    print("2: Urban Traffic Flow Optimization (New)")
    
    mode = input("Select a mode (1 or 2): ")
    if mode not in ['1', '2']:
        print("❌ Invalid mode selected.")
        exit(1)

    try:
        if mode == '1':
            num_qubits = int(input("🔢 Number of Qubits in GHZ circuit: "))
            pairs_name = "GHZ states"
        else:
            num_qubits = 0 
            pairs_name = "Traffic Frames"

        total_pairs = int(input(f"🔁 Total {pairs_name} to simulate: "))
        
        return mode, num_qubits, total_pairs
    except ValueError:
        print("❌ Invalid numerical input. Please enter integers.")
        exit(1)

# -----------------------------------------------------------------------------
#                   SUPPORT AND SIMULATION FUNCTIONS
# -----------------------------------------------------------------------------

def apply_manual_noise(circuit, qubit, prob):
    if random.random() < prob:
        circuit.x(qubit)

def generate_ghz_state_braket(num_qubits, noise_prob=1.00): 
    """Generates the GHZ circuit with guaranteed bit-flip (noise_prob=1.00 by default)."""
    circuit = Circuit()
    circuit.h(0)
    for i in range(1, num_qubits):
        circuit.cnot(0, i)
    if num_qubits > 1:
        qubit_to_noise = random.randint(1, num_qubits - 1)
        apply_manual_noise(circuit, qubit_to_noise, noise_prob)
    for i in range(num_qubits):
        circuit.measure(i)
    return circuit

def measure_ghz(circuit):
    """Executes the measurement on a local Braket simulator."""
    device = LocalSimulator()
    result_braket = device.run(circuit, shots=1).result()
    result = result_braket.measurements[0].tolist()
    return ''.join(map(str, result))

def simulate_ghz_frame(frame_data, sphy_coherence_ref):
    """Simulates a single GHZ stabilization frame."""
    frame, num_qubits, total_frames, noise_prob = frame_data
    sphy_coherence = sphy_coherence_ref.value 
    
    random.seed(os.getpid() * frame)
    ideal_states = ['0' * num_qubits, '1' * num_qubits]
    current_timestamp = datetime.utcnow().isoformat()
    circuit = generate_ghz_state_braket(num_qubits, noise_prob)
    result = measure_ghz(circuit)

    H = random.uniform(0.95, 1.0)
    S = random.uniform(0.95, 1.0)
    C = sphy_coherence / 100
    I = abs(H - S)
    T = frame
    psi_state = [3.0, 3.0, 1.2, 1.2, 0.6, 0.5]

    try:
        boost, _, _ = meissner_correction_step(H, S, C, I, T, psi_state)
    except Exception as e:
        return None, f"\nCritical error in frame {frame} (Meissner IA): {e}"

    delta = boost * 0.7
    new_coherence = min(100, sphy_coherence + delta)
    sphy_coherence_ref.value = new_coherence

    activated = delta > 0
    accepted = (result in ideal_states) and activated
    
    data_to_hash = f"{frame}:{result}:{H:.4f}:{S:.4f}:{C:.4f}:{I:.4f}:{boost:.4f}:{new_coherence:.4f}:{current_timestamp}"
    sha256_signature = hashlib.sha256(data_to_hash.encode('utf-8')).hexdigest()

    log_entry = [
        frame, result, round(H, 4), round(S, 4),
        round(C, 4), round(I, 4), round(boost, 4),
        round(new_coherence, 4), "✅" if accepted else "❌",
        sha256_signature, current_timestamp
    ]
    return log_entry, None

def simulate_traffic_frame(frame_data, sphy_coherence_ref):
    """Simulates a single Urban Flow Optimization frame."""
    frame, _, total_frames, noise_prob = frame_data
    sphy_flow_efficiency = sphy_coherence_ref.value
    
    random.seed(os.getpid() * frame)
    current_timestamp = datetime.utcnow().isoformat()
    
    # 1. Urban Chaos Simulation
    H = random.uniform(0.7, 1.0) # Heavy Volume
    S = random.uniform(0.7, 1.0) # Structural Stress

    raw_congestion = (H + S) * 50 
    final_congestion_index = max(0.0, raw_congestion * (1 - sphy_flow_efficiency / 100) * 0.7)
    
    # 2. Symbiotic HARPIA Correction
    C = sphy_flow_efficiency / 100 
    I = abs(H - S) 
    T = frame
    psi_state = [3.0, 3.0, 1.2, 1.2, 0.6, 0.5] 

    try:
        boost, _, _ = meissner_correction_step(H, S, C, I, T, psi_state)
    except Exception as e:
        return None, f"\nCritical error in frame {frame} (Meissner IA): {e}"

    delta = boost * 1.5 
    new_flow_efficiency = min(100, sphy_flow_efficiency + delta)
    sphy_coherence_ref.value = new_flow_efficiency
    
    # 3. Result and Validation
    result = round(final_congestion_index, 4) 
    activated = delta > 0
    accepted = (final_congestion_index < OPTIMAL_CONGESTION_THRESHOLD) and activated

    # 🔐 Generate UID_SHA256
    data_to_hash = f"{frame}:{result}:{H:.4f}:{S:.4f}:{C:.4f}:{I:.4f}:{boost:.4f}:{new_flow_efficiency:.4f}:{current_timestamp}"
    sha256_signature = hashlib.sha256(data_to_hash.encode('utf-8')).hexdigest()

    log_entry = [
        frame, result, round(H, 4), round(S, 4),
        round(C, 4), round(I, 4), round(boost, 4),
        round(new_flow_efficiency, 4), "✅" if accepted else "❌",
        sha256_signature, current_timestamp
    ]
    return log_entry, None

# Adds functions to the global dictionary for easy access
SIMULATION_FUNCS = {
    '1': simulate_ghz_frame,
    '2': simulate_traffic_frame
}

def sim_wrapper_pool(frame_data):
    """Serializable wrapper for the Pool that uses the injected global variables."""
    global shared_sphy_coherence, shared_mode 
    
    sim_func = SIMULATION_FUNCS[shared_mode]
    
    return sim_func(frame_data, shared_sphy_coherence)


# -----------------------------------------------------------------------------
#                         MAIN SIMULATION (Final)
# -----------------------------------------------------------------------------
def execute_simulation_multiprocessing_braket(mode, num_qubits, total_frames=100000, noise_prob=0.3, num_processes=4):
    
    manager = Manager()
    log_data, sphy_evolution = manager.list(), manager.list()
    valid_states = manager.Value('i', 0)

    if mode == '1':
        coherence_name = "SPHY Coherence"
        coherence_start = 90.0
        title_prefix = "📡 HARPIA QGHZ STABILIZER + Meissner"
        y_label = "SPHY Coherence (%)"
        csv_prefix = "qghz"
    else: # mode == '2'
        coherence_name = "Flow Efficiency"
        coherence_start = 50.0 
        title_prefix = "🏙️ HARPIA URBAN FLOW OPTIMIZER + Meissner"
        y_label = "Flow Efficiency (%)"
        csv_prefix = "urbanflow"
        
    sphy_coherence = manager.Value('f', coherence_start)
    
    print("=" * 60)
    print(f" {title_prefix} • {total_frames:,} Frames")
    print("=" * 60)

    timecode = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_filename = os.path.join(LOG_DIR, f"{csv_prefix}_{num_qubits}q_log_{timecode}.csv" if mode=='1' else f"{csv_prefix}_log_{timecode}.csv")
    fig_filename = os.path.join(LOG_DIR, f"{csv_prefix}_{num_qubits}q_graph_{timecode}.png" if mode=='1' else f"{csv_prefix}_graph_{timecode}.png")

    frame_inputs = [(f, num_qubits, total_frames, noise_prob) for f in range(1, total_frames + 1)]

    print(f"🔄 Using {num_processes} processes for simulation...")
    
    # Initializes the Pool with shared references (PICKLING FIX)
    with Pool(processes=num_processes, initializer=initializer_pool, initargs=(sphy_coherence, mode)) as pool:
        for log_entry, error in tqdm(pool.imap_unordered(sim_wrapper_pool, frame_inputs),
                                     total=total_frames, desc="⏳ Simulating Frames"):
            if error:
                print(error, file=sys.stderr)
                pool.terminate()
                return
            if log_entry:
                log_data.append(log_entry)
                sphy_evolution.append(log_entry[7])
                if log_entry[-3] == "✅":
                    valid_states.value += 1

    acceptance_rate = 100 * (valid_states.value / total_frames) if total_frames > 0 else 0
    
    log_data_list = list(log_data)
    
    # -------------------------------------------------------------------------
    #                     METRICS CALCULATION AND PRINTING
    # -------------------------------------------------------------------------
    
    print("\n" + "=" * 30)
    print("       📊 METRICS REPORT")
    print("=" * 30)
    print(f"✅ Accepted Frames: {valid_states.value}/{total_frames} | {acceptance_rate:.2f}%")

    if mode == '2' and log_data_list:
        ci_values = np.array([log_entry[1] for log_entry in log_data_list])
        efficiency_values = np.array([log_entry[7] for log_entry in log_data_list])

        print("\n--- Congestion Index (CI) Metric ---")
        print(f"CI Mean: {np.mean(ci_values):.4f}")
        print(f"CI Standard Deviation: {np.std(ci_values):.4f}")
        print(f"CI Minimum/Maximum: {np.min(ci_values):.4f} / {np.max(ci_values):.4f}")
        print(f"Optimal CI Threshold: {OPTIMAL_CONGESTION_THRESHOLD}")

        print("\n--- Efficiency (SPHY) Metric ---")
        print(f"Mean Efficiency: {np.mean(efficiency_values):.4f}%")
        print(f"Final Efficiency: {sphy_coherence.value:.4f}%")
        print(f"Efficiency Standard Deviation: {np.std(efficiency_values):.4f}")
    
    # -------------------------------------------------------------------------
    
    # 💾 Export CSV with SHA
    with open(csv_filename, mode="w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(["Frame", "Result (GHZ/CI)", "H", "S", "C", "I", "Boost", coherence_name, "Accepted", "SHA256_Signature", "Timestamp"])
        writer.writerows(log_data_list)
    print(f"\n🧾 CSV saved: {csv_filename}")

    sphy_evolution = np.array(list(sphy_evolution))
    if sphy_evolution.size == 0:
        print("❌ No data to plot.")
        return

    # -------------------------------------------------------------------------
    #             PLOTTING FOR RESEARCHERS: NOISE OSCILLATION (OPTIMIZED)
    # -------------------------------------------------------------------------

    # MATPLOTLIB OPTIMIZATION CONFIGURATION FOR LARGE DATASETS
    plt.rcParams['agg.path.chunksize'] = 10000 
    plt.rcParams['path.simplify'] = True
    plt.rcParams['path.simplify_threshold'] = 0.5 

    # Downsampling Factor: Reduces to 1 in every N points (Adjust for fast running)
    downsample_factor = 1
    if total_frames > 20000:
        downsample_factor = 100 # Reduces 1.2M frames to 12k frames
    
    frames = np.array(range(1, total_frames + 1))
    frames_plot = frames[::downsample_factor]
    
    
    if mode == '2':
        # Collects wave variables (H, S, I) and applies downsampling
        h_values = np.array([log_entry[2] for log_entry in log_data_list])[::downsample_factor]
        s_values = np.array([log_entry[3] for log_entry in log_data_list])[::downsample_factor]
        i_values = np.array([log_entry[5] for log_entry in log_data_list])[::downsample_factor] 
        sphy_plot = sphy_evolution[::downsample_factor] # Applies downsampling to SPHY evolution

        fig, ax1 = plt.subplots(figsize=(14, 7))

        # Axis 1: Oscillations (H, S, I)
        ax1.set_xlabel(f"Frames (Sampling: 1 in every {downsample_factor})")
        ax1.set_ylabel("Noise Factors (0.0 to 1.0)")
        ax1.plot(frames_plot, h_values, color='gold', linewidth=1, label="H (Traffic Volume)")
        ax1.plot(frames_plot, s_values, color='red', linewidth=1, alpha=0.6, label="S (Structural Stress)")
        ax1.plot(frames_plot, i_values * 10, color='purple', linestyle='--', linewidth=1, label="I (Imbalance x10)")
        ax1.tick_params(axis='y')
        ax1.set_ylim(0.0, 1.1)

        # Axis 2: SPHY Coherence (The Control Line)
        ax2 = ax1.twinx()  
        ax2.set_ylabel(y_label, color='darkcyan') 
        ax2.plot(frames_plot, sphy_plot, color='darkcyan', linewidth=3, label="Flow Efficiency (SPHY)")
        ax2.tick_params(axis='y', labelcolor='darkcyan')
        ax2.set_ylim(50, 105)
        
        # Minimum Coherence Line
        mean_sphy = np.mean(sphy_evolution)
        ax2.axhline(mean_sphy, color='darkcyan', linestyle=':', alpha=0.7, label=f"SPHY Mean ({mean_sphy:.2f}%)")


        # Title and Legends
        fig.suptitle(f"Wave Analysis: Adaptive Coherence x Imbalance ({total_frames:,} Frames)", fontsize=16)
        
        # Combines legends from both axes
        lines, labels = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax2.legend(lines + lines2, labels + labels2, loc='lower right')
        ax1.grid(True, linestyle="--", alpha=0.3)
        plt.tight_layout()
        
    elif mode == '1':
        # Original convergence graph for GHZ.
        plt.figure(figsize=(12, 5))
        plt.plot(range(1, total_frames + 1), sphy_evolution, color="blue", label=f"⧉ SPHY {coherence_name}")
        plt.axhline(90, color='gray', linestyle="dotted", linewidth=1, label="Initial Coherence (90%)")
        plt.title(f"{title_prefix} • {total_frames:,} Frames")
        plt.xlabel("Frames")
        plt.ylabel(y_label)
        plt.grid(True, linestyle="--", alpha=0.3)
        plt.legend()
        plt.tight_layout()


    plt.savefig(fig_filename, dpi=300)
    print(f"📊 Graph saved as: {fig_filename}")
    plt.show()


if __name__ == "__main__":
    mode, qubits, pairs = get_user_parameters()
    execute_simulation_multiprocessing_braket(mode=mode, num_qubits=qubits, total_frames=pairs, noise_prob=0.3, num_processes=4)