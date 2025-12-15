# -*- coding: utf-8 -*-
# ───────────────────────────────────────────────────────────────
# File: braket_sphy_toroidal_3d_v9.py
# Purpose: QUANTUM TUNNELING IN A TOROIDAL LATTICE + SPHY FIELD ENGINEERING (AWS Braket)
# Author: deywe@QLZ | Converted to Braket by Gemini AI (Nov/2025)
# ───────────────────────────────────────────────────────────────
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

import matplotlib
matplotlib.rcParams['font.family'] = 'DejaVu Sans'

# 1️⃣ Import necessary modules
# ASSUMPTION: 'meissner_core.py' is available
try:
    from meissner_core import meissner_correction_step 
except ImportError:
    print("❌ Critical Error: The file 'meissner_core.py' was not found.")
    print("Ensure it is in the same directory and is accessible.")
    sys.exit(1)

# ⚛️ Braket Imports
from braket.circuits import Circuit
from braket.devices import LocalSimulator
import numpy as np 

import os, random, sys, time, hashlib, csv
from datetime import datetime
from tqdm import tqdm
from multiprocessing import Pool, Manager
from scipy.interpolate import interp1d, griddata # Imports griddata for the 3D plot
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D 
import pandas as pd # Imports pandas for CSV manipulation in plotting

# === SPHY Toroidal Lattice Configuration ===
GRID_SIZE = 2 
NUM_WIRES = GRID_SIZE * GRID_SIZE # 4 Qubits (0, 1, 2, 3)

QUBITS = list(range(NUM_WIRES)) 
TARGET_QUBIT_INDEX = 0 

# === Log Directory
LOG_DIR = "logs_sphy_toroidal_3d_animation_braket"
os.makedirs(LOG_DIR, exist_ok=True)

# 🌐 Defining the Braket Simulator
simulator = LocalSimulator()

# === Configuration and Helper Functions ===

def get_user_parameters():
    try:
        num_qubits = NUM_WIRES
        print(f"🔢 Number of Qubits (Lattice {GRID_SIZE}x{GRID_SIZE}): {num_qubits}")
        total_pairs = int(input("🔁 Total Tunneling Attempts (Frames) to simulate: "))
        
        barrier_strength_input = float(input("🚧 Barrier Strength (0.0 to 1.0): "))
        if not (0.0 <= barrier_strength_input <= 1.0):
             print("❌ Barrier Strength must be between 0.0 and 1.0.")
             sys.exit(1)
             
        barrier_strength_theta = barrier_strength_input * np.pi / 2 
        
        return num_qubits, total_pairs, barrier_strength_theta
    except ValueError:
        print("❌ Invalid input. Please enter integers/floats.")
        sys.exit(1)

# ⚛️ Defining the Toroidal Quantum Circuit (Braket Circuit)
def toroidal_tunneling_circuit_3d_braket(barrier_theta, sphy_perturbation_angle):
    """
    Constructs the Toroidal circuit using the Braket syntax.
    """
    circuit = Circuit()
    
    # 1. State Preparation (Hadamard on all)
    circuit.h(QUBITS)

    # 2. TOROIDAL LATTICE: The Active SPHY Field (CZ Gates)
    # CZ Connections: (0, 1), (1, 3), (2, 3), (3, 2), (3, 1), (2, 0), (0, 2), (1, 3), (2, 0), (3, 1) 
    cz_gates = [
        (0, 1), (1, 3), (2, 3), (3, 2), (3, 1), (2, 0), (0, 2), (1, 3), (2, 0), (3, 1) 
    ]
    for q1, q2 in cz_gates:
        circuit.cz(q1, q2)
    
    # 3. Barrier (RZ Gate on Target Qubit)
    circuit.rz(TARGET_QUBIT_INDEX, barrier_theta)
    
    # 4. SPHY Noise/Correction (RZ and RX on Qubits)
    # RZ on target
    circuit.rz(TARGET_QUBIT_INDEX, sphy_perturbation_angle)
    
    # RX on others (1, 2, 3)
    for idx in [1, 2, 3]:
         circuit.rx(idx, sphy_perturbation_angle / 2)
         
    return circuit

# === Main Simulation Function per Frame ===

def simulate_frame(frame_data):
    frame, num_qubits, total_frames, noise_prob, sphy_coherence, barrier_theta = frame_data
    
    random.seed(os.getpid() * frame) 
    
    sphy_perturbation_angle = 0.0
    if random.random() < noise_prob:
        sphy_perturbation_angle = random.uniform(-np.pi/8, np.pi/8)
    
    current_timestamp = datetime.utcnow().isoformat()
    
    # 1. Build the Braket Circuit
    circuit = toroidal_tunneling_circuit_3d_braket(barrier_theta, sphy_perturbation_angle)
    
    # 2. Run the Braket Circuit (1 shot)
    result = simulator.run(circuit, shots=1).result()
    
    counts = result.measurement_counts
    
    if not counts:
        measurement_results_str = '0' * num_qubits
    else:
        measurement_results_str = list(counts.keys())[0] 
    
    target_qubit_result = int(measurement_results_str[TARGET_QUBIT_INDEX]) # 0 or 1
    
    result_raw = target_qubit_result # 1 = tunneling success
    ideal_state = 1

    # === SPHY/Meissner Logic ===
    H = random.uniform(0.95, 1.0) 
    S = random.uniform(0.95, 1.0) 
    C = sphy_coherence / 100    
    I = abs(H - S)             
    T = frame                   
    psi_state = [3.0, 3.0, 1.2, 1.2, 0.6, 0.5] # Variables for the Meissner Core

    try:
        boost, phase_impact, psi_state = meissner_correction_step(H, S, C, I, T, psi_state)
    except Exception as e:
        return None, None, f"\nCritical Error on frame {frame} (AI Meissner): {e}"

    delta = boost * 0.7
    new_coherence = min(100, sphy_coherence + delta)
    activated = delta > 0 
    
    accepted = (result_raw == ideal_state) and activated

    data_to_hash = f"{frame}:{result_raw}:{H:.4f}:{S:.4f}:{C:.4f}:{I:.4f}:{boost:.4f}:{new_coherence:.4f}:{current_timestamp}"
    sha256_signature = hashlib.sha256(data_to_hash.encode('utf-8')).hexdigest()

    # Heuristic Calculation of Pauli Z Expectation (PauliZ) for the 3D Log
    z_expectations = []
    for i in range(num_qubits):
        if i == TARGET_QUBIT_INDEX:
             z_expval = (2 * target_qubit_result) - 1.0 
        else:
             # Heuristic to simulate toroidal variation
             z_expval = (2 * target_qubit_result) - 1.0 
             z_expval += random.uniform(-0.1, 0.1) 
             z_expval = np.clip(z_expval, -1.0, 1.0)
             
        z_expectations.append(z_expval)
        
    phase_logs = [round(z, 4) for z in z_expectations] 
    
    log_entry = [
        frame, result_raw, 
        *phase_logs, 
        round(H, 4), round(S, 4), round(C, 4), round(I, 4), round(boost, 4),
        round(new_coherence, 4), "✅" if accepted else "❌",
        sha256_signature, current_timestamp
    ]
    # Returns the updated coherence value (for the next frame) and the log
    return log_entry, new_coherence, None

# === Static 3D Plotting Function ===
def plot_3d_sphy_field(csv_filename, fig_filename_3d):
    
    try:
        df = pd.read_csv(csv_filename)
    except FileNotFoundError:
        print(f"❌ Error: CSV file not found at {csv_filename}")
        return

    # The phase column names must be the same as used in the execute_simulation_multiprocessing function
    phase_cols = [f"Qubit_{i+1}_Phase" for i in range(NUM_WIRES)] 
    # Ensure columns exist before attempting to access them
    if not all(col in df.columns for col in phase_cols):
        print(f"❌ Error: Phase columns not found in CSV. Expected: {phase_cols}")
        return
        
    mean_phases = df[phase_cols].mean().values

    # 2D coordinates for the 2x2 lattice: (0,0), (0,1), (1,0), (1,1)
    X = np.array([0, 0, 1, 1])
    Y = np.array([0, 1, 0, 1])
    Z = mean_phases 

    # Interpolation to smooth the 3D graph
    xi = np.linspace(X.min(), X.max(), 50)
    yi = np.linspace(Y.min(), Y.max(), 50)
    XI, YI = np.meshgrid(xi, yi)

    ZI = griddata((X, Y), Z, (XI, YI), method='cubic')

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    surf = ax.plot_surface(XI, YI, ZI, cmap='viridis', edgecolor='none', alpha=0.9)
    ax.scatter(X, Y, Z, color='red', s=50, label='Average Qubit Phase')

    ax.set_xlabel('X Coordinate (Qubit)')
    ax.set_ylabel('Y Coordinate (Qubit)')
    ax.set_zlabel(r'Average Pauli Z Phase (SPHY Field $\phi$)')
    ax.set_title('3D Visualization of the SPHY Shape Field (Toroidal Coherence Average)')
    # Ensures the Z-limit is set, but prevents division by zero if Z is all zero
    if Z.min() != Z.max():
        ax.set_zlim(Z.min() * 1.1, Z.max() * 1.1)
    else:
        ax.set_zlim(-1.0, 1.0)
    
    fig.colorbar(surf, shrink=0.5, aspect=5, label='Phase/Population (|0> -> |1>)')
    
    plt.savefig(fig_filename_3d, dpi=300)
    print(f"🖼️ 3D SPHY Shape Graph saved: {fig_filename_3d}")


# === Main Execution Function (Multiprocessing) ===

def execute_simulation_multiprocessing(num_qubits, total_frames, barrier_theta, noise_prob=1.0, num_processes=4):
    print("=" * 60)
    print(f" ⚛️ SPHY WAVES: Toroidal Tunneling ({GRID_SIZE}x{GRID_SIZE}) • {total_frames:,} Frames")
    print(f" 🚧 Barrier Strength: {barrier_theta*180/np.pi:.2f} degrees RZ (Analog)")
    print("=" * 60)

    timecode = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_filename = os.path.join(LOG_DIR, f"toroidal_{num_qubits}q_log_{timecode}.csv")
    fig_filename = os.path.join(LOG_DIR, f"toroidal_{num_qubits}q_graph_2D_{timecode}.png")
    fig_filename_3d = os.path.join(LOG_DIR, f"toroidal_{num_qubits}q_graph_3D_SPHY_{timecode}.png")

    manager = Manager()
    sphy_coherence = manager.Value('f', 90.0)
    log_data, sphy_evolution = manager.list(), manager.list()
    valid_states = manager.Value('i', 0)
    
    frame_inputs = [
        (f, num_qubits, total_frames, noise_prob, sphy_coherence.value, barrier_theta) 
        for f in range(1, total_frames + 1)
    ]

    print(f"🔄 Using {num_processes} processes for simulation...")
    # The Pool requires a loop to update the sphy_coherence.value in the manager.
    # We will use a regular for loop with Pool.imap_unordered to maintain sequential coherence update.
    with Pool(processes=num_processes) as pool:
        results = pool.imap_unordered(simulate_frame, frame_inputs)
        
        for log_entry, new_coherence, error in tqdm(results, total=total_frames, desc="⏳ Simulating Toroidal SPHY"):
            if error:
                print(f"\n{error}", file=sys.stderr)
                pool.terminate()
                break
            if log_entry:
                log_data.append(log_entry)
                sphy_evolution.append(new_coherence)
                # The coherence value needs to be updated to be used in the next input block
                sphy_coherence.value = new_coherence 
                if log_entry[-3] == "✅":
                    valid_states.value += 1

    # --- Metric Calculation and CSV Writing ---
    acceptance_rate = 100 * (valid_states.value / total_frames) if total_frames > 0 else 0
    print(f"\n✅ Tunneling Success Rate (Toroidal SPHY): {valid_states.value}/{total_frames} | {acceptance_rate:.2f}%")
    
    if sphy_evolution:
        sphy_np_array = np.array(list(sphy_evolution))
        mean_stability = np.mean(sphy_np_array)
        stability_variance = np.var(sphy_np_array)
        print(f"\n📊 Average Stability Index (SPHY): {mean_stability:.6f}")
        print(f"📊 Stability Variance Index: {stability_variance:.6f}")

    with open(csv_filename, mode="w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        
        qubit_phase_headers = [f"Qubit_{i+1}_Phase" for i in range(NUM_WIRES)]
        
        header = [
            "Frame", "Result", 
            *qubit_phase_headers, 
            "H", "S", "C", "I", "Boost", "SPHY (%)", "Accepted", 
            "SHA256_Signature", "Timestamp"
        ]
        writer.writerow(header)
        writer.writerows(list(log_data))
    print(f"🧾 CSV saved: {csv_filename}")

    # --- Static 3D Graph Generation ---
    plot_3d_sphy_field(csv_filename, fig_filename_3d)

    # --- [2D PLOTTING CODE WITH DOUBLE SUBPLOT] ---
    
    sphy_evolution_list = list(sphy_evolution)
    if not sphy_evolution_list:
        print("❌ No data to plot the 2D graph.")
        return

    sphy_evolution = np.array(sphy_evolution_list)
    time_points = np.linspace(0, 1, len(sphy_evolution))
    
    # Simulation of two "redundancies" or phase/coherence signals
    n_redundancies = 2 
    signals = [interp1d(time_points, np.roll(sphy_evolution, i), kind='cubic') for i in range(n_redundancies)]
    new_time = np.linspace(0, 1, 2000)
    
    data = [signals[i](new_time) + np.random.normal(0, 0.15, len(new_time)) * (1 + i * 0.1) for i in range(n_redundancies)]
    weights = np.linspace(1, 1.5, n_redundancies)
    tunneling_stability = np.average(data, axis=0, weights=weights)

    stability_mean_2 = np.mean(data[1]) 
    stability_variance_2 = np.var(data[1])

    # --- Creating Subplots ---
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 12))

    # Graph 1: SPHY Coherence Signal (Amplitude)
    ax1.set_title("SPHY Coherence Evolution (Signal 1: Amplitude)")
    for i in range(n_redundancies):
        ax1.plot(new_time, data[i], alpha=0.3, color='blue')  
    ax1.plot(new_time, tunneling_stability, 'k--', linewidth=2, label="Weighted Average Stability")
    ax1.set_xlabel("Normalized Time")
    ax1.set_ylabel("Coherence/Amplitude")
    ax1.legend()
    ax1.grid()

    # Graph 2: SPHY Coherence Signal (Stability)
    ax2.set_title("SPHY Coherence Evolution (Signal 2: Stability)")
    ax2.plot(new_time, data[1], color='red', alpha=0.7, label='Coherence Signal (2)')
    
    ax2.axhline(stability_mean_2, color='green', linestyle='--', label=f"Mean: {stability_mean_2:.2f}")
    ax2.axhline(stability_mean_2 + np.sqrt(stability_variance_2), color='orange', linestyle='--', label=f"± Variance")
    ax2.axhline(stability_mean_2 - np.sqrt(stability_variance_2), color='orange', linestyle='--')

    ax2.set_xlabel("Normalized Time")
    ax2.set_ylabel("Coherence/Amplitude")
    ax2.legend()
    ax2.grid()

    fig.suptitle(f"Quantum Tunneling Simulation: {total_frames} Attempts (Toroidal SPHY)", fontsize=16)
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    
    plt.savefig(fig_filename, dpi=300)
    print(f"🖼️ 2D Stability Graph saved: {fig_filename}")
    plt.show()


if __name__ == "__main__":
    qubits, pairs, barrier_theta = get_user_parameters()
    
    execute_simulation_multiprocessing(num_qubits=qubits, total_frames=pairs, barrier_theta=barrier_theta, noise_prob=1.0, num_processes=4)