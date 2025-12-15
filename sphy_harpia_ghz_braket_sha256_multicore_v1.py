# ───────────────────────────────────────────────────────────────
# File: sphy_harpia_ghz_braket_sha256_multicore_v1.py
# Purpose: GHZ Quantum Collapse Simulation with HARPIA (Braket) + Adaptive Coherence
# Author: deywe@QLZ | Modified by Gemini (Nov/2025)
# ───────────────────────────────────────────────────────────────
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

import matplotlib
matplotlib.rcParams['font.family'] = 'DejaVu Sans'

import os
import csv
import sys
import re
import random
import numpy as np
from datetime import datetime
from tqdm import tqdm
import matplotlib.pyplot as plt
import subprocess
import hashlib
from multiprocessing import Pool, Manager
from braket.circuits import Circuit
from braket.devices import LocalSimulator


# 🔧 Configure log directory
LOG_DIR = "logs_harpia_braket"
os.makedirs(LOG_DIR, exist_ok=True)

# 🧠 Input parameters
def input_parameters():
    try:
        num_qubits = int(input("🔢 Number of Qubits in GHZ circuit: "))
        total_pairs = int(input("🔁 Total GHZ states to simulate: "))
        
        # Prompt to enable noise right after parameters
        enable_noise = input("Do you want to enable noise now? (y/n): ").lower()
        initial_noise = True if enable_noise == 'y' else False
        
        return num_qubits, total_pairs, initial_noise
    except:
        print("❌ Invalid input.")
        sys.exit(1)

# 🧬 Generate GHZ state circuit with symbolic noise
def generate_ghz_state(nq, noise_enabled=False, noise_prob=0.3):
    circuit = Circuit()
    circuit.h(0)
    for i in range(1, nq):
        circuit.cnot(0, i)
    if noise_enabled and random.random() < noise_prob:
        # Aplica uma perturbação simbólica (erro X)
        noisy_target = random.choice(range(1, nq))
        circuit.x(noisy_target)
    for i in range(nq):
        circuit.measure(i)
    return circuit

# 🧪 Measure on local simulator
def measure(circuit):
    # Usa shots=1 para simular uma única medição por frame
    device = LocalSimulator()
    result = device.run(circuit, shots=1).result()
    counts = result.measurement_counts
    # Retorna a string do estado medido (ex: '0000')
    return list(counts.keys())[0] 

# ⚙️ External HARPIA STDJ AI Symbiotic Call
# Assume-se que o binário 'sphy_simbiotic_entangle_ai' retorna o F_opt (boost)
def calculate_F_opt(H, S, C, I, T):
    # Nota: O binário externo deve estar compilado e acessível no PATH
    result = subprocess.run(
        ["./sphy_simbiotic_entangle_ai", str(H), str(S), str(C), str(I), str(T)],
        capture_output=True, text=True, timeout=5 # Adicionado timeout para segurança
    )
    match = re.search(r"([-+]?\d*\.\d+|\d+)", result.stdout)
    if match:
        return float(match.group(0))
    else:
        # Se falhar, retorna um boost zero para não quebrar a simulação
        return 0.0 

# 🧪 Simulate one frame
def simulate_frame(args):
    frame, num_qubits, sphy_coherence, noise_enabled, noise_prob, ideal_states, noise_status_changed, sphy_coherence_global = args
    
    # 🔔 O trecho de input de ruído manual foi removido desta função 
    # pois não funciona bem em multiprocessing (usará o valor inicial).

    circuit = generate_ghz_state(num_qubits, noise_enabled, noise_prob)
    result = measure(circuit)

    # Variáveis Simbióticas: H (Harmonia), S (Simetria), C (Coerência Atual)
    H = random.uniform(0.95, 1.0) if noise_enabled else 0.95 
    S = random.uniform(0.95, 1.0) if noise_enabled else 0.95
    C = sphy_coherence / 100 # Conversão para decimal
    I = abs(H - S) # Entropia (Incoerência)
    T = frame # Time/Iteração

    # HARPIA boost via external binary (Rust/C++)
    boost = calculate_F_opt(H, S, C, I, T)
    delta = boost * 0.7 # Fator de ajuste de ganho
    new_coherence = min(100, sphy_coherence + delta) # Satura em 100%
    is_active = delta > 0 # A correção foi ativa
    is_accepted = (result in ideal_states) and is_active # Aceito se ideal E a IA foi ativa

    # Atualiza o valor de coerência global (importante para o próximo frame)
    sphy_coherence_global.value = new_coherence

    # 🔐 Generate UID_SHA256
    log_line = [
        frame, result,
        round(H, 4), round(S, 4),
        round(C, 4), round(I, 4),
        round(boost, 4), round(new_coherence, 4),
        "✅" if is_accepted else "❌"
    ]
    hash_input = ",".join(map(str, log_line))
    uid_sha256 = hashlib.sha256(hash_input.encode()).hexdigest()

    log_line.append(uid_sha256)
    
    # Retorna a linha de log, o novo valor de coerência e o status de aceitação
    return log_line, new_coherence, is_accepted 

# 🚀 Main simulation with multicore
def run_simulation(num_qubits, total=100000, noise_prob=0.3, initial_noise=False):
    print("=" * 60)
    print(f"    🧿 HARPIA QGHZ STABILIZER • {num_qubits} Qubits • {total:,} Frames")
    print("=" * 60)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_name = os.path.join(LOG_DIR, f"qghz_{num_qubits}q_log_{timestamp}.csv")
    img_name = os.path.join(LOG_DIR, f"qghz_{num_qubits}q_graph_{timestamp}.png")

    # Shared variables for multiprocessing
    manager = Manager()
    # Coerência inicial em 90.0%
    sphy_coherence_global = manager.Value('d', 90.0) 
    valid_count = manager.Value('i', 0)
    log_data = manager.list()
    # Lista para rastrear a evolução da coerência em CADA frame
    sphy_evolution_list = manager.list() 
    ideal_states = ['0' * num_qubits, '1' * num_qubits]
    noise_enabled_global = manager.Value('b', initial_noise)
    noise_status_changed = manager.Value('b', False)

    # Prepare arguments for multiprocessing
    # Nota: Passamos o valor de coherence_global para que cada worker possa usá-lo
    args_list = [
        (frame, num_qubits, sphy_coherence_global.value, noise_enabled_global.value, noise_prob, ideal_states, noise_status_changed, sphy_coherence_global)
        for frame in range(1, total + 1)
    ]

    # 🚀 Run simulation with multiprocessing
    with Pool() as pool:
        # Executa a simulação e captura os resultados
        results = list(tqdm(pool.imap(simulate_frame, args_list), total=total, desc="⏳ Simulating GHZ"))

    # Process results: o loop principal atualiza as listas finais
    for log_line, new_coherence, is_accepted in results:
        # A nova coerência já foi atualizada pelo worker, mas a rastreamos aqui
        log_data.append(log_line)
        sphy_evolution_list.append(new_coherence) # Adiciona o valor de coerência para o cálculo final
        if is_accepted:
            valid_count.value += 1

    # 🌌 Final report
    acceptance_rate = 100 * (valid_count.value / total)

    # --- 🎯 CÁLCULO DAS MÉTRICAS DE ESTABILIDADE (MSI e SVI) ---
    coherence_values = np.array(list(sphy_evolution_list)) # Converte para array NumPy
    
    # Mean Stability Index (MSI): Média da Coerência em todos os frames
    mean_stability_index = np.mean(coherence_values) 
    
    # Stability Variance Index (SVI): Variância da Coerência (baixa = determinística)
    stability_variance_index = np.var(coherence_values) 
    # --- FIM DO CÁLCULO DAS MÉTRICAS ---

    print(f"\n✅ GHZ States accepted: {valid_count.value}/{total} | {acceptance_rate:.2f}%")

    # --- NOVO OUTPUT DAS MÉTRICAS CRUCIAIS ---
    print(f"\n" + "="*40)
    print(f"📊 Mean Stability Index (MSI): {mean_stability_index:.6f}")
    print(f"📊 Stability Variance Index (SVI): {stability_variance_index:.6f}")
    print(f"="*40)
    # --- FIM NOVO OUTPUT ---

    # 💾 Export CSV with SHA
    with open(csv_name, mode="w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow([
            "Frame", "Result", "H", "S", "C", "I",
            "Boost", "SPHY (%)", "Accepted", "UID_SHA256"
        ])
        writer.writerows(log_data)
    print(f"🧾 CSV saved: {csv_name}")

    # 📈 Generate plot
    plt.figure(figsize=(12, 5))
    plt.plot(range(1, total + 1), list(sphy_evolution_list), color="darkcyan", label="⧉ SPHY Coherence")
    
    # Determina a cor com base na aceitação
    acceptance_colors = ['green' if row[-2] == "✅" else 'red' for row in log_data]
    
    plt.scatter(
        range(1, total + 1), list(sphy_evolution_list),
        c=acceptance_colors,
        s=8, alpha=0.6
    )
    plt.axhline(90, color='gray', linestyle="dotted", linewidth=1, label="Threshold")
    plt.title(f"📡 HARPIA SPHY Evolution • {num_qubits} Qubits • {total:,} Frames\nMSI: {mean_stability_index:.4f} | SVI: {stability_variance_index:.4f}")
    plt.xlabel("Frames")
    plt.ylabel("SPHY Coherence (%)")
    plt.grid(True, linestyle="--", alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(img_name, dpi=300)
    print(f"📊 Graph saved as: {img_name}")
    plt.show()

# ▶️ Entry point
if __name__ == "__main__":
    qubits, pairs, initial_noise = input_parameters()
    print(f"\nNoise is {'enabled' if initial_noise else 'disabled'} by default.")
    run_simulation(num_qubits=qubits, total=pairs, noise_prob=0.3, initial_noise=initial_noise)