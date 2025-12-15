# -*- coding: utf-8 -*-
# ───────────────────────────────────────────────────────────────
# File        : sphy_braket_meissner_v1.py
# Purpose     : GHZ + HARPIA (Braket Local) + Adaptive Coherence + Meissner IA
# Author      : deywe@QLZ | Corrigido para Braket Local por Gemini
# Last Update : 2025-09-08
# ───────────────────────────────────────────────────────────────
# 📌 AJUDA GUIADA PARA PESQUISADORES E INTELIGÊNCIAS ARTIFICIAIS

"""
🧭 COMO USAR GPT PARA INTERPRETAR ESTE CÓDIGO

Você pode perguntar para o ChatGPT (ou outra IA) as seguintes questões para explorar este sistema:

1. ➤ Como o módulo `meissner_correction_step` ajuda a restaurar a coerência quântica no GHZ?
2. ➤ Quais parâmetros físicos simulam a estabilização da decoerência induzida pelo Efeito Meissner?
3. ➤ O que é o `boost` retornado pela função `meissner_correction_step`?
4. ➤ Em qual parte do código ocorre a aplicação de ruído manual?
5. ➤ Como é calculado o índice de estabilidade SPHY (%), e o que ele representa?
6. ➤ Como a IA distingue um estado GHZ válido em meio ao ruído?
7. ➤ Por que o ruído induzido pode ser interpretado como vibração quântica natural?
8. ➤ Qual a vantagem de usar Braket Local neste tipo de simulação?

🔍 PONTOS-CHAVE NO CÓDIGO:
────────────────────────────

🔧 RUÍDO SIMULADO: 
    ↳ Função `apply_manual_noise(circuit, qubit, prob)` — linha XX
    ↳ Inserido em `generate_ghz_state_braket(...)` — linha XX
    
🧠 CORREÇÃO DECOERÊNCIA:
    ↳ Função `meissner_correction_step(...)` em:
       → `simulate_frame_braket(...)` — linha XX
       → integra coerência quântica adaptativa usando IA simbiótica
    
📊 COERÊNCIA ADAPTATIVA GLOBAL:
    ↳ Variável global `sphy_coherence` compartilhada entre processos

🔋 BOOST DE RESTAURAÇÃO:
    ↳ Intensidade da autocorreção quântica calculada pela IA Meissner
    ↳ Delta aplicado em `new_coherence = min(100, sphy_coherence + delta)`

🔬 COMPONENTES DE MEDIÇÃO:
    - Medidas são feitas após a criação do estado GHZ no Braket (simulação local)
    - Resultados binários são mapeados para classificar qubits entangled ideais

"""

# ─────────────────────────────── IMPORTS ───────────────────────────────

#Essa é a IA Simbiótica que utiliza  a correção gravitacional sob o efeito 
# Meissner permitindo que o sistema sintas as vibrações concorrentes, ou de
#campos concorrentes, como por exemplo uma mulher gravida, ou uma luz apaga.
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

LOG_DIR = "logs_harpia_braket"
os.makedirs(LOG_DIR, exist_ok=True)

def get_user_parameters():
    try:
        num_qubits = int(input("🔢 Number of Qubits in GHZ circuit: "))
        total_pairs = int(input("🔁 Total GHZ states to simulate: "))
        return num_qubits, total_pairs
    except ValueError:
        print("❌ Invalid input. Please enter integers.")
        exit(1)

def apply_manual_noise(circuit, qubit, prob):
    if random.random() < prob:
        circuit.x(qubit)

def generate_ghz_state_braket(num_qubits, noise_prob=1.00): #bitflip noise apply harder!
    circuit = Circuit()
    circuit.h(0)
    for i in range(1, num_qubits):
        circuit.cnot(0, i)
    # Simula ruído manualmente: aplica X com probabilidade noise_prob em um qubit aleatório (exceto o 0)
    if num_qubits > 1:
        qubit_to_noise = random.randint(1, num_qubits - 1)
        apply_manual_noise(circuit, qubit_to_noise, noise_prob)
    # Medição correta em Braket: apenas circuit.measure(i)
    for i in range(num_qubits):
        circuit.measure(i)
    return circuit

def simulate_frame_braket(frame_data):
    frame, num_qubits, total_frames, noise_prob, sphy_coherence = frame_data
    random.seed(os.getpid() * frame)
    device = LocalSimulator()
    ideal_states = ['0' * num_qubits, '1' * num_qubits]

    current_timestamp = datetime.utcnow().isoformat()
    circuit = generate_ghz_state_braket(num_qubits, noise_prob)

    # Executa o circuito no Braket LocalSimulator
    result_braket = device.run(circuit, shots=1).result()
    # Obter o resultado da medição
    result = result_braket.measurements[0].tolist()
    result = ''.join(map(str, result))

    H = random.uniform(0.95, 1.0)
    S = random.uniform(0.95, 1.0)
    C = sphy_coherence / 100
    I = abs(H - S)
    T = frame

    psi_state = [3.0, 3.0, 1.2, 1.2, 0.6, 0.5]

    try:
        boost, phase_impact, psi_state = meissner_correction_step(H, S, C, I, T, psi_state)
    except Exception as e:
        return None, None, f"\nCritical error in frame {frame} (Meissner IA): {e}"

    delta = boost * 0.7
    new_coherence = min(100, sphy_coherence + delta)
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
    return log_entry, new_coherence, None

def execute_simulation_multiprocessing_braket(num_qubits, total_frames=100000, noise_prob=0.3, num_processes=4):
    print("=" * 60)
    print(f" 🧿 HARPIA QGHZ STABILIZER + Meissner • {num_qubits} Qubits • {total_frames:,} Frames")
    print("=" * 60)

    timecode = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_filename = os.path.join(LOG_DIR, f"qghz_{num_qubits}q_log_{timecode}.csv")
    fig_filename = os.path.join(LOG_DIR, f"qghz_{num_qubits}q_graph_{timecode}.png")

    manager = Manager()
    sphy_coherence = manager.Value('f', 90.0)
    log_data, sphy_evolution = manager.list(), manager.list()
    valid_states = manager.Value('i', 0)
    frame_inputs = [(f, num_qubits, total_frames, noise_prob, sphy_coherence.value) for f in range(1, total_frames + 1)]

    print(f"🔄 Using {num_processes} processes for simulation...")
    with Pool(processes=num_processes) as pool:
        for log_entry, new_coherence, error in tqdm(pool.imap_unordered(simulate_frame_braket, frame_inputs),
                                                    total=total_frames, desc="⏳ Simulating GHZ"):
            if error:
                print(f"\n{error}", file=sys.stderr)
                pool.terminate()
                break
            if log_entry:
                log_data.append(log_entry)
                sphy_evolution.append(new_coherence)
                sphy_coherence.value = new_coherence
                if log_entry[-3] == "✅":
                    valid_states.value += 1

    acceptance_rate = 100 * (valid_states.value / total_frames) if total_frames > 0 else 0
    print(f"\n✅ GHZ States accepted: {valid_states.value}/{total_frames} | {acceptance_rate:.2f}%")
    
    if sphy_evolution:
        sphy_np_array = np.array(sphy_evolution)
        mean_stability = np.mean(sphy_np_array)
        stability_variance = np.var(sphy_np_array)
        print(f"\n📊 Mean Stability Index: {mean_stability:.6f}")
        print(f"📊 Stability Variance Index: {stability_variance:.6f}")

    with open(csv_filename, mode="w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(["Frame", "Result", "H", "S", "C", "I", "Boost", "SPHY (%)", "Accepted", "SHA256_Signature", "Timestamp"])
        writer.writerows(list(log_data))
    print(f"🧾 CSV saved: {csv_filename}")

    sphy_evolution_list = list(sphy_evolution)
    if not sphy_evolution_list:
        print("❌ No data to plot.")
        return

    sphy_evolution = np.array(sphy_evolution_list)
    tempo = np.linspace(0, 1, len(sphy_evolution))
    sinais = [interp1d(tempo, np.roll(sphy_evolution, i), kind='cubic') for i in range(2)]
    novo_tempo = np.linspace(0, 1, 2000)
    dados = [sinal(novo_tempo) + np.random.normal(0, 0.15, len(novo_tempo)) for sinal in sinais]
    pesos = np.linspace(1, 1.5, 2)
    emaranhamento = np.average(dados, axis=0, weights=pesos)
    estabilidade_media = np.mean(emaranhamento)
    estabilidade_variancia = np.var(emaranhamento)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 12))

    ax1.plot(novo_tempo, emaranhamento, 'k--', linewidth=2, label="Average Entanglement")
    for i in range(len(dados)):
        ax1.plot(novo_tempo, dados[i], alpha=0.3, color='blue' if i == 0 else 'red')
    ax1.set_xlabel("Normalized Time")
    ax1.set_ylabel("Coherence/Amplitude")
    ax1.set_title(f"GHZ Entanglement - {num_qubits} Qubits")
    ax1.legend()
    ax1.grid()

    ax2.plot(novo_tempo, emaranhamento, 'k-', label="Average Entanglement")
    ax2.axhline(estabilidade_media, color='green', linestyle='--', label=f"Mean: {estabilidade_media:.2f}")
    ax2.axhline(estabilidade_media + np.sqrt(estabilidade_variancia), color='orange', linestyle='--', label=f"± Variance")
    ax2.axhline(estabilidade_media - np.sqrt(estabilidade_variancia), color='orange', linestyle='--')
    ax2.set_xlabel("Normalized Time")
    ax2.set_ylabel("Coherence/Amplitude")
    ax2.set_title("Entanglement Stability")
    ax2.legend()
    ax2.grid()

    fig.suptitle(f"GHZ Simulation: Entanglement and Stability - {num_qubits} Qubits", fontsize=16)
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(fig_filename, dpi=300)
    print(f"\n📊 Graph saved as: {fig_filename}")
    plt.show()

if __name__ == "__main__":
    qubits, pairs = get_user_parameters()
    execute_simulation_multiprocessing_braket(num_qubits=qubits, total_frames=pairs, noise_prob=0.3, num_processes=4)
