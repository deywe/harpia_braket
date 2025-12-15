# ───────────────────────────────────────────────────────────────
# File: braket_sphy_toroidal_3d_v9_analyzer.py
# Purpose: HARPIA SPHY Benchmarking Analyzer (Metrics and Graphics)
# Author: deywe@QLZ | Modified by Gemini (Nov/2025)
# ───────────────────────────────────────────────────────────────
import os
import csv
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import sys
from scipy.interpolate import interp1d, griddata # Para plotagem 3D

# 🔧 Configure output directory
ANALYZER_LOG_DIR = "logs_harpia_braket_analyzer"
os.makedirs(ANALYZER_LOG_DIR, exist_ok=True)

def plot_3d_sphy_field(df, csv_filename, fig_filename_3d, num_wires):
    """
    Gera o gráfico 3D da forma do campo SPHY (média da fase Z dos qubits).
    """
    try:
        # Verifica se o número de qubits é 4 (o padrão do toroidal 2x2)
        if num_wires != 4:
            print(f"⚠️ Aviso: Plotagem 3D projetada para 4 qubits. O arquivo tem {num_wires}. Adaptando...")
        
        phase_cols = [f"Qubit_{i+1}_Phase" for i in range(num_wires)]
        
        if not all(col in df.columns for col in phase_cols):
             print(f"❌ Erro: Colunas de fase não encontradas para Plotagem 3D. Esperadas: {phase_cols}")
             return

        mean_phases = df[phase_cols].mean().values

        # Coordenadas 2D para a rede 2x2 (4 qubits): (0,0), (0,1), (1,0), (1,1)
        # Adaptável para o número de qubits lido (se for 4, usa este padrão)
        if num_wires == 4:
            X = np.array([0, 0, 1, 1])
            Y = np.array([0, 1, 0, 1])
        else:
            # Simplifica a plotagem para 1D se não for a rede 2x2
            X = np.arange(num_wires)
            Y = np.zeros(num_wires)

        Z = mean_phases 

        # Interpolação para suavizar o gráfico 3D (ignora se for 1D)
        if num_wires == 4:
            xi = np.linspace(X.min(), X.max(), 50)
            yi = np.linspace(Y.min(), Y.max(), 50)
            XI, YI = np.meshgrid(xi, yi)
            ZI = griddata((X, Y), Z, (XI, YI), method='cubic')
        else:
             XI, YI, ZI = X, Y, Z # Não faz interpolação em 1D

        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')

        if num_wires == 4:
            surf = ax.plot_surface(XI, YI, ZI, cmap='viridis', edgecolor='none', alpha=0.9)
            ax.set_zlim(Z.min() * 1.1 if Z.min() != Z.max() else -1.0, Z.max() * 1.1 if Z.min() != Z.max() else 1.0)
            fig.colorbar(surf, shrink=0.5, aspect=5, label='Phase/Population (|0> -> |1>)')
        
        ax.scatter(X, Y, Z, color='red', s=50, label='Average Qubit Phase')

        ax.set_xlabel('X Coordinate (Qubit)')
        ax.set_ylabel('Y Coordinate (Qubit)')
        ax.set_zlabel(r'Average Pauli Z Phase (SPHY Field $\phi$)')
        file_name_base = os.path.basename(csv_filename).replace(".csv", "")
        ax.set_title(f'3D SPHY Shape Field - {file_name_base}')
        
        fig_filename_3d = os.path.join(ANALYZER_LOG_DIR, f"analisado_{file_name_base}_graph_3D.png")
        plt.savefig(fig_filename_3d, dpi=300)
        print(f"🖼️ Gráfico 3D SPHY Shape Field salvo: {fig_filename_3d}")

    except Exception as e:
        print(f"❌ Erro ao gerar o gráfico 3D: {e}")

def plot_2d_stability(sphy_evolution, total_frames, acceptance_rate, mean_stability, stability_variance, csv_filename):
    """
    Gera o gráfico 2D de Estabilidade (similar ao original).
    """
    try:
        if not sphy_evolution:
            print("❌ Não há dados de SPHY (%) para plotar o gráfico 2D.")
            return

        sphy_evolution = np.array(sphy_evolution)
        time_points = np.linspace(0, 1, len(sphy_evolution))
        
        # Replicação da lógica de suavização/redundância do script original
        n_redundancias = 2 
        signals = [interp1d(time_points, np.roll(sphy_evolution, i), kind='cubic') for i in range(n_redundancias)]
        new_time = np.linspace(0, 1, 2000)
        
        data = [signals[i](new_time) + np.random.normal(0, 0.15, len(new_time)) * (1 + i * 0.1) for i in range(n_redundancias)]
        weights = np.linspace(1, 1.5, n_redundancias)
        tunneling_stability = np.average(data, axis=0, weights=weights)

        stability_mean_2 = np.mean(data[1]) 
        stability_variance_2 = np.var(data[1])

        # --- Creating Subplots ---
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 12))

        # Graph 1: SPHY Coherence Signal (Amplitude)
        ax1.set_title("SPHY Coherence Evolution (Signal 1: Amplitude)")
        for i in range(n_redundancias):
            ax1.plot(new_time, data[i], alpha=0.3, color='blue')  
        ax1.plot(new_time, tunneling_stability, 'k--', linewidth=2, label="Weighted Average Stability")
        ax1.set_xlabel("Normalized Time")
        ax1.set_ylabel("Coherence/Amplitude")
        ax1.legend()
        ax1.grid()

        # Graph 2: SPHY Coherence Signal (Stability)
        ax2.set_title(f"SPHY Coherence Evolution (Signal 2: Stability) | Acceptance: {acceptance_rate:.2f}%")
        ax2.plot(new_time, data[1], color='red', alpha=0.7, label='Coherence Signal (2)')
        
        # Linhas de Média e Variância
        ax2.axhline(stability_mean_2, color='green', linestyle='--', label=f"Mean: {stability_mean_2:.2f}")
        ax2.axhline(stability_mean_2 + np.sqrt(stability_variance_2), color='orange', linestyle='--', label=f"± Variance")
        ax2.axhline(stability_mean_2 - np.sqrt(stability_variance_2), color='orange', linestyle='--')

        ax2.set_xlabel("Normalized Time")
        ax2.set_ylabel("Coherence/Amplitude")
        ax2.legend()
        ax2.grid()

        file_name_base = os.path.basename(csv_filename).replace(".csv", "")
        fig.suptitle(f"Quantum Tunneling Simulation: {total_frames} Attempts (Toroidal SPHY)\nMSI: {mean_stability:.6f} | SVI: {stability_variance:.6f}", fontsize=16)
        fig.tight_layout(rect=[0, 0.03, 1, 0.95])
        
        fig_filename = os.path.join(ANALYZER_LOG_DIR, f"analisado_{file_name_base}_graph_2D.png")
        plt.savefig(fig_filename, dpi=300)
        print(f"🖼️ Gráfico 2D Stability salvo: {fig_filename}")
        plt.show()

    except Exception as e:
        print(f"❌ Erro ao gerar o gráfico 2D: {e}")


def run_analyzer():
    """
    Roda o analisador de CSV, calcula métricas e gera os dois gráficos.
    """
    print("=" * 60)
    print("      🧪 HARPIA SPHY BENCHMARK ANALYZER v1.0")
    print("=" * 60)
    
    csv_path = input("📁 Digite o caminho completo do arquivo CSV para análise: ")
    
    if not os.path.exists(csv_path):
        print(f"❌ Erro: Arquivo não encontrado no caminho: {csv_path}")
        return

    try:
        # 2. Leitura e Preparação dos Dados
        # Força 'Result' para string para evitar o erro de formato de números binários
        df = pd.read_csv(csv_path, dtype={'Result': str})
        
        # Garante que as colunas essenciais existem
        required_cols = ["SPHY (%)", "Accepted", "Frame", "Result"]
        if not all(col in df.columns for col in required_cols):
            missing = [col for col in required_cols if col not in df.columns]
            print(f"❌ Erro: O arquivo CSV não contém as colunas necessárias: {missing}")
            return

        df.dropna(subset=["SPHY (%)"], inplace=True)
        
        # Converte a coluna SPHY para numpy array
        coherence_values = df["SPHY (%)"].values
        total_frames = len(df)
        
        # Tenta descobrir o número de qubits com base nas colunas de fase
        phase_cols = [col for col in df.columns if "_Phase" in col]
        num_wires = len(phase_cols)
        
        # 3. Cálculo das Métricas de Estabilidade
        mean_stability_index = np.mean(coherence_values) 
        stability_variance_index = np.var(coherence_values)
        
        valid_count = df["Accepted"].eq("✅").sum()
        acceptance_rate = 100 * (valid_count / total_frames) if total_frames > 0 else 0
        
        # 4. Impressão do Log Final (Relatório)
        file_name_base = os.path.basename(csv_path).replace(".csv", "")
        
        print("\n" + "=" * 40)
        print(f"🧾 RELATÓRIO DE ANÁLISE SPHY - {file_name_base}")
        print("=" * 40)
        print(f"🔢 Total de Frames Analisados: {total_frames:,}")
        print(f"🔢 Número de Qubits Detectados: {num_wires}")
        print(f"✅ Estados Aceitos: {valid_count}/{total_frames} | {acceptance_rate:.2f}%")
        print("\n--- Métricas de Estabilidade SPHY ---")
        print(f"📊 Mean Stability Index (MSI): {mean_stability_index:.6f}")
        print(f"📊 Stability Variance Index (SVI): {stability_variance_index:.6f}")
        print("-" * 40)

        # 5. Geração dos Gráficos
        
        # Gráfico 2D: Estabilidade
        plot_2d_stability(coherence_values.tolist(), total_frames, acceptance_rate, mean_stability_index, stability_variance_index, csv_path)
        
        # Gráfico 3D: Campo SPHY
        if num_wires > 0:
            plot_3d_sphy_field(df, csv_path, "", num_wires)

    except Exception as e:
        print(f"❌ Ocorreu um erro fatal durante a análise: {e}")
        
if __name__ == "__main__":
    # Garante que as bibliotecas necessárias estejam instaladas (opcional, mas robusto)
    try:
        import pandas as pd
    except ImportError:
        print("A biblioteca 'pandas' é necessária. Por favor, instale com: pip install pandas")
        sys.exit(1)
        
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("A biblioteca 'matplotlib' é necessária. Por favor, instale com: pip install matplotlib")
        sys.exit(1)
        
    run_analyzer()