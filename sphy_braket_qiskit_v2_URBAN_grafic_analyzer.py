# -*- coding: utf-8 -*-
# ───────────────────────────────────────────────────────────────
# File        : sphy_analyzer_benchmark_v2.py (CORRIGIDO PARA MODOS 1/2)
# Purpose     : Analyzer for sphy_braket_qiskit_v2_URBAN.py simulation logs.
#               Reproduces metrics and the dual-axis graph.
# Author      : Gemini
# ───────────────────────────────────────────────────────────────
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import os
import sys
import re

# -----------------------------------------------------------------------------
#                   CONFIGURATION AND GLOBAL VARIABLES
# -----------------------------------------------------------------------------

# Limiar de otimização de congestionamento usado no script de simulação.
OPTIMAL_CONGESTION_THRESHOLD = 30.0 

def get_coherence_name(df):
    """
    Determina o nome da coluna de coerência/eficiência e o modo de simulação
    baseado nos nomes normalizados das colunas.
    Retorna (coherence_name, coherence_col_normalized, mode_id).
    """
    normalized_cols = df.columns
    
    if 'flow_efficiency' in normalized_cols:
        return "Flow Efficiency", 'flow_efficiency', '2' # Urban Mode
    if 'sphy_coherence' in normalized_cols:
        return "SPHY Coherence", 'sphy_coherence', '1' # GHZ Mode

    return None, None, None

def load_and_prepare_data():
    """Solicita o caminho do CSV e carrega/prepara os dados."""
    print("=" * 70)
    print("    🔬 HARPIA SPHY Benchmark Analyzer (Metrics & Graph Reproduction)")
    print("=" * 70)

    try:
        csv_path = input("Por favor, digite o caminho COMPLETO do arquivo CSV: ")
        
        # 1. Carregar os dados
        df = pd.read_csv(csv_path, dtype={'SHA256_Signature': str})
    except FileNotFoundError:
        print(f"❌ ERRO: O arquivo '{csv_path}' não foi encontrado. Verifique o caminho.")
        sys.exit(1)
    except Exception as e:
        print(f"❌ ERRO ao carregar o CSV: {e}")
        sys.exit(1)

    # 2. Normalizar Nomes de Colunas
    # Nota: A coluna 'Result (GHZ/CI)' vira 'result_ghzci'
    df.columns = df.columns.str.strip().str.replace(r'[^a-zA-Z0-9\s%]', '', regex=True).str.lower().str.replace(' ', '_')
    
    # 3. Determinar o Modo
    coherence_name, coherence_col, mode = get_coherence_name(df)
    
    if not coherence_name:
        print("❌ ERRO: Coluna de SPHY Coherence ou Flow Efficiency não encontrada no CSV.")
        print(f"Colunas encontradas após normalização: {df.columns.tolist()}")
        print("Verifique se o CSV foi gerado pelo script URBAN e se as colunas estão corretas.")
        sys.exit(1)
        
    # 4. Preparar Dados
    try:
        df['timestamps'] = pd.to_datetime(df['timestamp'])
        df['coherence_value'] = df[coherence_col]
        
        # Colunas específicas do modo 2 (Urban)
        if mode == '2':
            # A coluna 'result_ghzci' armazena o Congestion Index (CI)
            df['ci_value'] = df['result_ghzci'] 
            # Colunas de onda para gráfico de dois eixos
            df['h_value'] = df['h']
            df['s_value'] = df['s']
            df['i_value'] = df['i'] 
            
    except KeyError as e:
        print(f"❌ ERRO: Coluna essencial {e} não encontrada no CSV. Verifique se o cabeçalho está completo.")
        sys.exit(1)
        
    # 5. Coluna de aceitação
    df['is_accepted'] = df['accepted'].apply(lambda x: True if '✅' in str(x) else False)
    
    total_frames = len(df)
    print(f"✅ Dados carregados com {total_frames} frames. Modo: {'Urban (Flow)' if mode == '2' else 'GHZ (Stabilization)'}")
    
    return df, mode, total_frames, csv_path, coherence_name

# -----------------------------------------------------------------------------
#                         METRICS REPRODUCTION
# -----------------------------------------------------------------------------

def reproduce_metrics(df, mode, total_frames, coherence_name):
    """Reproduz e imprime as métricas."""
    
    accepted_count = df['is_accepted'].sum()
    acceptance_rate = 100 * (accepted_count / total_frames)
    
    print("\n" + "=" * 50)
    print("       📊 RELATÓRIO DE MÉTRICAS REPRODUZIDO")
    print("=" * 50)
    print(f"✅ Accepted Frames: {accepted_count}/{total_frames} | {acceptance_rate:.2f}%")
    
    efficiency_values = df['coherence_value'].values

    if mode == '2':
        ci_values = df['ci_value'].values
        
        print("\n--- Congestion Index (CI) Metric ---")
        print(f"CI Mean: {np.mean(ci_values):.4f}")
        print(f"CI Standard Deviation: {np.std(ci_values):.4f}")
        print(f"CI Minimum/Maximum: {np.min(ci_values):.4f} / {np.max(ci_values):.4f}")
        print(f"Optimal CI Threshold: {OPTIMAL_CONGESTION_THRESHOLD}")

        print("\n--- Efficiency (SPHY) Metric ---")
    else:
        print(f"\n--- {coherence_name} Metric ---")

    print(f"Mean {coherence_name}: {np.mean(efficiency_values):.4f}%")
    print(f"Final {coherence_name}: {df['coherence_value'].iloc[-1]:.4f}%")
    print(f"Standard Deviation: {np.std(efficiency_values):.4f}")
    print("-" * 50)


# -----------------------------------------------------------------------------
#                         GRAPH REPRODUCTION
# -----------------------------------------------------------------------------

def reproduce_graph(df, mode, total_frames, coherence_name, csv_path):
    """Reproduz o gráfico de acordo com o modo de simulação."""
    
    # MATPLOTLIB OPTIMIZATION CONFIGURATION (copied from original script)
    plt.rcParams['agg.path.chunksize'] = 10000 
    plt.rcParams['path.simplify'] = True
    plt.rcParams['path.simplify_threshold'] = 0.5 

    sphy_evolution = df['coherence_value'].values
    
    # Define o downsample factor (igual ao script original)
    downsample_factor = 1
    if total_frames > 20000:
        downsample_factor = 100 
    
    frames = np.array(range(1, total_frames + 1))
    frames_plot = frames[::downsample_factor]
    
    # -------------------------------------------------------------------------
    #                            MODE 2: URBAN (Dual-Axis Graph)
    # -------------------------------------------------------------------------
    if mode == '2':
        h_values = df['h_value'].values[::downsample_factor]
        s_values = df['s_value'].values[::downsample_factor]
        i_values = df['i_value'].values[::downsample_factor]
        sphy_plot = sphy_evolution[::downsample_factor] 
        
        y_label = coherence_name
        
        fig, ax1 = plt.subplots(figsize=(14, 7))

        # Eixo 1: Oscilações (H, S, I)
        ax1.set_xlabel(f"Frames (Sampling: 1 in every {downsample_factor})")
        ax1.set_ylabel("Noise Factors (0.0 to 1.0)")
        ax1.plot(frames_plot, h_values, color='gold', linewidth=1, label="H (Traffic Volume)")
        ax1.plot(frames_plot, s_values, color='red', linewidth=1, alpha=0.6, label="S (Structural Stress)")
        # Multiplica I por 10 (igual ao script original)
        ax1.plot(frames_plot, i_values * 10, color='purple', linestyle='--', linewidth=1, label="I (Imbalance x10)")
        ax1.tick_params(axis='y')
        ax1.set_ylim(0.0, 1.1)

        # Eixo 2: SPHY Coherence (Linha de Controle)
        ax2 = ax1.twinx()  
        ax2.set_ylabel(y_label, color='darkcyan') 
        ax2.plot(frames_plot, sphy_plot, color='darkcyan', linewidth=3, label=f"{coherence_name} (SPHY)")
        ax2.tick_params(axis='y', labelcolor='darkcyan')
        ax2.set_ylim(50, 105)
        
        # Linha Média
        mean_sphy = np.mean(sphy_evolution)
        ax2.axhline(mean_sphy, color='darkcyan', linestyle=':', alpha=0.7, label=f"SPHY Mean ({mean_sphy:.2f}%)")

        # Título e Legendas
        fig.suptitle(f"Wave Analysis: Adaptive Coherence x Imbalance ({total_frames:,} Frames) - REPRODUCED", fontsize=16)
        
        # Combina legendas dos dois eixos
        lines, labels = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax2.legend(lines + lines2, labels + labels2, loc='lower right')
        ax1.grid(True, linestyle="--", alpha=0.3)

    # -------------------------------------------------------------------------
    #                          MODE 1: GHZ (Convergence Graph)
    # -------------------------------------------------------------------------
    elif mode == '1':
        y_label = coherence_name
        title_prefix = "📡 HARPIA QGHZ STABILIZER + Meissner"
        
        plt.figure(figsize=(12, 5))
        plt.plot(frames, sphy_evolution, color="blue", label=f"⧉ SPHY {coherence_name}")
        plt.axhline(90, color='gray', linestyle="dotted", linewidth=1, label="Initial Coherence (90%)")
        plt.title(f"{title_prefix} • {total_frames:,} Frames - REPRODUCED")
        plt.xlabel("Frames")
        plt.ylabel(y_label)
        plt.grid(True, linestyle="--", alpha=0.3)
        plt.legend()
        

    # 7. Salvar e mostrar o gráfico
    output_dir = os.path.dirname(csv_path) if os.path.dirname(csv_path) else '.'
    base_name = os.path.splitext(os.path.basename(csv_path))[0]
    img_name = os.path.join(output_dir, f"{base_name}_REPRODUCED_BENCHMARK.png")
    
    plt.tight_layout()
    plt.savefig(img_name, dpi=300)
    print(f"\n📊 Gráfico reproduzido salvo como: {img_name}")
    plt.show()

# -----------------------------------------------------------------------------
#                               MAIN EXECUTION
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    df, mode, total_frames, csv_path, coherence_name = load_and_prepare_data()
    reproduce_metrics(df, mode, total_frames, coherence_name)
    reproduce_graph(df, mode, total_frames, coherence_name, csv_path)