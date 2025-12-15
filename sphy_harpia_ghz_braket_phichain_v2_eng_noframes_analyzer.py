# -*- coding: utf-8 -*-
# ───────────────────────────────────────────────────────────────
# File: sphy_harpia_ghz_braket_phichain_v2_eng_noframes_analyzer.py
# Purpose: Analyzer for the HARPIA QPoC (Quantum Proof of Coherence) simulation CSV
# ───────────────────────────────────────────────────────────────
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime
import sys
import os

# 🔹 Limiar B(t) usado no script de simulação original
BSCORE_THRESHOLD = 0.900

def run_qpoc_analyzer():
    """
    Solicita o caminho do CSV, calcula as métricas QPoC e reproduz o gráfico B-score.
    """
    
    print("=" * 70)
    print("    🔬 HARPIA QPoC Benchmark Analyzer (B-score Validation)")
    print("=" * 70)

    # 1. Solicitar o caminho completo do arquivo CSV
    try:
        csv_path = input("Por favor, digite o caminho COMPLETO do arquivo CSV (ex: phi_chain/qghz_4q_log_xxxxxx.csv): ")
        
        # 2. Carregar os dados
        # Forçar a leitura correta do B(t) (que pode vir como objeto/string)
        df = pd.read_csv(csv_path, dtype={'UID': str, 'UID_SHA256': str}) 
    except FileNotFoundError:
        print(f"❌ ERRO: O arquivo '{csv_path}' não foi encontrado. Verifique o caminho.")
        sys.exit(1)
    except Exception as e:
        print(f"❌ ERRO ao carregar o CSV: {e}")
        sys.exit(1)

    # 3. Normalizar e Mapear Colunas
    
    # Normalização robusta: minúsculas, remove espaços e caracteres especiais (exceto B(t))
    df.columns = df.columns.str.strip().str.replace(r'[^a-zA-Z0-9\s()%]', '', regex=True).str.lower().str.replace(' ', '_')
    
    # Colunas esperadas após a normalização
    bscore_col = 'b(t)'
    accepted_col = 'accepted'

    # Mapeamento e Verificação com tratamento de erro
    try:
        # Tenta converter B(t) para float, caso o Pandas não o tenha feito automaticamente
        df[bscore_col] = pd.to_numeric(df[bscore_col], errors='coerce')
        bscore_values = df[bscore_col].values
        
        # 🌟 CORREÇÃO 1: Criação do Timestamp
        # Gera timestamps artificiais para o Matplotlib, pois o CSV não tem coluna 'timestamp'.
        start_time = datetime.now()
        timestamps = [start_time + pd.Timedelta(milliseconds=1 * i) for i in range(len(df))]
        
        _ = df[accepted_col] # Verifica a presença da coluna 'accepted'
        
    except KeyError as e:
        print(f"❌ ERRO: Coluna essencial {e} não encontrada no CSV APÓS a normalização.")
        print(f"Colunas encontradas: {df.columns.tolist()}")
        print("\nVerifique se o script de simulação original foi executado corretamente.")
        sys.exit(1)
        
    total_entries = len(df)
    print(f"✅ Dados carregados com {total_entries} frames.")
    
    # 4. Cálculo das Métricas de Aceitação
    
    # Confirma o cálculo de aceitação baseado no B(t) > Limiar
    df['is_accepted'] = bscore_values >= BSCORE_THRESHOLD
    
    accepted_count = df['is_accepted'].sum()
    rejected_count = total_entries - accepted_count
    
    # 🌟 CORREÇÃO 2: Definição das variáveis de taxa (NameError resolvido)
    acceptance_rate = 100 * (accepted_count / total_entries)
    rejection_rate = 100 * (rejected_count / total_entries)

    # --- Reprodução das Métricas ---
    print("\n" + "=" * 50)
    print("       📊 RELATÓRIO DE MÉTRICAS QPoC REPRODUZIDO")
    print("=" * 50)
    print(f"✅ Total authorized accesses by the QPoC: {accepted_count}/{total_entries} | {acceptance_rate:.2f}%")
    print(f"❌ Total unauthorized accesses by the QPoC: {rejected_count}/{total_entries} | {rejection_rate:.2f}%")
    print("--------------------------------------------------")

    # 5. Configuração para Geração do Gráfico B-score por UID
    
    x_values = mdates.date2num(timestamps) # Converte Timestamps para o formato Matplotlib
    
    # Filtra os dados com base na aceitação e prepara os arrays
    df['x_values'] = x_values
    
    accepted_data = df[df['is_accepted']]
    rejected_data = df[~df['is_accepted']]
    
    accepted_x = accepted_data['x_values'].values
    accepted_y = accepted_data[bscore_col].values
    
    rejected_x = rejected_data['x_values'].values
    rejected_y = rejected_data[bscore_col].values
    
    # 6. Geração do Gráfico
    
    fig, ax = plt.subplots(figsize=(16, 9))
    
    # Pontos aceitos (acima do limiar)
    ax.scatter(accepted_x, accepted_y, color='green', label='B-score per UID Login (Accepted)', s=10, zorder=3)

    # Pontos negados (abaixo do limiar)
    ax.scatter(rejected_x, rejected_y, color='red', label='B-score per UID Login (Rejected)', s=10, zorder=3)

    # Linha de limiar
    ax.axhline(y=BSCORE_THRESHOLD, color='blue', linestyle='--', linewidth=1.5, label=f'Limiar B(t) = {BSCORE_THRESHOLD:.3f}')

    # Títulos e rótulos
    ax.set_title(" Continum Simulation B(t) - Vibrational Validation UID")
    ax.set_xlabel("Timestamp UTC")
    ax.set_ylabel(" Vibrational Coherence (B-score)")
    ax.grid(True, linestyle='--', alpha=0.6)
    ax.legend()
    
    # Formata o eixo X para exibir timestamps de forma legível
    fig.autofmt_xdate(rotation=45)
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M:%S.%f'))
    
    # 7. Salvar e mostrar o gráfico
    output_dir = os.path.dirname(csv_path) if os.path.dirname(csv_path) else '.'
    base_name = os.path.splitext(os.path.basename(csv_path))[0]
    img_name = os.path.join(output_dir, f"{base_name}_reproduced_bscore_chart.png")
    
    plt.tight_layout()
    plt.savefig(img_name, dpi=300)
    print(f"\n📊 Gráfico reproduzido salvo como: {img_name}")
    plt.show()

# Execução do analisador
if __name__ == "__main__":
    run_qpoc_analyzer()