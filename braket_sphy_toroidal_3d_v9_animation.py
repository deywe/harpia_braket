# -*- coding: utf-8 -*-
# ───────────────────────────────────────────────────────────────
# File: braket_sphy_toroidal_3d_v9_animation.py
# Purpose: RENDERIZAÇÃO DA ANIMAÇÃO 3D DO CAMPO SPHY (PÓS-PROCESSAMENTO INTERATIVO)
# Author: deywe@QLZ | Adapted by Gemini AI (Interactive File Path)
# ───────────────────────────────────────────────────────────────
import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation 
from scipy.interpolate import griddata
import glob
from datetime import datetime

# === Configuração da Malha Toroidal SPHY (DEVE CORRESPONDER AO PRINCIPAL) ===
GRID_SIZE = 2 
NUM_WIRES = GRID_SIZE * GRID_SIZE # 4 Qubits (0, 1, 2, 3)
LOG_DIR = "logs_sphy_toroidal_3d" # Mantido para fins de salvamento da saída da animação

def get_csv_path():
    """Solicita interativamente ao usuário o caminho do arquivo CSV."""
    if not os.path.isdir(LOG_DIR):
         os.makedirs(LOG_DIR, exist_ok=True)
         print(f"⚠️ Criado diretório de logs: {LOG_DIR}")

    print("=" * 60)
    print("🎬 Renderizador de Animação 3D do Campo SPHY ($\phi$)")
    print("=" * 60)
    
    # Tenta listar arquivos recentes como sugestão
    search_path = os.path.join(LOG_DIR, "toroidal_*q_log_*.csv")
    recent_files = sorted(glob.glob(search_path), key=os.path.getctime, reverse=True)[:5]

    if recent_files:
        print("\nArquivos CSV recentes encontrados (Sugestões):")
        for i, f in enumerate(recent_files):
            print(f"  [{i+1}]: {os.path.basename(f)}")
        print("  [0]: Inserir caminho manual")
        
        choice = input("Selecione o número do arquivo ou insira '0' para o caminho manual: ")
        
        if choice.isdigit() and 1 <= int(choice) <= len(recent_files):
            return recent_files[int(choice) - 1]
    
    # Solicita caminho manual
    while True:
        path = input("\n➡️ Por favor, insira o **caminho completo** do arquivo CSV SPHY a ser renderizado:\n> ")
        if os.path.exists(path) and path.endswith('.csv'):
            return path
        print("❌ Erro: O caminho não existe ou não é um arquivo CSV válido. Tente novamente.")

def generate_3d_animation(csv_filename):
    """
    Cria uma animação 3D da forma do Campo SPHY evoluindo ao longo do tempo.
    """
    if not csv_filename:
        print("❌ Operação cancelada. Nenhum arquivo CSV válido fornecido.")
        return

    try:
        df = pd.read_csv(csv_filename)
        print(f"\n✔️ Carregando dados de: {os.path.basename(csv_filename)} ({len(df)} frames)")
    except Exception as e:
        print(f"❌ Erro ao ler o arquivo CSV. Verifique a formatação. Detalhes: {e}")
        return

    # Nome do arquivo de saída da animação (salva no LOG_DIR)
    base_name = os.path.basename(csv_filename).replace(".csv", "")
    anim_filename = os.path.join(LOG_DIR, f"{base_name}_animation_SPHY.mp4")

    # Mapeamento de Qubits para Coordenadas 2D (X, Y fixos)
    X_fixed = np.array([0, 0, 1, 1])
    Y_fixed = np.array([0, 1, 0, 1])
    
    # Colunas de Fase (Indexado em 1)
    phase_cols = [f"Qubit_{i+1}_Phase" for i in range(NUM_WIRES)]
    
    # Preparação da Figura 3D
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # Determina o alcance Z mínimo e máximo para estabilizar o gráfico
    min_z = df[phase_cols].values.min() * 1.1
    max_z = df[phase_cols].values.max() * 1.1

    ax.set_zlim(min_z, max_z)
    ax.set_xlabel('Coordenada X')
    ax.set_ylabel('Coordenada Y')
    ax.set_zlabel('Fase Pauli Z (Campo SPHY $\phi$)')

    # Interpolação base (meshgrid)
    xi = np.linspace(X_fixed.min(), X_fixed.max(), 50)
    yi = np.linspace(Y_fixed.min(), Y_fixed.max(), 50)
    XI, YI = np.meshgrid(xi, yi)

    # Função de Atualização por Frame
    def update_surface(frame_index):
        ax.clear() # Limpa o frame anterior

        # Configurações do Eixo (devem ser mantidas em cada frame)
        ax.set_zlim(min_z, max_z) 
        ax.set_xlabel('Coordenada X')
        ax.set_ylabel('Coordenada Y')
        ax.set_zlabel('Fase Pauli Z (Campo SPHY $\phi$)')
        ax.set_title(f'Campo SPHY $\phi$ - Frame: {frame_index + 1} / {len(df)}')
        
        # 1. Dados do Frame Atual
        Z_current = df[phase_cols].iloc[frame_index].values
        
        # 2. Interpolação
        ZI_current = griddata((X_fixed, Y_fixed), Z_current, (XI, YI), method='cubic')

        # 3. Plotar a nova superfície
        new_surf = ax.plot_surface(XI, YI, ZI_current, cmap='viridis', edgecolor='none', alpha=0.9)
        ax.scatter(X_fixed, Y_fixed, Z_current, color='red', s=30)
        
        return new_surf,

    print("\n🎬 Gerando Animação 3D (Isso pode ser demorado. Requer FFmpeg)...")
    
    # Cria a animação - Pula 10 frames para renderização mais rápida, 20 FPS
    skip_frames = 10 
    frames_to_render = np.arange(0, len(df), skip_frames)

    ani = FuncAnimation(fig, update_surface, frames=frames_to_render, 
                        interval=50, blit=False)
    
    try:
        # Salva a animação
        ani.save(anim_filename, writer='ffmpeg', fps=20) 
        print(f"✔️ Animação 3D da Malha SPHY salva: {anim_filename}")
    except ValueError as e:
        print(f"❌ Erro ao salvar animação. Certifique-se de que o FFmpeg está instalado e no PATH. Detalhes: {e}")
    except Exception as e:
        print(f"❌ Erro desconhecido durante a renderização: {e}")

    plt.close(fig)

if __name__ == "__main__":
    csv_to_render = get_csv_path()
    generate_3d_animation(csv_to_render)