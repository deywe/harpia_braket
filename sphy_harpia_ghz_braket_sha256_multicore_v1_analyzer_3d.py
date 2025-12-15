import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import os
import sys

# Define o diretório de saída para salvar o gráfico
OUTPUT_DIR = "logs_harpia_diagram"
os.makedirs(OUTPUT_DIR, exist_ok=True)

def generate_harpia_3d_diagram(csv_file_path):
    """
    Lê o log de simulação HARPIA (Hilbertless) e gera um 
    gráfico 3D de trajetória do controle simbiótico.
    
    Eixos: X=Frame (Tempo), Y=Coerência (SPHY %), Z=Boost (Correção)
    Cor (4D): Incerteza (I)
    """
    
    # 1. Verificação e Carregamento do Arquivo
    if not os.path.exists(csv_file_path):
        print(f"❌ Erro: Arquivo não encontrado no caminho: {csv_file_path}", file=sys.stderr)
        return

    try:
        # Tenta carregar o CSV, acomodando diferentes separadores
        df = pd.read_csv(csv_file_path, sep=None, engine='python')
    except Exception as e:
        print(f"❌ Erro na leitura do CSV: {e}", file=sys.stderr)
        return

    # 2. Normalização e Verificação das Colunas
    # Remove espaços, converte para minúsculas e remove caracteres não alfanuméricos exceto %
    df.columns = df.columns.str.strip().str.lower().str.replace(r'[^a-z0-9%]', '', regex=True)
    
    # Mapeamento das colunas esperadas para o gráfico 3D
    COLUMNS = {
        'x': 'frame',      # Coluna 'Frame'
        'y': 'sphy%',      # Coluna 'SPHY (%)'
        'z': 'boost',      # Coluna 'Boost'
        'c': 'i'           # Coluna 'I' (Cor/Incerteza)
    }

    data_cols = {}
    
    try:
        for key, col_name in COLUMNS.items():
            if col_name in df.columns:
                data_cols[key] = df[col_name].values
            else:
                # Se a coluna 'frame' não for encontrada, tenta 'indice' ou a primeira coluna
                if col_name == 'frame' and df.columns[0] == 'frame':
                    data_cols[key] = df[df.columns[0]].values
                else:
                    raise KeyError(f"Coluna obrigatória '{col_name}' não encontrada.")
                
    except KeyError as e:
        print(f"\n❌ Erro: {e}. Certifique-se de que o CSV contém os cabeçalhos necessários: Frame, SPHY (%), Boost e I.", file=sys.stderr)
        return

    # 3. Geração do Gráfico 3D
    
    fig = plt.figure(figsize=(14, 10))
    ax = fig.add_subplot(111, projection='3d')

    # Usa 'I' (Incerteza/Entropia) para colorir os pontos (4ª Dimensão)
    colors = data_cols['c'] 
    
    # Plota a trajetória 3D (Scatter plot)
    scatter = ax.scatter(
        data_cols['x'], 
        data_cols['y'], 
        data_cols['z'], 
        c=colors, 
        cmap='viridis', # Mapa de cores vibrante
        s=10,          # Tamanho do ponto ligeiramente maior
        alpha=0.8
    )

    # Conecta os pontos com uma linha sutil para mostrar a trajetória (evolução ao longo do tempo)
    ax.plot(
        data_cols['x'], 
        data_cols['y'], 
        data_cols['z'], 
        color='darkblue', 
        linewidth=0.7, 
        alpha=0.5
    )

    # 4. Ajustes Visuais e Títulos
    
    # Barra de Cores
    cbar = fig.colorbar(scatter, ax=ax, pad=0.1)
    cbar.set_label('4D: Incerteza / Entropia (I)', rotation=270, labelpad=20)

    # Rótulos dos Eixos
    ax.set_xlabel("Eixo X: Frame da Simulação (Tempo)", fontsize=12)
    ax.set_ylabel("Eixo Y: SPHY Coerência (%)", fontsize=12)
    ax.set_zlabel("Eixo Z: Boost (Correção Gravitacional)", fontsize=12)
    
    file_name_base = os.path.basename(csv_file_path).replace('.csv', '')
    ax.set_title(f"Diagrama 3D de Trajetória do Controle Simbiótico HARPIA\nFonte: {file_name_base}", fontsize=14)

    # Ajusta o ângulo de visão
    ax.view_init(elev=25, azim=120) 
    ax.grid(True, linestyle='--', alpha=0.6)

    plt.tight_layout()
    
    # 5. Salva e Mostra
    output_filename = os.path.join(OUTPUT_DIR, f"diagram_3d_harpia_{file_name_base}.png")
    plt.savefig(output_filename, dpi=300)
    
    print("-" * 50)
    print(f"✅ Gráfico 3D gerado com sucesso!")
    print(f"💾 Arquivo salvo em: {output_filename}")
    print("-" * 50)

    plt.show()

# --- EXECUÇÃO ---

if __name__ == "__main__":
    
    # Garante que as bibliotecas necessárias estejam instaladas
    required_libraries = ['pandas', 'numpy', 'matplotlib']
    for lib in required_libraries:
        try:
            __import__(lib)
        except ImportError:
            print(f"A biblioteca '{lib}' é necessária. Instalando...")
            try:
                import subprocess
                subprocess.check_call([sys.executable, "-m", "pip", "install", lib])
                __import__(lib)
            except Exception as e:
                print(f"Erro ao instalar {lib}: {e}")
                sys.exit(1)

    # Solicita o caminho do CSV ao usuário
    csv_path = input("🔗 Digite o caminho completo do arquivo CSV para o diagrama 3D: ")
    
    if csv_path:
        generate_harpia_3d_diagram(csv_path.strip())
    else:
        print("Operação cancelada.")