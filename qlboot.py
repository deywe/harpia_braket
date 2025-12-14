import os
import sys

# ===============================
# 🧠 Configuração do Ambiente
# ===============================

BASE_DIR = os.path.abspath(os.path.dirname(__file__))
ROOT = BASE_DIR

# Adiciona caminhos principais à stack de importação
sys.path.insert(0, ROOT)  # Raiz do projeto (interface, boot, etc)
sys.path.insert(0, os.path.join(ROOT, "qfs_e"))  # Acesso ao núcleo QFS-E
sys.path.insert(0, os.path.join(ROOT, "qfs_e/modules"))  # Módulos promovidos da IA fractal

# ===============================
# 🚀 Importações do Sistema
# ===============================

from boot.qosgenesis import QOSGenesis
from boot.qos0.qos0_pipeline import start_qos0_stage
from boot.qos0.qos0_pipeline import metadata_shield_run, ethics_gate_run



# ===============================
# 🔁 Execução principal
# ===============================

if __name__ == "__main__":
    args = sys.argv
    system = QOSGenesis()

    if "--terminal" in args:
        system.only_boot_terminal()
    else:
        system.run_all()
        start_qos0_stage()

