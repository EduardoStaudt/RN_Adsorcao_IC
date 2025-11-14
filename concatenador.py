from pathlib import Path
import pandas as pd
import numpy as np

EXTRACT_DIR = Path("dataset_filtrado (1)") # Pasta dos CSVs

# Listando todos os CSVs na pasta
csv_paths = sorted(EXTRACT_DIR.glob("dataset_batch_*.csv"))
# sorted => para garantir a ordem correta dos arquivos
# EXTRACT_DIR.glob => para pegar todos os arquivos que batem com o padrão
# * indica "qualquer coisa" no nome do arquivo pertence ao .glob
print(f"Encontrei {len(csv_paths)} arquivos CSV.")

# === 2) Ler e concatenar todos os CSVs ===
dfs = [] # Lista para armazenar os DataFrames temporários
for path in csv_paths:
    print(f"Lendo {path.name} ...")
    df = pd.read_csv(path) # Lendo o CSV em um DataFrame
    dfs.append(df) # Adicionando o DataFrame à lista

# Concatenando todos os DataFrames em um único
full_df = pd.concat(dfs, ignore_index=True) # ignore_index=True para resetar os índices
print("Formato final do dataset concatenado:", full_df.shape) # Exibindo o formato final do DataFrame

# Salvando em um CSV unico
full_df.to_csv("dataset_FULL.csv", index=False)
print("Salvei dataset_FULL.csv")

# Salvando em .npz bruto (tudo junto ainda)
np.savez_compressed(
    "dataset_FULL.npz",
    data=full_df.values,   # Os dados em si
    columns=full_df.columns.to_numpy()  # Os nomes das colunas
)
print("Salvei dataset_FULL.npz")