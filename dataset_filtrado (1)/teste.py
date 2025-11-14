import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Caminho para o arquivo
arquivo = "dataset_batch_7739.csv"

# Carrega o dataset
df = pd.read_csv(arquivo)

# Filtra colunas de temperatura
cols_tz = [c for c in df.columns if c.startswith("T_z")]
df_tz = df[cols_tz]

# Estatísticas básicas
desc = df_tz.describe().T
print("\n📊 Estatísticas de T_z:")
print(desc[["min", "max", "mean", "std"]])

# Contagem de negativos
negativos = (df_tz < 0).sum().sum()
total = df_tz.size
print(f"\n🔍 Valores negativos: {negativos} de {total} ({100*negativos/total:.2f}%)")

# Faixa geral
vmin, vmax = df_tz.min().min(), df_tz.max().max()
print(f"\n📈 Faixa global de T_z: {vmin:.4f} a {vmax:.4f}")

# Interpretação da escala
if vmax < 10 and abs(vmin) < 10:
    print("🧮 Escala parece ADIMENSIONAL (valores entre -1 e 1).")
elif 100 < vmax < 500:
    print("🌡️ Escala parece em KELVIN (valores típicos de 250–400 K).")
elif 0 < vmax < 100:
    print("🌡️ Escala parece em °C.")
else:
    print("⚠️ Escala fora do esperado, verificar origem dos dados.")

# Plot de uma amostra aleatória
idx = np.random.randint(0, len(df_tz))
plt.figure(figsize=(8,4))
plt.plot(df_tz.iloc[idx].values, marker="o")
plt.title(f"T_z - Seed 77394 - Amostra {idx}")
plt.xlabel("Posição axial (0–50)")
plt.ylabel("T_z")
plt.grid(True)
plt.tight_layout()
plt.show()
