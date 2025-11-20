from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
import joblib

# ================================
# 1) Caminhos: modelo e scalers
# ================================
BASE_DIR = Path(__file__).resolve().parent          # .../src
MODEL_DIR = BASE_DIR.parent / "models"              # .../models

model_path = MODEL_DIR / "best_model.keras"
model = load_model(model_path, compile=False)

scaler_in_path = MODEL_DIR / "scaler_input.save"
scaler_out_path = MODEL_DIR / "scaler_output.save"

scaler_X = joblib.load(scaler_in_path)
scaler_Y = joblib.load(scaler_out_path)

# ================================
# 2) Parâmetros de entrada
# ================================
param_cols = [
    "L", "Nz", "eps", "rho_B", "u", "D_ax", "kL", "qmax", "b", "n",
    "lam_z", "rho_g", "cp_g", "cp_s", "D_col", "h_w", "T_wall", "dH",
    "dt", "t_end", "C_in", "T_in",
]

# valores "coerentes" dentro (mais ou menos) dos ranges do treino
inputs_dict = {
    "L": 1,      # entre 0.05 e 1.00
    "Nz": 51,      # fixo
    "eps": 0.35,   # entre 0.20 e 0.80
    "rho_B": 70,  # entre 50 e 500
    "u": 1.5,      # entre 0.05 e 1.00
    "D_ax": 1e-6,  # entre 1e-7 e 1e-5
    "kL": 0.01,    # entre 1e-3 e 1e-1
    "qmax": 2.5,   # entre 0.1 e 10
    "b": 0.8,      # entre 0.1 e 10
    "n": 1.2,      # entre 0.1 e 10
    "lam_z": 0.4,  # entre 0.1 e 1.0
    "rho_g": 1.2,  # entre 0.2 e 2.0
    "cp_g": 1000,  # entre 800 e 1200
    "cp_s": 800,   # entre 800 e 1200
    "D_col": 0.05, # entre 0.01 e 0.50
    "h_w": 10.0,   # entre 10 e 50
    "T_wall": 300, # entre 298.15 e 318.15
    "dH": -35000, # entre 25000 e 65000  (no treino era positivo)
    "dt": 0.1,     # típico: t_end/1000; você pode ajustar
    "t_end": 1000.0,
    "C_in": 1.0,   # entre 0.1 e 5.0
    "T_in": 300.0, # entre 298.15 e 318.15
}

# monta vetor de entrada X na ordem correta
X_input = np.array([inputs_dict[col] for col in param_cols], dtype=np.float32).reshape(1, -1)

# ================================
# 3) Escalar, prever, desscalar
# ================================
X_scaled = scaler_X.transform(X_input)
Y_scaled = model.predict(X_scaled)
Y = scaler_Y.inverse_transform(Y_scaled)


print("\n=========SAÍDAS==========")

print("Shape de Y:", Y.shape)   # deve ser (1, 157)

# ================================
# 4) Separar saídas
# ================================
# Mesmo esquema do train.py
n_final = 4
n_Cz = 51
n_Tz = 51
n_Qtot = 51

# 4 saídas finais escalares
C_out_final, q_out_final, T_out_final, N_ads_final = Y[0, :n_final]

# 3 vetores de 51
C_z = Y[0, n_final: n_final + n_Cz]                           # 4:55
T_z = Y[0, n_final + n_Cz: n_final + n_Cz + n_Tz]             # 55:106
Qtot_t = Y[0, n_final + n_Cz + n_Tz: n_final + n_Cz + n_Tz + n_Qtot]  # 106:157

print("\nSaídas finais escalares:")
print(f"C_out_final = {C_out_final:.6f}")
print(f"q_out_final = {q_out_final:.6f}")
print(f"T_out_final = {T_out_final:.6f}")
print(f"N_ads_final = {N_ads_final:.6f}")

# ================================
# 5) Plotar perfis (3 vetores)
# ================================
x = np.arange(51)

plt.figure(figsize=(12, 8))

plt.subplot(2, 2, 1)
plt.plot(x, C_z)
plt.title("C_z (perfil ao longo da coluna)")
plt.xlabel("índice")
plt.ylabel("C_z")

plt.subplot(2, 2, 2)
plt.plot(x, T_z)
plt.title("T_z (perfil ao longo da coluna)")
plt.xlabel("índice")
plt.ylabel("T_z")

plt.subplot(2, 2, 3)
plt.plot(x, Qtot_t)
plt.title("Qtot_t (evolução / perfil)")
plt.xlabel("índice")
plt.ylabel("Qtot_t")

# 4ª figura: mostrar as saídas finais escalares em barras
plt.subplot(2, 2, 4)
labels = ["C_out", "q_out", "T_out", "N_ads"]
values = [C_out_final, q_out_final, T_out_final, N_ads_final]
plt.bar(labels, values)
plt.title("Saídas finais (escalares)")

plt.tight_layout()
plt.show()
