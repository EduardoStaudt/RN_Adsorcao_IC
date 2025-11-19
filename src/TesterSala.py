from numpy._core.numeric import tensordot
import numpy as np
import pickle
import matplotlib.pyplot as plt
import tensorflow
from tensorflow.keras.models import load_model
import joblib
import pandas as pd
import numpy as np

# ================================
# 1) Carregar modelo e scalers
# ================================
# Pass custom objects to load_model as 'weighted_mse' is a custom loss function.
# The 'weighted_mse' function is defined in a previous cell (uWEhUs7mzmZP).
custom_objects = {
    'weighted_mse': weighted_mse
}
model = load_model("best_model.keras", custom_objects=custom_objects)

with open("scaler_input.save", "rb") as f:
    scaler_X = joblib.load("scaler_input.save")

with open("scaler_output.save", "rb") as f:
    scaler_Y = joblib.load("scaler_output.save")

def weighted_mse(y_true, y_pred):
    """
    MSE ponderado por grupos de saídas:
    - 4 finais (C_out, q_out, T_out, N_ads)
    - C_z*
    - T_z*
    - Qtot_t*
    """

    err2 = tf.square(y_true - y_pred)  # (batch, 157)

    # fatias por grupo
    e_final = err2[:, :n_final]
    e_Cz    = err2[:, n_final:n_final + n_Cz]
    e_Tz    = err2[:, n_final + n_Cz : n_final + n_Cz + n_Tz]
    e_Qtot  = err2[:, n_final + n_Cz + n_Tz : ]

    # MSE de cada grupo (média em batch + saídas)
    mse_final = tf.reduce_mean(e_final)
    mse_Cz    = tf.reduce_mean(e_Cz)
    mse_Tz    = tf.reduce_mean(e_Tz)
    mse_Qtot  = tf.reduce_mean(e_Qtot)

    # Combinação ponderada
    return (
        W_FINAL * mse_final +
        W_CZ    * mse_Cz +
        W_TZ    * mse_Tz +
        W_QTOT  * mse_Qtot
    )

# ======================================================================
# PARAMETROS DE ENTRADA JÁ COM VALLOERERS ATILIZADOS PARA TREINAR A REDE
# ======================================================================
# def sample_params(seed, Nz=51):
#     rng = np.random.default_rng(seed)
#     L      = _randu(rng, 0.05, 1.00)
#     eps    = _randu(rng, 0.20, 0.80)
#     rho_B  = _randu(rng, 50.0, 500.0)
#     u      = _randu(rng, 0.05, 1.00)
#     D_ax   = _randlog(rng, 1e-7, 1e-5)
#     kL     = _randlog(rng, 1e-3, 1e-1)
#     qmax   = _randlog(rng, 0.1, 10.0)
#     b      = _randlog(rng, 0.1, 10.0)
#     n      = _randlog(rng, 0.1, 10.0)
#     lam_z  = _randu(rng, 0.1, 1.0)
#     rho_g  = _randu(rng, 0.2, 2.0)
#     cp_g   = _randu(rng, 800.0, 1200.0)
#     cp_s   = _randu(rng, 800.0, 1200.0)
#     D_col  = _randu(rng, 0.01, 0.50)
#     h_w    = _randu(rng, 10.0, 50.0)
#     T_wall = _randu(rng, 298.15, 318.15)
#     dH     = _randu(rng, 25000.0, 65000.0)
#     t_end  = _randu(rng, 1.0, 500.0)
#     C_in   = _randu(rng, 0.1, 5.0)
#     T_in   = _randu(rng, 298.15, 318.15)
#     dt = t_end/1000.0
#     p = Params(L=L, Nz=Nz, eps=eps, rho_B=rho_B, u=u, D_ax=D_ax, kL=kL,qmax=qmax, b=b, n=n,
#             lam_z=lam_z, rho_g=rho_g, cp_g=cp_g, cp_s=cp_s,
#             D_col=D_col, h_w=h_w, T_wall=T_wall, dH=dH,
#             dt=dt, t_end=t_end)
    
#     return dict(seed=int(seed), Nz=Nz, params=p, C_in=C_in, T_in=T_in)

inputs_dict = {
    "L": 1.0, # L (rng, 0.05, 1.00)
    "Nz": 51, # 51 FIXO
    "eps": 0.35, # eps (rng, 0.20, 0.80)
    "rho_B": 70, # rho_B (rng, 50.0, 500.0)
    "u": 1.5, # u (rng, 0.05, 1.00)
    "D_ax": 1e-6, # D_ax (rng, 1e-7, 1e-5)
    "kL": 0.01, # kL (rng, 1e-3, 1e-1)
    "qmax": 2.5, # qmax (rng, 0.1, 10.0)
    "b": 0.8, # b (rng, 0.1, 10.0)
    "n": 1.2, # n (rng, 0.1, 10.0)
    "lam_z": 0.4, # lam_z (rng, 0.1, 1.0)
    "rho_g": 1.2, # rho_g (rng, 0.2, 2.0)
    "cp_g": 1000, # cp_g (rng, 800.0, 1200.0)
    "cp_s": 800, # cp_s (rng, 800.0, 1200.0)
    "D_col": 0.05, # D_col (rng, 0.01, 0.50)
    "h_w": 10, # h_w (rng, 10.0, 50.0)
    "T_wall": 300, # T_wall (rng, 298.15, 318.15)
    "dH": -35000, # dH (rng, 25000.0, 65000.0)
    "dt": 0.1, # dt = t_end/1000.0
    "t_end": 1000, # t_end (rng, 1.0, 500.0)
    "C_in": 1.0, #   C_in (rng, 0.1, 5.0)
    "T_in": 300, # T_in (rng, 298.15, 318.15)
}

param_cols = [
    "L", "Nz", "eps", "rho_B", "u", "D_ax", "kL", "qmax", "b", "n",
    "lam_z", "rho_g", "cp_g", "cp_s", "D_col", "h_w", "T_wall", "dH",
    "dt", "t_end", "C_in", "T_in",
]

# montar vetor de entrada na ordem correta
X_input = np.array([inputs_dict[col] for col in param_cols]).reshape(1, -1)

# ================================
# 3) Aplicar scaler
# ================================
X_scaled = scaler_X.transform(X_input)

# ================================
# 4) Rodar o modelo
# ================================
Y_scaled = model.predict(X_scaled)

# Reverter o scaling das saídas
Y = scaler_Y.inverse_transform(Y_scaled)

# ================================
# 5) Separar os quatro vetores
# Cada um tem 51 pontos
# ================================
out_final_cols = ["C_out_final", "q_out_final", "T_out_final", "N_ads_final"]

print(len(Y[0]))

C_out = Y[0, 4:55]
q_out = Y[0, 55:106]
T_out = Y[0, 106:157]
N_ads = Y[0, 157:208]

# ================================
# 6) Plotar gráficos
# ================================
x = np.arange(51)

plt.figure(figsize=(14, 10))

plt.subplot(2, 2, 1)
plt.plot(x, C_out)
plt.title("C_out_final")
plt.xlabel("índice")
plt.ylabel("Concentração")

plt.subplot(2, 2, 2)
plt.plot(x, q_out)
plt.title("T_out_final")
plt.xlabel("índice")
plt.ylabel("Temperatura (K)")

plt.subplot(2, 2, 3)
plt.plot(x, T_out)
plt.title("q_out_final")
plt.xlabel("índice")
plt.ylabel("q")


plt.tight_layout()
plt.show()