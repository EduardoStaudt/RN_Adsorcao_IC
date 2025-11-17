import numpy as np
from pathlib import Path
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
import joblib

# ==== caminhos básicos ====
BASE_DIR = Path(__file__).resolve().parent
CSV_PATH = BASE_DIR.parent / "data" / "processed" / "dataset_FULL.csv"
MODEL_PATH = BASE_DIR.parent / "models" / "best_model.keras"

print("Lendo:", CSV_PATH)
df = pd.read_csv(CSV_PATH)

# ==== mesmas colunas que no train.py ====
param_cols = [
    "L", "Nz", "eps", "rho_B", "u", "D_ax", "kL", "qmax", "b", "n",
    "lam_z", "rho_g", "cp_g", "cp_s", "D_col", "h_w", "T_wall", "dH",
    "dt", "t_end", "C_in", "T_in",
]

out_final_cols = ["C_out_final", "q_out_final", "T_out_final", "N_ads_final"]
all_cols = df.columns
Cz_cols   = [c for c in all_cols if c.startswith("C_z")]
Tz_cols   = [c for c in all_cols if c.startswith("T_z")]
Qtot_cols = [c for c in all_cols if c.startswith("Qtot_t")]
output_cols = out_final_cols + Cz_cols + Tz_cols + Qtot_cols

n_final = len(out_final_cols)
n_Cz    = len(Cz_cols)
n_Tz    = len(Tz_cols)
n_Qtot  = len(Qtot_cols)

print("X cols:", len(param_cols), "Y cols:", len(output_cols))

# ==== dados brutos ====
X_raw = df[param_cols].to_numpy(dtype=np.float32)
Y_raw = df[output_cols].to_numpy(dtype=np.float32)

# ==== carrega scalers salvos ====
scaler_X = joblib.load(BASE_DIR / "scaler_input.save")
scaler_Y = joblib.load(BASE_DIR / "scaler_output.save")

X = scaler_X.transform(X_raw)
Y = scaler_Y.transform(Y_raw)

print("Shapes -> X:", X.shape, "Y:", Y.shape)

# ==== mesma weighted_mse do train.py ====
W_FINAL = 4.0
W_CZ    = 1.0
W_TZ    = 1.0
W_QTOT  = 1.0

def weighted_mse(y_true, y_pred):
    err2 = tf.square(y_true - y_pred)

    e_final = err2[:, :n_final]
    e_Cz    = err2[:, n_final:n_final + n_Cz]
    e_Tz    = err2[:, n_final + n_Cz : n_final + n_Cz + n_Tz]
    e_Qtot  = err2[:, n_final + n_Cz + n_Tz : ]

    mse_final = tf.reduce_mean(e_final)
    mse_Cz    = tf.reduce_mean(e_Cz)
    mse_Tz    = tf.reduce_mean(e_Tz)
    mse_Qtot  = tf.reduce_mean(e_Qtot)

    return (
        W_FINAL * mse_final +
        W_CZ    * mse_Cz +
        W_TZ    * mse_Tz +
        W_QTOT  * mse_Qtot
    )

# ==== carrega o modelo salvo ====
print("Carregando modelo de:", MODEL_PATH)
model = tf.keras.models.load_model(
    MODEL_PATH,
    custom_objects={"weighted_mse": weighted_mse}
)

# ==== previsões ====
Y_pred_norm = model.predict(X, batch_size=512)

# ==== métricas no espaço normalizado ====
rmse_norm = np.sqrt(mean_squared_error(Y, Y_pred_norm))
print(f"\nRMSE Global (Treino, espaço normalizado): {rmse_norm:.6f}")

# ==== desnormalizar ====
Y_real      = scaler_Y.inverse_transform(Y)
Y_pred_real = scaler_Y.inverse_transform(Y_pred_norm)

rmse_real_global = np.sqrt(mean_squared_error(Y_real, Y_pred_real))
print(f"RMSE Global (Treino, espaço físico, todas as saídas): {rmse_real_global:.6f}")

# ==== índices das saídas finais em output_cols ====
idx_C_out = output_cols.index("C_out_final")   # 0
idx_T_out = output_cols.index("T_out_final")   # 2
idx_N_ads = output_cols.index("N_ads_final")   # 3

rmse_C_out = np.sqrt(mean_squared_error(Y_real[:, idx_C_out], Y_pred_real[:, idx_C_out]))
rmse_T_out = np.sqrt(mean_squared_error(Y_real[:, idx_T_out], Y_pred_real[:, idx_T_out]))
rmse_N_ads = np.sqrt(mean_squared_error(Y_real[:, idx_N_ads], Y_pred_real[:, idx_N_ads]))

print("\n============RMSEs em espaço físico (saídas finais): ==========")
print(f"C_out_final  (concentração na saída): {rmse_C_out:.6f}")
print(f"T_out_final  (temperatura na saída):  {rmse_T_out:.6f}")
print(f"N_ads_final  (adsorção total):       {rmse_N_ads:.6f}")
print("=============================================================")


# MELHOR RESULTADO ATÉ AGORA: UTILIZANDO MSE PONDERADA COM PESOS 3,1,1,1
# RMSE Global (Treino, espaço normalizado): 0.086504
# RMSE Global (Treino, espaço físico, todas as saídas): 2.271604

# ============RMSEs em espaço físico (saídas finais): ==========
# C_out_final  (concentração na saída): 0.091858
# T_out_final  (temperatura na saída):  0.962744
# N_ads_final  (adsorção total):       4.595787
# =============================================================

# NOVO MELHOR RESULTADO: UTILIZANDO MSE PONDERADA COM PESOS 4,1,1,1
# RMSE Global (Treino, espaço normalizado): 0.084424
# RMSE Global (Treino, espaço físico, todas as saídas): 2.206466

# ============RMSEs em espaço físico (saídas finais): ==========
# C_out_final  (concentração na saída): 0.089676
# T_out_final  (temperatura na saída):  0.934232
# N_ads_final  (adsorção total):       4.493598
# =============================================================

# RESULTADO ANTERIOR: UTILIZANDO MSE SEM PONDERADA
# RMSE Global (Treino, espaço normalizado): 0.091134
# RMSE Global (Treino, espaço físico, todas as saídas): 2.421087

# =============RMSEs em espaço físico (saídas finais): =========
# C_out_final   (concentração na saída):   0.154512
# T_out_final   (temperatura na saída):    1.160093
# N_ads_final   (adsorção total):          5.501272
# ==================================================================
