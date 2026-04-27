import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
import joblib
import matplotlib.pyplot as plt


# REFACTOR:
# - Adicionar o Optuna
# - Analisar o Cross- Validation 
# - adicionar o stick-optimize e hyperoth
# - utilizar e entender metricas no processo de treinamento
# - começar a desencolver a interface em flutter nem que seja só uma tela branca
# - tirar o Q_tot ao longo do tempo ou leito e tranformar em um unico valor final sendo ele a soma dos q_z


# TODO: PARA TREINAR SÓ O O MODELO FINAL É SÓ RODAR COM O  USE_TUNER = False (ta na linha 177)

# -------------------------
# Bootstrap import config
# -------------------------
ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import adsorption_nn.config as cfg
cfg.ensure_dirs()

print("[DEBUG] ROOT =", cfg.ROOT)
print("[DEBUG] CSV  =", cfg.ADS_FULL_CSV)
print("[DEBUG] OUT_TRAIN =", cfg.ADS_OUT_TRAIN)
print("[DEBUG] OUT_TUNER =", cfg.ADS_OUT_TUNER)
print("[DEBUG] MODELS =", cfg.ADS_MODELS_DIR)

if not cfg.ADS_FULL_CSV.exists():
    raise FileNotFoundError(f"[ERRO] Dataset não encontrado: {cfg.ADS_FULL_CSV}")

# =======================================================
# 1) Ler dataset
# =======================================================
print("[INFO] Lendo:", cfg.ADS_FULL_CSV)
df = pd.read_csv(cfg.ADS_FULL_CSV)
print("[INFO] Dataset carregado:", df.shape)

# =======================================================
# 2) Definir colunas (X e Y)
# =======================================================
PARAM_COLS = [
    "L", "Nz", "eps", "rho_B", "u", "D_ax", "kL", "qmax", "b", "n",
    "lam_z", "rho_g", "cp_g", "cp_s", "D_col", "h_w", "T_wall", "dH",
    "dt", "t_end", "C_in", "T_in"
]
FINAL_COLS = ["C_out_final", "q_out_final", "T_out_final", "N_ads_final"]

BLOCK_SIZE = 51

def cols_by_prefix(all_cols, pref: str):
    cols = [c for c in all_cols if c.startswith(pref)]
    def key(cname: str):
        m = re.search(r"(\d+)$", cname)
        return int(m.group(1)) if m else 10**9
    return sorted(cols, key=key)

all_cols = df.columns.tolist()

missing_x = [c for c in PARAM_COLS if c not in all_cols]
missing_f = [c for c in FINAL_COLS if c not in all_cols]
if missing_x:
    raise ValueError(f"[ERRO] Faltam colunas de entrada no CSV: {missing_x}")
if missing_f:
    raise ValueError(f"[ERRO] Faltam colunas finais no CSV: {missing_f}")

Cz_cols   = cols_by_prefix(all_cols, "C_z")
qz_cols   = cols_by_prefix(all_cols, "q_z")
Tz_cols   = cols_by_prefix(all_cols, "T_z")
#Qtot_cols = cols_by_prefix(all_cols, "Qtot_t") nao precisa mais

for name, cols in [("C_z", Cz_cols), ("q_z", qz_cols), ("T_z", Tz_cols)]:
    # tirei o Qtot_cols
    if len(cols) != BLOCK_SIZE:
        raise ValueError(f"[ERRO] Esperava {BLOCK_SIZE} colunas para {name}, mas achei {len(cols)}.")

# cria a colola Qtot_final sendo ela a soma dos q_z
# df["Qtot_final"] = df[qz_cols].sum(axis=1) # pega a soma de todas as linhas axis=1

PROFILE_COLS = Cz_cols + qz_cols + Tz_cols
OUTPUT_COLS = FINAL_COLS + PROFILE_COLS

print("\n============= Dimensões esperadas =============")
print("Entradas (X):", len(PARAM_COLS))
print("Saídas (Y):  ", len(OUTPUT_COLS), "(4 finais + 153 perfis)")
print("================================================\n")

# =======================================================
# 3) Montar X e Y
# =======================================================
X_raw = df[PARAM_COLS].to_numpy(dtype=np.float32)
Y_raw = df[OUTPUT_COLS].to_numpy(dtype=np.float32)
print("[INFO] X_raw:", X_raw.shape)
print("[INFO] Y_raw:", Y_raw.shape)

# =======================================================
# 4) Normalização Z-score
# =======================================================
scaler_X = StandardScaler().fit(X_raw)
scaler_Y = StandardScaler().fit(Y_raw)

X = scaler_X.transform(X_raw).astype(np.float32)
Y = scaler_Y.transform(Y_raw).astype(np.float32)

joblib.dump(scaler_X, cfg.ADS_SCALER_IN)
joblib.dump(scaler_Y, cfg.ADS_SCALER_OUT)

meta = {
    "param_cols": PARAM_COLS,
    "final_cols": FINAL_COLS,
    "profile_cols": PROFILE_COLS,
    "output_cols": OUTPUT_COLS,
    "block_size": BLOCK_SIZE,
    "dataset_csv": str(cfg.ADS_FULL_CSV),
    "x_dim": int(X.shape[1]),
    "y_dim": int(Y.shape[1]),
}
cfg.ADS_META.write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")

print("[OK] Salvei scalers e meta em:", cfg.ADS_MODELS_DIR)

# =======================================================
# 5) Loss ponderada por blocos ########################ALTERAR####################
# =======================================================
# W_FINAL = 1.0
# W_CZ = 1.0
# W_QZ = 1.0
# W_TZ = 1.0
# W_QTOT = 1.0

# def weighted_mse(y_true, y_pred):
#     err2 = tf.square(y_true - y_pred)
#     e_final = err2[:, 0:4]
#     e_Cz    = err2[:, 4:4 + BLOCK_SIZE]
#     e_qz    = err2[:, 4 + BLOCK_SIZE:4 + 2 * BLOCK_SIZE]
#     e_Tz    = err2[:, 4 + 2 * BLOCK_SIZE:4 + 3 * BLOCK_SIZE]
#     e_Qtot = err2[:, 4 + 3 * BLOCK_SIZE :4 + 3 * BLOCK_SIZE + 1]
#     return (
#         W_FINAL * tf.reduce_mean(e_final) +
#         W_CZ    * tf.reduce_mean(e_Cz) +
#         W_QZ    * tf.reduce_mean(e_qz) +
#         W_TZ    * tf.reduce_mean(e_Tz) +
#         W_QTOT  * tf.reduce_mean(e_Qtot)
#     )

# =======================================================
# 6) Modelo
# =======================================================
input_dim = X.shape[1]
output_dim = Y.shape[1]

def build_model(n1=352, n2=352, n3=176, dropout=0.10, l2_reg=1e-5, lr=5e-4):
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(input_dim,)),
        tf.keras.layers.Dense(n1, activation="elu", kernel_regularizer=tf.keras.regularizers.l2(l2_reg)),
        tf.keras.layers.Dropout(dropout),
        tf.keras.layers.Dense(n2, activation="elu", kernel_regularizer=tf.keras.regularizers.l2(l2_reg)),
        tf.keras.layers.Dropout(dropout),
        tf.keras.layers.Dense(n3, activation="elu", kernel_regularizer=tf.keras.regularizers.l2(l2_reg)),
        tf.keras.layers.Dense(output_dim, activation="linear"),
    ])
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
        loss="mse",
        metrics=[
            tf.keras.metrics.MeanAbsoluteError(name="mae"),
            tf.keras.metrics.RootMeanSquaredError(name="rmse"),
        ],
    )
    return model

# =======================================================
# 7) Keras Tuner (salvando em outputs/adsorption/training/tuner)
# Observação: para evitar erro de deletar pasta no OneDrive,
# usamos project_name único e overwrite=False.
# =======================================================
USE_TUNER = True
use_tuner = False
kt = None
if USE_TUNER:
    try:
        import keras_tuner as kt
        use_tuner = True
    except Exception:
        use_tuner = False

RUN_ID = cfg.now_tag()

if use_tuner:
    print("[INFO] keras_tuner -> BayesianOptimization")
    print("[INFO] Logs do tuner em:", cfg.ADS_OUT_TUNER)

    def HyperModel(hp):
        n1 = hp.Int("n1", 88, 528, step=44)
        n2 = hp.Int("n2", 176, 528, step=44)
        n3 = hp.Int("n3", 88, 264, step=22)
        dropout = hp.Float("dropout", 0.0, 0.30, step=0.05)
        l2_reg = hp.Choice("l2_reg", values=[1e-6, 1e-5, 1e-4])
        lr = hp.Choice("lr", values=[1e-3, 5e-4, 1e-4])
        return build_model(n1=n1, n2=n2, n3=n3, dropout=dropout, l2_reg=l2_reg, lr=lr)

    tuner = kt.BayesianOptimization(
        HyperModel,
        objective=kt.Objective("val_rmse", direction="min"),
        max_trials=30,
        overwrite=False,                         # evita tentar apagar pastas
        directory=str(cfg.ADS_OUT_TUNER),
        project_name=f"adsorption_{RUN_ID}",
    )

    tuner.search(
        X, Y,
        validation_split=0.2,
        epochs=200,
        batch_size=512,
        verbose=1
    )

    best_hp = tuner.get_best_hyperparameters(1)[0]
    print("[INFO] Best HP:", best_hp.values)
    cfg.ADS_BEST_HP.write_text(json.dumps(best_hp.values, indent=2), encoding="utf-8")
    model = tuner.hypermodel.build(best_hp)
else:
    print("[INFO] Sem tuner -> arquitetura fixa")
    model = build_model()

# =======================================================
# 8) Treino final
# =======================================================
callbacks = [
    tf.keras.callbacks.ReduceLROnPlateau(monitor="val_rmse", factor=0.5, patience=12, verbose=1),
    tf.keras.callbacks.EarlyStopping(monitor="val_rmse", patience=30, restore_best_weights=True),
    tf.keras.callbacks.ModelCheckpoint(
        filepath=str(cfg.ADS_BEST_MODEL),
        monitor="val_rmse",
        save_best_only=True,
        save_weights_only=False,
    ),
]

print("[INFO] Treinando modelo final...")
history = model.fit(
    X, Y,
    validation_split=0.1,
    epochs=500,
    batch_size=512,
    callbacks=callbacks,
    verbose=1
)

print("[OK] Modelo salvo em:", cfg.ADS_BEST_MODEL)

# =======================================================
# 9) Curva na raiz de training
# =======================================================
fig = plt.figure(figsize=(8, 5))
plt.plot(history.history.get("rmse", []), label="rmse")
plt.plot(history.history.get("val_rmse", []), label="val_rmse")
plt.xlabel("Época")
plt.ylabel("RMSE (normalizado)")
plt.legend()
plt.grid(True)

plt.savefig(cfg.ADS_CURVE_PATH, dpi=300, bbox_inches="tight")
plt.close(fig)
print("[OK] Curva salva em:", cfg.ADS_CURVE_PATH)

