import json
import re
import sys
from pathlib import Path

import optuna
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
import joblib
import matplotlib.pyplot as plt

#++++++++++++++++++++++++++++++++++++++++++++#
#          Se não tiver UV baixar            #
# Rodar o UV com o comando paara baixar tudo #
#++++++++++++++++++++++++++++++++++++++++++++#

# REFACTOR:
# - Adicionar o Optuna ✅
# - Analisar o Cross-Validation 
# - adicionar o stick-optimize e hyperoth ❌ nao vou mais fazer todos fazem a mesma coisa mas optuna é bem melhor
# - utilizar e entender metricas no processo de treinamento ✅
# - começar a desencolver a interface em flutter nem que seja só uma tela branca 
# - tirar o Q_tot ao longo do tempo ou leito e tranformar em um unico valor final sendo ele a soma dos q_z ✅

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
print("[DEBUG] OUT_OPTUNA =", cfg.ADS_OUT_OPTUNA)
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
# 5) Modelo
# =======================================================
input_dim = X.shape[1]
output_dim = Y.shape[1]

def build_model(n_layers=3, n_units=352, activation="elu", dropout=0.10, l2_reg=1e-5, lr=5e-4):
    reg = tf.keras.regularizers.l2(l2_reg)
    layers = [tf.keras.layers.Input(shape=(input_dim,))]
    for _ in range(n_layers):
        layers.append(tf.keras.layers.Dense(n_units, activation=activation, kernel_regularizer=reg))
        layers.append(tf.keras.layers.Dropout(dropout))
    layers.append(tf.keras.layers.Dense(output_dim, activation="linear"))
    model = tf.keras.Sequential(layers)
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
# 6) Optuna (substitui keras_tuner — espaço de busca equivalente)
# USE_OPTUNA = True  -> roda busca de HPs com MedianPruner
# USE_OPTUNA = False -> usa arquitetura fixa (build_model padrão)
# =======================================================
USE_OPTUNA = True
use_optuna = USE_OPTUNA

RUN_ID = cfg.now_tag()

if use_optuna:
    print("[INFO] Optuna -> MedianPruner + study")

    N_TRIALS    = 30
    TUNE_EPOCHS = 200
    TUNE_SPLIT  = 0.2

    n_val = int(len(X) * TUNE_SPLIT)
    X_tr, Y_tr = X[n_val:], Y[n_val:]
    X_vl, Y_vl = X[:n_val], Y[:n_val]

    class _PruningCallback(tf.keras.callbacks.Callback):
        """Reporta val_rmse ao Optuna a cada época e poda trials fracos."""
        def __init__(self, trial, monitor="val_rmse"):
            super().__init__()
            self._trial   = trial
            self._monitor = monitor

        def on_epoch_end(self, epoch, logs=None):
            val = (logs or {}).get(self._monitor, float("inf"))
            self._trial.report(val, epoch)
            if self._trial.should_prune():
                raise optuna.TrialPruned()

    def optuna_objective(trial):
        n_layers   = trial.suggest_int("n_layers", 2, 4)
        n_units    = trial.suggest_int("n_units", 64, 256, step=32)
        activation = trial.suggest_categorical("activation", ["relu", "elu"])
        dropout    = trial.suggest_float("dropout", 0.0, 0.2, step=0.05)
        l2_reg     = trial.suggest_float("l2_reg", 1e-6, 1e-3, log=True)
        lr         = trial.suggest_float("lr", 1e-4, 1e-2, log=True)

        m = build_model(n_layers=n_layers, n_units=n_units, activation=activation,
                        dropout=dropout, l2_reg=l2_reg, lr=lr)
        hist = m.fit(
            X_tr, Y_tr,
            validation_data=(X_vl, Y_vl),
            epochs=TUNE_EPOCHS,
            batch_size=512,
            callbacks=[
                _PruningCallback(trial),
                tf.keras.callbacks.EarlyStopping(monitor="val_rmse", patience=15, restore_best_weights=True),
            ],
            verbose=0,
        )
        return min(hist.history.get("val_rmse", [float("inf")]))

    pruner = optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=10)
    _db_uri = (cfg.ADS_OUT_OPTUNA / "optuna_adsorption.db").as_posix()
    storage = optuna.storages.RDBStorage(f"sqlite:///{_db_uri}")
    study = optuna.create_study(
        direction="minimize",
        pruner=pruner,
        storage=storage,
        study_name=f"adsorption_{RUN_ID}",
        load_if_exists=True,
    )
    optuna.logging.set_verbosity(optuna.logging.INFO)
    study.optimize(optuna_objective, n_trials=N_TRIALS, show_progress_bar=True)

    try:
        import optuna.visualization as vis
        vis.plot_optimization_history(study).write_html(
            str(cfg.ADS_OUT_OPTUNA / "opt_history.html"))
        vis.plot_param_importances(study).write_html(
            str(cfg.ADS_OUT_OPTUNA / "param_importance.html"))
        vis.plot_parallel_coordinate(study).write_html(
            str(cfg.ADS_OUT_OPTUNA / "parallel_coord.html"))
        print("[OK] Gráficos Optuna salvos em:", cfg.ADS_OUT_OPTUNA)
    except Exception as e:
        print(f"[AVISO] Não foi possível gerar gráficos Optuna: {e}")

    best = study.best_params
    print("[INFO] Best HP:", best)
    cfg.ADS_BEST_HP.write_text(json.dumps(best, indent=2), encoding="utf-8")
    model = build_model(**best)
else:
    print("[INFO] Sem Optuna -> arquitetura fixa")
    model = build_model()

# =======================================================
# 7) Treino final
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
# 8) Curva na raiz de training
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

#=================================================================================#
#                Porta que roda o dash board http://localhost:8080                #
#=================================================================================#