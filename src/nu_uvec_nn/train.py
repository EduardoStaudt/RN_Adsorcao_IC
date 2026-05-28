# -*- coding: utf-8 -*-
"""Treino do modelo NU -> U_vec (1 -> 25).

Versão organizada do treino:
- split train/val/test (80/10/10)
- normalização por mean/std (StandardScaler)
- salva modelo + scalers + meta + NPZ de split (estilo vali.py)

Arquivos gerados:
models/nu_uvec/
  - best_model.keras
  - scaler_X.pkl
  - scaler_Y.pkl
  - model_meta.json
  - dataset_split.npz   (X_* e y_* NORMALIZADOS + mean/std)

outputs/nu_uvec/training/
  - curva_treinamento.png
"""

from __future__ import annotations

import sys
import json
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib


# =======================================================
# Bootstrap para importar config.py
# =======================================================
ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import nu_uvec_nn.config as cfg  # wrapper do config central
cfg.ensure_dirs()


# =======================================================
# Colunas esperadas
# =======================================================
X_COLS = ["nu"]
Y_COLS = [f"u_{k}" for k in range(25)]


# =======================================================
# Modelo (duas opções: "simple" replica seu melhor antigo)
# =======================================================
def build_model(arch: str, lr: float, l2: float = 0.0) -> tf.keras.Model:
    reg = tf.keras.regularizers.l2(l2) if l2 and l2 > 0 else None

    if arch == "simple":
        # Seu modelo "curto" antigo:
        # Dense(1, relu) -> Dense(25, linear)
        model = tf.keras.Sequential([
            tf.keras.layers.Input(shape=(1,)),
            tf.keras.layers.Dense(1, activation="relu", use_bias=True, kernel_regularizer=reg),
            tf.keras.layers.Dense(25, activation="linear", use_bias=True),
        ])
    elif arch == "mlp":
        # Um MLP um pouco mais flexível (às vezes melhora)
        model = tf.keras.Sequential([
            tf.keras.layers.Input(shape=(1,)),
            tf.keras.layers.Dense(64, activation="swish", kernel_initializer="he_uniform", kernel_regularizer=reg),
            tf.keras.layers.Dense(64, activation="swish", kernel_initializer="he_uniform", kernel_regularizer=reg),
            tf.keras.layers.Dense(25, activation="linear"),
        ])
    else:
        raise ValueError("arch inválida. Use: simple | mlp")

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
        loss="mse",
        metrics=[
            tf.keras.metrics.MeanAbsoluteError(name="mae"),
            tf.keras.metrics.RootMeanSquaredError(name="rmse"),
        ],
    )
    return model


def save_split_npz(
    path: Path,
    X_train_n: np.ndarray,
    X_val_n: np.ndarray,
    X_test_n: np.ndarray,
    y_train_n: np.ndarray,
    y_val_n: np.ndarray,
    y_test_n: np.ndarray,
    X_mean: np.ndarray,
    X_std: np.ndarray,
    y_mean: np.ndarray,
    y_std: np.ndarray,
) -> None:
    """Salva NPZ compatível com o "método" do vali.py.

    - X_* e y_* já normalizados
    - mean/std para desnormalizar
    """
    np.savez_compressed(
        path,
        X_train=X_train_n.astype("float32"),
        X_val=X_val_n.astype("float32"),
        X_test=X_test_n.astype("float32"),
        y_train=y_train_n.astype("float32"),
        y_val=y_val_n.astype("float32"),
        y_test=y_test_n.astype("float32"),
        X_mean=X_mean.astype("float32"),
        X_std=X_std.astype("float32"),
        y_mean=y_mean.astype("float32"),
        y_std=y_std.astype("float32"),
        input_cols=np.array(X_COLS, dtype=object),
        output_cols=np.array(Y_COLS, dtype=object),
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=2000)
    parser.add_argument("--batch", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--arch", type=str, default="simple", choices=["simple", "mlp"])
    parser.add_argument("--l2", type=float, default=0.0)
    parser.add_argument("--csv", type=str, default=str(cfg.NU_ANALITICO_CSV))
    args = parser.parse_args()

    np.random.seed(args.seed)
    tf.random.set_seed(args.seed)

    csv_path = Path(args.csv)
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV não encontrado: {csv_path}")

    print(f"[INFO] Lendo: {csv_path}")
    df = pd.read_csv(csv_path)

    # Checagem rápida
    missing_x = [c for c in X_COLS if c not in df.columns]
    missing_y = [c for c in Y_COLS if c not in df.columns]
    if missing_x or missing_y:
        raise ValueError(
            "CSV não tem as colunas esperadas.\n"
            f"Faltando X: {missing_x}\n"
            f"Faltando Y: {missing_y}"
        )

    X = df[X_COLS].to_numpy(dtype="float32")  # (N, 1)
    y = df[Y_COLS].to_numpy(dtype="float32")  # (N, 25)

    print(f"[INFO] X: {X.shape} | y: {y.shape}")

    # Split 80/10/10
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.2, random_state=args.seed, shuffle=True
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, random_state=args.seed, shuffle=True
    )
    print(f"[INFO] Split: train={X_train.shape[0]} val={X_val.shape[0]} test={X_test.shape[0]}")

    # Normalização mean/std (só no treino)
    scaler_x = StandardScaler()
    scaler_y = StandardScaler()

    X_train_n = scaler_x.fit_transform(X_train)
    X_val_n = scaler_x.transform(X_val)
    X_test_n = scaler_x.transform(X_test)

    y_train_n = scaler_y.fit_transform(y_train)
    y_val_n = scaler_y.transform(y_val)
    y_test_n = scaler_y.transform(y_test)

    # Salva scalers
    joblib.dump(scaler_x, cfg.NU_SCALER_X)
    joblib.dump(scaler_y, cfg.NU_SCALER_Y)
    print(f"[OK] Salvei scalers em: {cfg.NU_MODELS_DIR}")

    # Salva meta
    meta = {
        "task": "nu_uvec",
        "x_cols": X_COLS,
        "y_cols": Y_COLS,
        "x_dim": int(X_train_n.shape[1]),
        "y_dim": int(y_train_n.shape[1]),
        "norm": "standard_scaler",
        "seed": int(args.seed),
        "arch": args.arch,
    }
    cfg.NU_META.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"[OK] Salvei meta: {cfg.NU_META}")

    # Salva split NPZ (estilo vali.py)
    X_mean = scaler_x.mean_.reshape(1, -1)
    X_std = scaler_x.scale_.reshape(1, -1)
    y_mean = scaler_y.mean_.reshape(1, -1)
    y_std = scaler_y.scale_.reshape(1, -1)

    save_split_npz(
        cfg.NU_SPLIT_NPZ,
        X_train_n, X_val_n, X_test_n,
        y_train_n, y_val_n, y_test_n,
        X_mean, X_std, y_mean, y_std,
    )
    print(f"[OK] Salvei split NPZ: {cfg.NU_SPLIT_NPZ}")

    # Modelo
    model = build_model(arch=args.arch, lr=args.lr, l2=args.l2)
    model.summary()

    # Callbacks
    cbs = [
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.5, patience=20, min_lr=1e-6, verbose=1
        ),
        tf.keras.callbacks.ModelCheckpoint(
            filepath=str(cfg.NU_BEST_MODEL), monitor="val_loss", save_best_only=True, verbose=1
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=80, restore_best_weights=True, verbose=1
        ),
    ]

    hist = model.fit(
        X_train_n, y_train_n,
        validation_data=(X_val_n, y_val_n),
        epochs=args.epochs,
        batch_size=args.batch,
        callbacks=cbs,
        verbose=1,
    )

    # Salva curva de treino
    cfg.NU_OUT_TRAIN.mkdir(parents=True, exist_ok=True)
    plt.figure()
    plt.plot(hist.history.get("loss", []), label="train")
    plt.plot(hist.history.get("val_loss", []), label="val")
    plt.xlabel("Epoch")
    plt.ylabel("MSE (normalizado)")
    plt.title("Curva de Treino (NU -> U_vec)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(cfg.NU_CURVE_PATH, dpi=160)
    plt.close()
    print(f"[OK] Curva salva em: {cfg.NU_CURVE_PATH}")

    print(f"[OK] Melhor modelo salvo em: {cfg.NU_BEST_MODEL}")


if __name__ == "__main__":
    main()



# MELHOR MODELO DE TODOS OS TESTADOS
    # tf.keras.layers.Dense(1, activation="relu", input_shape=(1,), use_bias=True),
    # tf.keras.layers.Dense(25, activation="linear", use_bias=True)  
#com 5000 epochs
    # Acurácia por célula (tolerância): 99.60%
    # Acurácia por amostra (≥90% células ok): 99.50%
#com 2000 epochs
    # Acurácia por célula (tolerância): 84.50%
    # Acurácia por amostra (≥90% células ok): 67.20%
#com 10000 epochs
    # Acurácia por célula (tolerância): 100.00%
    # Acurácia por amostra (≥90% células ok): 100.00%

#RESULTADOS APÓS EDIÇÕES

# TOL_ABS = 0.00001   
# TOL_REL = 0.00005  
#CAMADA DUPLA
# Acurácia por célula (tolerância): 80.62%
# Acurácia por amostra (≥90% células ok): 71.50%
# CAMADA SIMPLES
# Acurácia por célula (tolerância): 71.60%
# Acurácia por amostra (≥90% células ok): 0.00%

# NOVO MODELO COM 3 CAMADAS 
# TOL_ABS = 0.00001   
# TOL_REL = 0.00005  
# SWISH:
# Acurácia por célula (tolerância): 20.52%
# Acurácia por amostra (≥90% células ok): 0.00%
# TANH:
# Acurácia por célula (tolerância): 20.66%
# Acurácia por amostra (≥90% células ok): 0.00%
# SWISH AND TANH:
# Acurácia por célula (tolerância): 20.78%
# Acurácia por amostra (≥90% células ok): 0.00%




























# PLOT CURVA DE TREINO()
# plt.figure()
# plt.plot(history.history["loss"], label="train")
# plt.plot(history.history["val_loss"], label="val")
# plt.xlabel("Epoch"); plt.ylabel("MSE (norm)"); plt.legend(); plt.title("Curva de treino")
# plt.tight_layout(); plt.show()

# PLOT CURVA DE TREINO()
# plt.figure()
# plt.plot(history.history["loss"], label="train")
# plt.plot(history.history["val_loss"], label="val")
# plt.xlabel("Epoch"); plt.ylabel("MSE (norm)"); plt.legend(); plt.title("Curva de treino")
# plt.tight_layout(); plt.show()

# PLOTAGEM DOS MAPAS 5x5()
# mapas 5x5
# t5 = u_true.reshape(5,5)
# p5 = u_pred.reshape(5,5)
# e5 = (p5 - t5)

# fig, axs = plt.subplots(1,3, figsize=(10,3))
# im0 = axs[0].imshow(t5, origin="lower"); axs[0].set_title("True");  plt.colorbar(im0, ax=axs[0])
# im1 = axs[1].imshow(p5, origin="lower"); axs[1].set_title("Pred");  plt.colorbar(im1, ax=axs[1])
# im2 = axs[2].imshow(e5, origin="lower", cmap="seismic"); axs[2].set_title("Error"); plt.colorbar(im2, ax=axs[2])
# plt.suptitle(f"nu={nu_val:.6f} — True vs Pred vs Error")
# plt.tight_layout(); plt.show()