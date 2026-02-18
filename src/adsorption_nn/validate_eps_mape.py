# -*- coding: utf-8 -*-
"""
Validação / inferência do modelo de ADSORÇÃO (22 -> 208) usando MAPE_EPS (vali.py).

MAPE_EPS:
  mape = mean( abs((y_true - y_pred) / max(abs(y_true), eps)) ) * 100

Salva em:
  outputs/adsorption/inference/<TAG>/eps/

Gera:
- metricas_por_bloco_val.csv
- metricas_finais_individuais.csv
(opcional) predicoes_val_<N>.csv
"""

import sys
import json
import math
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import joblib
import tensorflow as tf

from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


# =======================================================
# Bootstrap para importar config.py
# =======================================================
ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import adsorption_nn.config as cfg
cfg.ensure_dirs()


def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)


def mape_eps(y_true, y_pred, eps=1e-8):
    """
    MAPE com EPS (igual ao vali.py):
      denom = max(|y_true|, eps)
    """
    y_true = np.asarray(y_true, dtype=float).ravel()
    y_pred = np.asarray(y_pred, dtype=float).ravel()
    denom = np.maximum(np.abs(y_true), float(eps))
    return float(np.mean(np.abs((y_true - y_pred) / denom)) * 100.0), int(y_true.size), int(y_true.size)


def compute_metrics_eps(y_true, y_pred, eps=1e-8):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    mse = float(mean_squared_error(y_true, y_pred))
    rmse = float(math.sqrt(mse))
    mae = float(mean_absolute_error(y_true, y_pred))
    r2 = float(r2_score(y_true, y_pred))

    mape_v, n_used, n_total = mape_eps(y_true, y_pred, eps=eps)

    return {
        "mse": mse,
        "rmse": rmse,
        "mae": mae,
        "mape_%": mape_v,
        "r2": r2,
        "mape_used": n_used,
        "mape_total": n_total,
    }


def fmt(x):
    try:
        if np.isnan(x):
            return "NA"
    except Exception:
        pass
    return f"{x:.6g}"


# =======================================================
# CLI
# =======================================================
parser = argparse.ArgumentParser()
parser.add_argument("--model", default=str(cfg.ADS_BEST_MODEL))
parser.add_argument("--dataset_npz", default=str(cfg.ADS_FULL_NPZ))
parser.add_argument("--outdir_base", default=str(cfg.ADS_OUT_INFER))

parser.add_argument("--tag", default=None, help="Ex: 20260216_200000 (se não passar, gera automático)")
parser.add_argument("--seed", type=int, default=42)

parser.add_argument("--eps", type=float, default=1e-8, help="EPS do MAPE (igual vali.py)")
parser.add_argument("--max_samples", type=int, default=20000)
parser.add_argument("--save_pred_n", type=int, default=0, help="0 desliga; se >0 salva CSV com true/pred (N amostras)")
parser.add_argument("--print_finals_n", type=int, default=10, help="Quantas amostras imprimir (true vs pred) dos 4 finais")
args = parser.parse_args()

model_path = Path(args.model)
npz_path = Path(args.dataset_npz)

out_base = Path(args.outdir_base)
ensure_dir(out_base)

tag = args.tag if args.tag else cfg.now_tag()
outdir = out_base / tag / "eps"
ensure_dir(outdir)

# Registrar última execução (base)
latest_file = out_base / "LATEST.txt"
latest_file.write_text(str(out_base / tag), encoding="utf-8")

print("[DEBUG] ROOT        =", cfg.ROOT)
print("[DEBUG] MODEL       =", model_path)
print("[DEBUG] NPZ         =", npz_path)
print("[DEBUG] OUT_BASE    =", out_base)
print("[DEBUG] OUT_RUN_DIR =", outdir)
print("[DEBUG] LATEST.txt  =", latest_file)
print("[DEBUG] METHOD      = EPS (vali.py)")
print("[DEBUG] EPS         =", args.eps)
print("[DEBUG] SEED        =", args.seed)
print("[DEBUG] TAG         =", tag)

if not model_path.exists():
    raise FileNotFoundError(f"Modelo não encontrado: {model_path}")
if not npz_path.exists():
    raise FileNotFoundError(f"Dataset NPZ não encontrado: {npz_path}")
if not cfg.ADS_META.exists():
    raise FileNotFoundError(f"Meta não encontrada: {cfg.ADS_META} (rode o train.py primeiro!)")
if not cfg.ADS_SCALER_IN.exists() or not cfg.ADS_SCALER_OUT.exists():
    raise FileNotFoundError("Scalers não encontrados em models/adsorption (rode o train.py primeiro!)")


# =======================================================
# Ler meta: ordem de colunas
# =======================================================
meta = json.loads(cfg.ADS_META.read_text(encoding="utf-8"))
PARAM_COLS = meta["param_cols"]
FINAL_COLS = meta["final_cols"]
OUTPUT_COLS = meta["output_cols"]
BLOCK_SIZE = int(meta.get("block_size", 51))

if len(PARAM_COLS) != 22:
    raise ValueError(f"Meta diz {len(PARAM_COLS)} entradas, esperado 22.")
if len(OUTPUT_COLS) != 208:
    raise ValueError(f"Meta diz {len(OUTPUT_COLS)} saídas, esperado 208.")


# =======================================================
# Carregar modelo/scalers
# =======================================================
model = tf.keras.models.load_model(str(model_path), compile=False)
scaler_in = joblib.load(cfg.ADS_SCALER_IN)
scaler_out = joblib.load(cfg.ADS_SCALER_OUT)

pred_dim = int(model.output_shape[-1])
if pred_dim != 208:
    raise ValueError(
        f"Seu modelo atual retorna {pred_dim} saídas, mas a validação espera 208.\n"
        f"Provável modelo antigo em models/adsorption. Rode train.py para sobrescrever."
    )


# =======================================================
# Carregar NPZ -> DataFrame (chaves: data e columns)
# =======================================================
data = np.load(str(npz_path), allow_pickle=True)
if "data" not in data.files or "columns" not in data.files:
    raise ValueError("Esse validate_eps_mape.py espera NPZ com chaves: data e columns.")

mat = data["data"]
cols = [str(c) for c in data["columns"].tolist()]
df = pd.DataFrame(mat, columns=cols)

missing_x = [c for c in PARAM_COLS if c not in df.columns]
missing_y = [c for c in OUTPUT_COLS if c not in df.columns]
if missing_x:
    raise ValueError(f"Faltam colunas X no NPZ: {missing_x[:10]} ...")
if missing_y:
    raise ValueError(f"Faltam colunas Y no NPZ: {missing_y[:10]} ...")

X_raw = df[PARAM_COLS].to_numpy(dtype=np.float32)
y_true = df[OUTPUT_COLS].to_numpy(dtype=np.float32)

# limitar amostras (com seed)
rng = np.random.default_rng(args.seed)
if X_raw.shape[0] > args.max_samples:
    idx = rng.choice(X_raw.shape[0], size=args.max_samples, replace=False)
    X_raw = X_raw[idx]
    y_true = y_true[idx]


# =======================================================
# Predição (norm -> desnorm)
# =======================================================
Xn = scaler_in.transform(X_raw)
y_pred_norm = model.predict(Xn, verbose=0)
y_pred = scaler_out.inverse_transform(y_pred_norm)


# =======================================================
# Métricas globais
# =======================================================
glob = compute_metrics_eps(y_true, y_pred, eps=args.eps)
print(
    f"[GLOBAL][EPS] RMSE={fmt(glob['rmse'])}  MAE={fmt(glob['mae'])}  "
    f"MAPE_eps={fmt(glob['mape_%'])}%  R2={fmt(glob['r2'])}"
)


# =======================================================
# Métricas por blocos
# finais: 0..3
# C_z:    4..54
# q_z:    55..105
# T_z:    106..156
# Qtot_t: 157..207
# =======================================================
def sl(a, b):
    return slice(a, b)

S_FINAL = sl(0, 4)
S_CZ    = sl(4, 4 + BLOCK_SIZE)
S_QZ    = sl(4 + BLOCK_SIZE, 4 + 2 * BLOCK_SIZE)
S_TZ    = sl(4 + 2 * BLOCK_SIZE, 4 + 3 * BLOCK_SIZE)
S_QTOT  = sl(4 + 3 * BLOCK_SIZE, 4 + 4 * BLOCK_SIZE)

rows = []
for name, s in [
    ("Finais(4)", S_FINAL),
    ("C_z(51)", S_CZ),
    ("q_z(51)", S_QZ),
    ("T_z(51)", S_TZ),
    ("Qtot_t(51)", S_QTOT),
]:
    m = compute_metrics_eps(y_true[:, s], y_pred[:, s], eps=args.eps)
    rows.append([name, m["mse"], m["rmse"], m["mae"], m["mape_%"], m["r2"], m["mape_used"], m["mape_total"]])

df_blocks = pd.DataFrame(
    rows,
    columns=["bloco", "mse", "rmse", "mae", "mape_%", "r2", "mape_used", "mape_total"]
)

blocks_path = outdir / "metricas_por_bloco_val.csv"
df_blocks.to_csv(blocks_path, index=False, encoding="utf-8")

print("\n================= MÉTRICAS POR BLOCO (EPS) =================")
print(df_blocks.to_string(index=False))
print("=============================================================")
print("[OK] Métricas por bloco salvas em:", blocks_path)


# =======================================================
# Métricas individuais dos FINAIS (4 variáveis)
# =======================================================
rows_f = []
for j, name in enumerate(FINAL_COLS):
    yt = y_true[:, j]
    yp = y_pred[:, j]
    m = compute_metrics_eps(yt, yp, eps=args.eps)
    rows_f.append([name, m["mse"], m["rmse"], m["mae"], m["mape_%"], m["r2"], m["mape_used"], m["mape_total"]])

df_finals = pd.DataFrame(
    rows_f,
    columns=["variavel", "mse", "rmse", "mae", "mape_%", "r2", "mape_used", "mape_total"]
)

finals_path = outdir / "metricas_finais_individuais.csv"
df_finals.to_csv(finals_path, index=False, encoding="utf-8")

print("\n============== MÉTRICAS FINAIS (INDIVIDUAIS) [EPS] ==============")
print(df_finals.to_string(index=False))
print("=================================================================")
print("[OK] Métricas finais individuais salvas em:", finals_path)


# =======================================================
# Imprimir finais (true vs pred)
# =======================================================
print("\n================== FINAIS (true vs pred) [EPS] ==================")
k = min(int(args.print_finals_n), y_true.shape[0])
idxs = rng.choice(y_true.shape[0], size=k, replace=False)

for idx in idxs:
    tfinal = y_true[idx, 0:4]
    pfinal = y_pred[idx, 0:4]
    print(f"Amostra idx={int(idx)}")
    for j, name in enumerate(FINAL_COLS):
        print(f"  {name}: true={tfinal[j]:.6g}  pred={pfinal[j]:.6g}")
    print("-" * 60)
print("===============================================================\n")


# =======================================================
# Salvar predições (opcional)
# =======================================================
if args.save_pred_n and args.save_pred_n > 0:
    n_save = min(int(args.save_pred_n), y_true.shape[0])
    idx_save = np.arange(n_save)

    pred_df = pd.DataFrame(
        np.hstack([y_true[idx_save], y_pred[idx_save]]),
        columns=[f"true_{c}" for c in OUTPUT_COLS] + [f"pred_{c}" for c in OUTPUT_COLS]
    )
    pred_path = outdir / f"predicoes_val_{n_save}.csv"
    pred_df.to_csv(pred_path, index=False, encoding="utf-8")
    print("[OK] Predições salvas em:", pred_path)
