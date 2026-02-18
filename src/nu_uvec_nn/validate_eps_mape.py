# -*- coding: utf-8 -*-
"""Validação NU -> U_vec usando o "método do vali.py" (MAPE com eps)."""

from __future__ import annotations

import sys
import argparse
from pathlib import Path
import math
import json

import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt

from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import nu_uvec_nn.config as cfg
cfg.ensure_dirs()


def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def mape_eps(y_true: np.ndarray, y_pred: np.ndarray, eps: float = 1e-8) -> float:
    denom = np.maximum(np.abs(y_true), eps)
    return float(np.mean(np.abs((y_true - y_pred) / denom)) * 100.0)


def acc_from_mape(mape_value: float) -> float:
    return float(max(0.0, 100.0 - mape_value))


def _pick_split(data: np.lib.npyio.NpzFile, split: str) -> tuple[np.ndarray, np.ndarray]:
    if split == "train":
        return data["X_train"], data["y_train"]
    if split == "val":
        return data["X_val"], data["y_val"]
    if split == "test":
        return data["X_test"], data["y_test"]
    raise ValueError("split inválido: use train|val|test")


def run_validation(
    tag: str,
    split: str,
    model_path: Path,
    dataset_npz: Path,
    outdir: Path,
    eps: float,
    n_show: int,
    seed: int,
) -> dict:
    ensure_dir(outdir)

    print(f"[DEBUG] MODEL  = {model_path}")
    print(f"[DEBUG] NPZ    = {dataset_npz}")
    print(f"[DEBUG] OUTDIR = {outdir}")
    print(f"[INFO] split={split} | eps={eps}")

    if not model_path.exists():
        raise FileNotFoundError(f"Modelo não encontrado: {model_path}")
    if not dataset_npz.exists():
        raise FileNotFoundError(f"NPZ não encontrado: {dataset_npz}")

    model = tf.keras.models.load_model(model_path)

    data = np.load(dataset_npz, allow_pickle=True)
    Xn, yn = _pick_split(data, split)

    y_mean = data["y_mean"]
    y_std = data["y_std"]

    y_true = yn * y_std + y_mean
    y_pred_n = model.predict(Xn, verbose=0)
    y_pred = y_pred_n * y_std + y_mean

    mse = float(mean_squared_error(y_true, y_pred))
    rmse = float(math.sqrt(mse))
    mae = float(mean_absolute_error(y_true, y_pred))
    r2 = float(r2_score(y_true, y_pred))
    mape = mape_eps(y_true, y_pred, eps=eps)
    acc = acc_from_mape(mape)

    metrics_global = {
        "method": "vali_eps",
        "split": split,
        "mse": mse,
        "rmse": rmse,
        "mae": mae,
        "r2": r2,
        "mape_%": mape,
        "acc_%": acc,
        "mape_used": int(np.prod(y_true.shape)),
        "mape_total": int(np.prod(y_true.shape)),
    }

    print("\n================= MÉTRICAS GLOBAIS (vali_eps) =================")
    print(f"MSE={mse:.6g} RMSE={rmse:.6g} MAE={mae:.6g} R2={r2:.6g} MAPE={mape:.3f}% ACC={acc:.2f}%")
    print("=============================================================\n")

    out_cols = data["output_cols"].astype(str).tolist() if "output_cols" in data.files else [f"u_{k}" for k in range(y_true.shape[1])]
    rows = []
    for j, name in enumerate(out_cols):
        yt = y_true[:, j]
        yp = y_pred[:, j]
        mse_j = float(mean_squared_error(yt, yp))
        rmse_j = float(math.sqrt(mse_j))
        mae_j = float(mean_absolute_error(yt, yp))
        r2_j = float(r2_score(yt, yp))
        mape_j = mape_eps(yt, yp, eps=eps)
        acc_j = acc_from_mape(mape_j)
        rows.append({"variavel": name, "mse": mse_j, "rmse": rmse_j, "mae": mae_j, "r2": r2_j, "mape_%": mape_j, "acc_%": acc_j})
    df_out = pd.DataFrame(rows)

    pd.DataFrame([metrics_global]).to_csv(outdir / "metricas_global.csv", index=False)
    df_out.to_csv(outdir / "metricas_por_saida.csv", index=False)

    rng = np.random.default_rng(seed)
    n = y_true.shape[0]
    k = min(1000, n)
    idx = rng.choice(n, size=k, replace=False)
    df_pred = pd.DataFrame(y_true[idx], columns=[f"true_{c}" for c in out_cols])
    df_pred2 = pd.DataFrame(y_pred[idx], columns=[f"pred_{c}" for c in out_cols])
    pd.concat([df_pred.reset_index(drop=True), df_pred2.reset_index(drop=True)], axis=1).to_csv(outdir / f"predicoes_{split}_{k}.csv", index=False)

    mae_per_sample = np.mean(np.abs(y_pred - y_true), axis=1)
    worst_idx = np.argsort(mae_per_sample)[-n_show:]
    best_idx = np.argsort(mae_per_sample)[:n_show]
    show_idx = [0] + list(worst_idx[::-1]) + list(best_idx)

    def plot_sample(i: int) -> None:
        yt = y_true[i]
        yp = y_pred[i]
        plt.figure()
        plt.plot(yt, label="true")
        plt.plot(yp, label="pred")
        plt.title(f"U_vec | amostra={i} | split={split}")
        plt.xlabel("k (0..24)")
        plt.ylabel("u_k")
        plt.legend()
        plt.tight_layout()
        plt.savefig(outdir / f"comp_uvec_amostra_{i}.png", dpi=160)
        plt.close()

    print("\n================= AMOSTRAS (true vs pred) =================")
    for i in show_idx:
        if 0 <= int(i) < n:
            print(f"Amostra {int(i)} | MAE_amostra={mae_per_sample[int(i)]:.6g}")
            plot_sample(int(i))
    print("===========================================================\n")

    (cfg.NU_OUT_INFER / "LATEST.txt").write_text(str(outdir), encoding="utf-8")
    (outdir / "run_meta.json").write_text(json.dumps({
        "tag": tag, "method": "vali_eps", "split": split,
        "model": str(model_path), "dataset_npz": str(dataset_npz), "eps": eps,
    }, indent=2), encoding="utf-8")

    return metrics_global


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--model", type=str, default=str(cfg.NU_BEST_MODEL))
    p.add_argument("--dataset_npz", type=str, default=str(cfg.NU_SPLIT_NPZ))
    p.add_argument("--split", type=str, default="test", choices=["train", "val", "test"])
    p.add_argument("--eps", type=float, default=1e-8)
    p.add_argument("--tag", type=str, default="")
    p.add_argument("--n_show", type=int, default=3)
    p.add_argument("--seed", type=int, default=42)
    a = p.parse_args()

    tag = a.tag.strip() or cfg.now_tag()
    outdir = cfg.NU_OUT_INFER / tag / "vali_eps"

    run_validation(
        tag=tag, split=a.split,
        model_path=Path(a.model), dataset_npz=Path(a.dataset_npz),
        outdir=outdir, eps=a.eps, n_show=a.n_show, seed=a.seed
    )


if __name__ == "__main__":
    main()
