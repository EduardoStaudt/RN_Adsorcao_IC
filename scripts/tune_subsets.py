"""
tune_subsets.py
Roda busca Optuna em cada subset e compara os melhores HPs entre eles.
Serve também como laboratório de experimentos (TAREFA 4): cada execução
é registrada em experiments_log.jsonl de forma acumulativa.

Uso:
    python scripts/tune_subsets.py [--n-trials N] [--epochs E] [--subset 1000,10000,50000]

Exemplos:
    python scripts/tune_subsets.py --n-trials 32 --epochs 100 --subset 1000
    python scripts/tune_subsets.py --n-trials 64 --epochs 200
"""

import sys
import argparse
import json
import re
import time
from datetime import datetime
from pathlib import Path

import optuna
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import adsorption_nn.config as cfg
cfg.ensure_dirs()

# --- Colunas (idênticas ao train.py) ---
PARAM_COLS = [
    "L", "Nz", "eps", "rho_B", "u", "D_ax", "kL", "qmax", "b", "n",
    "lam_z", "rho_g", "cp_g", "cp_s", "D_col", "h_w", "T_wall", "dH",
    "dt", "t_end", "C_in", "T_in"
]
FINAL_COLS = ["C_out_final", "q_out_final", "T_out_final", "N_ads_final"]
BLOCK_SIZE = 51


def _cols_by_prefix(all_cols: list[str], pref: str) -> list[str]:
    cols = [c for c in all_cols if c.startswith(pref)]
    def _key(cname: str):
        m = re.search(r"(\d+)$", cname)
        return int(m.group(1)) if m else 10**9
    return sorted(cols, key=_key)


def load_xy(path: Path) -> tuple[np.ndarray, np.ndarray, int]:
    df = pd.read_csv(path)
    all_cols = df.columns.tolist()

    Cz_cols = _cols_by_prefix(all_cols, "C_z")
    qz_cols = _cols_by_prefix(all_cols, "q_z")
    Tz_cols = _cols_by_prefix(all_cols, "T_z")

    for name, cols in [("C_z", Cz_cols), ("q_z", qz_cols), ("T_z", Tz_cols)]:
        if len(cols) != BLOCK_SIZE:
            raise ValueError(f"[ERRO] Esperava {BLOCK_SIZE} colunas para {name}, achei {len(cols)}.")

    OUTPUT_COLS = FINAL_COLS + Cz_cols + qz_cols + Tz_cols

    X_raw = df[PARAM_COLS].to_numpy(dtype=np.float32)
    Y_raw = df[OUTPUT_COLS].to_numpy(dtype=np.float32)

    scaler_X = StandardScaler().fit(X_raw)
    scaler_Y = StandardScaler().fit(Y_raw)

    X = scaler_X.transform(X_raw).astype(np.float32)
    Y = scaler_Y.transform(Y_raw).astype(np.float32)
    return X, Y, len(df)


def build_model(input_dim: int, output_dim: int,
                n_layers: int = 3, n_units: int = 352,
                activation: str = "elu", dropout: float = 0.10,
                l2_reg: float = 1e-5, lr: float = 5e-4):
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


class _PruningCallback(tf.keras.callbacks.Callback):
    """Reporta val_rmse ao Optuna a cada época e poda trials fracos."""
    def __init__(self, trial, monitor: str = "val_rmse"):
        super().__init__()
        self._trial = trial
        self._monitor = monitor

    def on_epoch_end(self, epoch, logs=None):
        val = (logs or {}).get(self._monitor, float("inf"))
        self._trial.report(val, epoch)
        if self._trial.should_prune():
            raise optuna.TrialPruned()


def run_optuna_for_subset(name: str, path: Path, n_trials: int, tune_epochs: int,
                           run_dir: Path, run_id: str, sub_db: Path, global_db: Path) -> dict:
    print(f"\n{'='*60}")
    print(f"[INFO] Subset: {name}  |  arquivo: {path.name}")

    X, Y, n_rows = load_xy(path)
    input_dim, output_dim = X.shape[1], Y.shape[1]
    print(f"[INFO] Linhas: {n_rows:,}  |  X: {X.shape}  |  Y: {Y.shape}")

    n_val = int(len(X) * 0.2)
    X_tr, Y_tr = X[n_val:], Y[n_val:]
    X_vl, Y_vl = X[:n_val], Y[:n_val]

    def objective(trial):
        n_l  = trial.suggest_int("n_layers", 2, 4)
        n_u  = trial.suggest_int("n_units", 64, 256, step=32)
        act  = trial.suggest_categorical("activation", ["relu", "elu"])
        drop = trial.suggest_float("dropout", 0.0, 0.2, step=0.05)
        l2   = trial.suggest_float("l2_reg", 1e-6, 1e-3, log=True)
        lr   = trial.suggest_float("lr", 1e-4, 1e-2, log=True)

        m = build_model(input_dim, output_dim,
                        n_layers=n_l, n_units=n_u, activation=act,
                        dropout=drop, l2_reg=l2, lr=lr)
        hist = m.fit(
            X_tr, Y_tr,
            validation_data=(X_vl, Y_vl),
            epochs=tune_epochs,
            batch_size=512,
            callbacks=[
                _PruningCallback(trial),
                tf.keras.callbacks.EarlyStopping(monitor="val_rmse", patience=15, restore_best_weights=True),
            ],
            verbose=0,
        )
        return min(hist.history.get("val_rmse", [float("inf")]))

    pruner     = optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=10)
    storage    = optuna.storages.RDBStorage(f"sqlite:///{sub_db.as_posix()}")
    study_name = f"sub_{name}_{run_id}"
    study      = optuna.create_study(
        direction="minimize",
        pruner=pruner,
        storage=storage,
        study_name=study_name,
        load_if_exists=True,
    )
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    t0 = time.time()
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)
    elapsed = time.time() - t0

    best_params = study.best_params
    best_val    = study.best_value
    print(f"[OK] best val_rmse = {best_val:.6f}  |  tempo = {elapsed:.1f}s")
    print(f"[OK] best HP: {best_params}")

    # Registra o best trial no DB global
    try:
        _global_storage = optuna.storages.RDBStorage(f"sqlite:///{global_db.as_posix()}")
        study_global = optuna.create_study(
            direction="minimize",
            storage=_global_storage,
            study_name=study_name,
            load_if_exists=True,
        )
        study_global.add_trial(study.best_trial)
    except Exception as e:
        print(f"[AVISO] Não foi possível registrar no DB global: {e}")

    # Gráficos HTML do subset
    try:
        import optuna.visualization as vis
        plots_dir = run_dir / "plots"
        vis.plot_optimization_history(study).write_html(
            str(plots_dir / f"opt_history_{name}.html"))
        vis.plot_param_importances(study).write_html(
            str(plots_dir / f"param_importance_{name}.html"))
        vis.plot_parallel_coordinate(study).write_html(
            str(plots_dir / f"parallel_coord_{name}.html"))
        print(f"[OK] Gráficos HTML salvos em: {plots_dir}")
    except Exception as e:
        print(f"[AVISO] Não foi possível gerar gráficos: {e}")

    # CSV de trials (para análise estatística)
    trials_csv = cfg.ADS_OUT_OPTUNA_CSV / f"trials_{name}_{run_id}.csv"
    trials_rows = [
        {"trial_number": t.number, "state": t.state.name,
         "val_rmse": t.value if t.value is not None else float("nan"),
         **t.params}
        for t in study.trials
    ]
    pd.DataFrame(trials_rows).to_csv(trials_csv, index=False, encoding="utf-8")
    print(f"[OK] CSV de trials salvo em: {trials_csv}")

    result = {
        "run_id":        run_id,
        "subset":        name,
        "n_rows":        n_rows,
        "n_trials":      n_trials,
        "tune_epochs":   tune_epochs,
        "best_val_rmse": best_val,
        "elapsed_s":     round(elapsed, 1),
        "best_params":   best_params,
        "db_path":       str(sub_db),
    }

    # JSON por subset
    hp_json = run_dir / f"best_hp_sub_{name}.json"
    hp_json.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[OK] Salvo em: {hp_json}")

    return result


def _append_to_log(result: dict, log_path: Path) -> None:
    """Acrescenta uma linha JSON ao log acumulativo (TAREFA 4)."""
    with log_path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(result, ensure_ascii=False) + "\n")


def _next_run_dir(sub_dir: Path) -> tuple[Path, str]:
    """Retorna (pasta, run_id) para a próxima execução incremental dentro de sub_dir."""
    today    = datetime.now().strftime("%Y%m%d")
    existing = sorted(sub_dir.glob("run_[0-9][0-9][0-9]_*"))
    n        = len(existing) + 1
    run_id   = f"run_{n:03d}_{today}"
    return sub_dir / run_id, run_id


def print_comparison(results: list[dict]) -> None:
    if not results:
        return

    HP_KEYS = ["n_layers", "n_units", "activation", "dropout", "l2_reg", "lr"]
    col_w   = 12

    header = (f"{'subset':<10} {'n_rows':>8} {'val_rmse':>10} {'tempo(s)':>9}" + "".join(f"  {k:>{col_w}}" for k in HP_KEYS))
    sep = "=" * len(header)

    print(f"\n{sep}")
    print("COMPARAÇÃO DE HIPERPARÂMETROS POR SUBSET")
    print(sep)
    print(header)
    print("-" * len(header))

    for r in results:
        line = (f"{r['subset']:<10} {r['n_rows']:>8,} "
                f"{r['best_val_rmse']:>10.6f} {r['elapsed_s']:>9.1f}")
        for k in HP_KEYS:
            val = r["best_params"].get(k, "N/A")
            if isinstance(val, float):
                line += f"  {val:>{col_w}.2e}"
            else:
                line += f"  {str(val):>{col_w}}"
        print(line)

    print(sep)


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--n-trials", type=int, default=64,
                        help="Número de trials Optuna por subset (padrão: 64)")
    parser.add_argument("--epochs", type=int, default=200,
                        help="Épocas por trial (padrão: 200)")
    parser.add_argument("--subset", type=str, default="all",
                        help="Subsets a rodar: 1000, 10000, 50000 ou all (padrão: all)")
    args = parser.parse_args()

    ALL_SUBSETS: dict[str, Path] = {
        "1000":  cfg.ADS_SUB_1000,
        "10000": cfg.ADS_SUB_10000,
        "50000": cfg.ADS_SUB_50000,
    }

    if args.subset == "all":
        selected = ALL_SUBSETS
    else:
        keys     = [s.strip() for s in args.subset.split(",")]
        selected = {k: v for k, v in ALL_SUBSETS.items() if k in keys}
        if not selected:
            print(f"[ERRO] Subset inválido: '{args.subset}'. Use: 1000, 10000, 50000 ou all.")
            sys.exit(1)

    out_dir   = cfg.ADS_OUT_OPTUNA / "subsets"
    log_path  = cfg.ADS_OUT_OPTUNA / "experiments_log.jsonl"
    global_db = cfg.ADS_OUT_OPTUNA / "optuna_global.db"
    out_dir.mkdir(parents=True, exist_ok=True)

    results = []
    for name, path in selected.items():
        if not path.exists():
            print(f"[AVISO] Arquivo não encontrado, pulando: {path}")
            continue

        sub_dir = out_dir / f"sub_{name}"
        sub_dir.mkdir(parents=True, exist_ok=True)
        run_dir, run_id = _next_run_dir(sub_dir)
        (run_dir / "plots").mkdir(parents=True, exist_ok=True)
        print(f"[INFO] Iniciando {run_id} em: {run_dir}")

        r = run_optuna_for_subset(
            name=name, path=path,
            n_trials=args.n_trials, tune_epochs=args.epochs,
            run_dir=run_dir, run_id=run_id,
            sub_db=sub_dir / f"sub_{name}.db",
            global_db=global_db,
        )
        results.append(r)
        _append_to_log(r, log_path)  # log acumulativo — TAREFA 4

    print_comparison(results)

    if results:
        parts = " | ".join(f"sub_{r['subset']}: {r['elapsed_s']:.1f}s" for r in results)
        total = sum(r["elapsed_s"] for r in results)
        print(f"\n[TEMPO] {parts} | TOTAL: {total:.1f}s")

        summary_path = cfg.ADS_OUT_OPTUNA / "comparison_summary.json"
        summary_path.write_text(
            json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        print(f"\n[OK] Sumário da sessão salvo em:  {summary_path}")
        print(f"[OK] Log acumulativo atualizado: {log_path}")


if __name__ == "__main__":
    main()
