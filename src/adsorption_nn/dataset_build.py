# -*- coding: utf-8 -*-
"""
dataset_build.py - Gera dataset_FULL (CSV + NPZ) a partir dos mini CSVs dataset_batch_*.csv.

- Lê APENAS de: data/raw/adsorption/
- Salva em:     data/processed/adsorption/dataset_FULL.csv e .npz

Formato do NPZ (compatível com validate.py):
- data:    matriz (n_amostras, n_colunas)
- columns: vetor com nomes das colunas
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np

# Bootstrap import config
ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import adsorption_nn.config as cfg
cfg.ensure_dirs()

RAW_ADS_DIR = cfg.RAW_DIR / "adsorption"
OUT_CSV = cfg.ADS_FULL_CSV
OUT_NPZ = cfg.ADS_FULL_NPZ

print(f"[INFO] Procurando em: {RAW_ADS_DIR}")
csv_paths = sorted(RAW_ADS_DIR.rglob("dataset_batch_*.csv"))
print(f"[INFO] Encontrei {len(csv_paths)} arquivos dataset_batch_*.csv")

if not csv_paths:
    raise FileNotFoundError(f"Não achei dataset_batch_*.csv dentro de {RAW_ADS_DIR}")

dfs = []
for p in csv_paths:
    print(f"Lendo {p.name} ...")
    dfs.append(pd.read_csv(p))

full_df = pd.concat(dfs, ignore_index=True)
print("[OK] Dataset concatenado:", full_df.shape)

OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
full_df.to_csv(OUT_CSV, index=False)
print("[OK] Salvei:", OUT_CSV)

np.savez_compressed(OUT_NPZ, data=full_df.values, columns=full_df.columns.to_numpy())
print("[OK] Salvei:", OUT_NPZ)
