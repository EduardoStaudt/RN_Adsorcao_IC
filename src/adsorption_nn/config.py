# -*- coding: utf-8 -*-
"""
config.py - Centraliza caminhos e padrões do projeto

Assim você não precisa consertar path em 5 arquivos diferentes.

Obs.: apesar de estar dentro de src/adsorption_nn/, este arquivo define
caminhos do projeto TODO (adsorption e nu_uvec). O módulo nu_uvec_nn
importa este config via um wrapper em src/nu_uvec_nn/config.py.
"""

from __future__ import annotations
from pathlib import Path
from datetime import datetime

# Raiz do projeto: RN_Adsorcao_IC/
ROOT = Path(__file__).resolve().parents[2]

# -----------------------
# DATASETS (canônicos)
# -----------------------
DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"

ADS_PROCESSED = PROCESSED_DIR / "adsorption"
NU_PROCESSED = PROCESSED_DIR / "nu_uvec"

## SUBSETS
SUBSETS_DIR = DATA_DIR / "newdataset" / "SubSets"
ADS_SUB_1000  = SUBSETS_DIR / "dataset_optuna_1000.csv"
ADS_SUB_10000 = SUBSETS_DIR / "dataset_optuna_10000.csv"
ADS_SUB_50000 = SUBSETS_DIR / "dataset_optuna_50000.csv"

ADS_FULL_CSV = ADS_PROCESSED / "dataset_FULL.csv"
ADS_FULL_NPZ = ADS_PROCESSED / "dataset_FULL.npz"

NU_ANALITICO_CSV = NU_PROCESSED / "dataset_nu_Uvec_analitico.csv"

# -----------------------
# MODELS
# -----------------------
MODELS_DIR = ROOT / "models"
ADS_MODELS_DIR = MODELS_DIR / "adsorption"
NU_MODELS_DIR = MODELS_DIR / "nu_uvec"

# Adsorption
ADS_BEST_MODEL = ADS_MODELS_DIR / "best_model.keras"
ADS_SCALER_IN = ADS_MODELS_DIR / "scaler_input.save"
ADS_SCALER_OUT = ADS_MODELS_DIR / "scaler_output.save"
ADS_META = ADS_MODELS_DIR / "model_meta.json"
ADS_BEST_HP = ADS_MODELS_DIR / "best_hp.json"

# -----------------------
# MÉTRICAS "DE RELEASE" (versionáveis junto com o modelo)
# -----------------------
ADS_VALIDATION_DIR = ADS_MODELS_DIR / "validation"
ADS_VAL_EPS_DIR = ADS_VALIDATION_DIR / "eps"
ADS_VAL_MASKED_DIR = ADS_VALIDATION_DIR / "masked"

ADS_VAL_EPS_BLOCKS = ADS_VAL_EPS_DIR / "blocks.csv"
ADS_VAL_EPS_FINALS = ADS_VAL_EPS_DIR / "finals.csv"
ADS_VAL_EPS_SUMMARY = ADS_VAL_EPS_DIR / "summary.json"

ADS_VAL_MASKED_BLOCKS = ADS_VAL_MASKED_DIR / "blocks.csv"
ADS_VAL_MASKED_FINALS = ADS_VAL_MASKED_DIR / "finals.csv"
ADS_VAL_MASKED_SUMMARY = ADS_VAL_MASKED_DIR / "summary.json"

ADS_VAL_LATEST = ADS_VALIDATION_DIR / "LATEST.json"

# Nu/Uvec
NU_BEST_MODEL = NU_MODELS_DIR / "best_model.keras"
NU_SCALER_X = NU_MODELS_DIR / "scaler_X.pkl"
NU_SCALER_Y = NU_MODELS_DIR / "scaler_Y.pkl"
NU_META = NU_MODELS_DIR / "model_meta.json"
# Split em NPZ (estilo vali.py): X_* e y_* NORMALIZADOS + mean/std para desnormalizar
NU_SPLIT_NPZ = NU_MODELS_DIR / "dataset_split.npz"

# -----------------------
# OUTPUTS
# -----------------------
OUT_DIR = ROOT / "outputs"

# Adsorption
ADS_OUT_DIR = OUT_DIR / "adsorption"
ADS_OUT_TRAIN = ADS_OUT_DIR / "training"
ADS_OUT_INFER = ADS_OUT_DIR / "inference"
ADS_OUT_OPTUNA = ADS_OUT_TRAIN / "optuna"
ADS_CURVE_PATH = ADS_OUT_TRAIN / "curva_treinamento.png"

# Nu/Uvec
NU_OUT_DIR = OUT_DIR / "nu_uvec"
NU_OUT_TRAIN = NU_OUT_DIR / "training"
NU_OUT_INFER = NU_OUT_DIR / "inference"
NU_CURVE_PATH = NU_OUT_TRAIN / "curva_treinamento.png"


def now_tag() -> str:
    """Tag YYYYMMDD_HHMMSS para nomes únicos."""
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def ensure_dirs() -> None:
    """Cria todas as pastas esperadas do projeto."""
    ADS_PROCESSED.mkdir(parents=True, exist_ok=True)
    NU_PROCESSED.mkdir(parents=True, exist_ok=True)

    ADS_MODELS_DIR.mkdir(parents=True, exist_ok=True)
    NU_MODELS_DIR.mkdir(parents=True, exist_ok=True)

    ADS_VALIDATION_DIR.mkdir(parents=True, exist_ok=True)
    ADS_VAL_EPS_DIR.mkdir(parents=True, exist_ok=True)
    ADS_VAL_MASKED_DIR.mkdir(parents=True, exist_ok=True)

    ADS_OUT_TRAIN.mkdir(parents=True, exist_ok=True)
    ADS_OUT_INFER.mkdir(parents=True, exist_ok=True)
    ADS_OUT_OPTUNA.mkdir(parents=True, exist_ok=True)

    NU_OUT_TRAIN.mkdir(parents=True, exist_ok=True)
    NU_OUT_INFER.mkdir(parents=True, exist_ok=True)