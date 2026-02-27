# -*- coding: utf-8 -*-
"""inference/predictor.py

Módulo de **inferência** (produção) para a rede de ADSORÇÃO.

Objetivo:
- Receber um dicionário com os **22 inputs** (chaves científicas do FRONT)
- Carregar modelo + scalers (uma vez) e executar a predição
- Devolver um dicionário pronto para JSON com o padrão:
  - escalares: *_final
  - vetores:   *_points

Este código reaproveita a lógica do `src/adsorption_nn/gui_flet.py`, mas SEM interface.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np

# TensorFlow + joblib são dependências do projeto
import tensorflow as tf
import joblib


# -----------------------------------------------------------------------------
# Bootstrap para importar `adsorption_nn.config` (a pasta src/ não é um pacote).
# -----------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import adsorption_nn.config as cfg  # noqa: E402


BLOCK_SIZE = 51  # tamanho dos vetores previstos pelo modelo (atual)

# Ordem de features esperada pelo scaler/modelo (mantida igual ao treino)
MODEL_INPUT_COLS = [
    "L", "Nz", "eps", "rho_B", "u", "D_ax", "kL", "qmax", "b", "n",
    "lam_z", "rho_g", "cp_g", "cp_s", "D_col", "h_w", "T_wall", "dH",
    "dt", "t_end", "C_in", "T_in"
]

# Chaves do FRONT (contrato) → chaves internas do modelo
# (o modelo atual usa rho_B e kL)
FRONT_TO_MODEL_KEY = {
    "rho_b": "rho_B",
    "kl": "kL",
}

FINAL_COLS = ["C_out_final", "q_out_final", "T_out_final", "N_ads_final"]


def _split_208(y_vec: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Divide o vetor (208,) em blocos.

    índices:
    - finais: 0..3
    - C_z:    4..54
    - q_z:    55..105
    - T_z:    106..156
    - Qtot_t: 157..207
    """
    y_vec = y_vec.reshape(-1)
    finals = y_vec[0:4]
    Cz = y_vec[4:4 + BLOCK_SIZE]
    qz = y_vec[4 + BLOCK_SIZE:4 + 2 * BLOCK_SIZE]
    Tz = y_vec[4 + 2 * BLOCK_SIZE:4 + 3 * BLOCK_SIZE]
    Qt = y_vec[4 + 3 * BLOCK_SIZE:4 + 4 * BLOCK_SIZE]
    return finals, Cz, qz, Tz, Qt


def _as_float(x: Any) -> float:
    """Converte valores vindos do JSON (numérico ou string) para float."""
    if isinstance(x, (int, float, np.number)):
        return float(x)
    if isinstance(x, str):
        # aceita vírgula decimal
        return float(x.replace(",", ".").strip())
    raise TypeError(f"Valor inválido para número: {x!r} (type={type(x)})")


class AdsorptionPredictor:
    """Carrega modelo e scalers e executa predições."""

    def __init__(self, model_path: Optional[Path] = None,
                 scaler_in_path: Optional[Path] = None,
                 scaler_out_path: Optional[Path] = None) -> None:
        cfg.ensure_dirs()

        self.model_path = model_path or cfg.ADS_BEST_MODEL
        self.scaler_in_path = scaler_in_path or cfg.ADS_SCALER_IN
        self.scaler_out_path = scaler_out_path or cfg.ADS_SCALER_OUT

        if not self.model_path.exists():
            raise FileNotFoundError(
                f"Modelo não encontrado: {self.model_path}. Rode `python src/adsorption_nn/train.py`."
            )
        if not self.scaler_in_path.exists() or not self.scaler_out_path.exists():
            raise FileNotFoundError(
                f"Scalers não encontrados em {self.scaler_in_path.parent}. Rode o treino."
            )

        # Carrega uma vez (o backend mantém em memória)
        self.model = tf.keras.models.load_model(str(self.model_path), compile=False)
        self.scaler_in = joblib.load(self.scaler_in_path)
        self.scaler_out = joblib.load(self.scaler_out_path)

    def _normalize_payload(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Aceita formatos:
        - { ... } (flat)
        - {"inputs": { ... }} (aninhado)

        Ignora chaves extras (ex.: seed).
        """
        if "inputs" in payload and isinstance(payload["inputs"], dict):
            payload = payload["inputs"]

        # aplica aliases FRONT → MODEL (sem destruir as originais)
        normalized: Dict[str, Any] = dict(payload)
        for front_key, model_key in FRONT_TO_MODEL_KEY.items():
            if model_key not in normalized and front_key in normalized:
                normalized[model_key] = normalized[front_key]

        return normalized

    def _build_X(self, payload: Dict[str, Any]) -> np.ndarray:
        """Monta o vetor X (1, 22) na ordem esperada pelo scaler/modelo."""
        payload = self._normalize_payload(payload)

        missing = [k for k in MODEL_INPUT_COLS if k not in payload]
        if missing:
            raise KeyError(
                "Faltando inputs: " + ", ".join(missing) +
                ". (Obs.: o backend aceita rho_b→rho_B e kl→kL)"
            )

        values = [_as_float(payload[k]) for k in MODEL_INPUT_COLS]
        X = np.array(values, dtype=float).reshape(1, -1)

        expected = int(getattr(self.scaler_in, "n_features_in_", X.shape[1]))
        if X.shape[1] != expected:
            raise ValueError(f"Scaler espera {expected} features, mas recebeu {X.shape[1]}.")

        return X

    def predict(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Executa a predição e devolve dict pronto para JSON."""
        X = self._build_X(payload)

        # Pegamos t_end e L para construir eixos (t_points e z_points)
        # Obs.: Mesmo que Nz venha diferente, o modelo atual prevê sempre BLOCK_SIZE pontos.
        payload_norm = self._normalize_payload(payload)
        L = _as_float(payload_norm.get("L"))
        t_end = _as_float(payload_norm.get("t_end"))

        Xn = self.scaler_in.transform(X)
        y_norm = self.model.predict(Xn, verbose=0)
        y = self.scaler_out.inverse_transform(y_norm).reshape(-1)

        if y.shape[0] != 208:
            raise ValueError(f"Modelo retornou {y.shape[0]} saídas, esperado 208.")

        finals, Cz, qz, Tz, Qt = _split_208(y)

        # Eixos (sempre com 51 pontos no modelo atual)
        t_points = np.linspace(0.0, t_end, BLOCK_SIZE)
        z_points = np.linspace(0.0, L, BLOCK_SIZE)

        # Placeholder para breakthrough (até a RNA prever explicitamente C_out(t))
        # Mantém compatibilidade com o FRONT.
        C_out_points = np.linspace(0.0, float(finals[0]), BLOCK_SIZE)

        result: Dict[str, Any] = {
            "C_out_final": float(finals[0]),
            "q_out_final": float(finals[1]),
            "T_out_final": float(finals[2]),
            "N_ads_final": float(finals[3]),

            "t_points": t_points.tolist(),
            "C_out_points": C_out_points.tolist(),
            "Qtot_points": Qt.tolist(),

            "z_points": z_points.tolist(),
            "C_z_points": Cz.tolist(),
            "q_z_points": qz.tolist(),
            "T_z_points": Tz.tolist(),
        }

        return result


# Singleton simples (carrega uma vez quando o módulo é importado)
_PREDICTOR: Optional[AdsorptionPredictor] = None


def get_predictor() -> AdsorptionPredictor:
    global _PREDICTOR
    if _PREDICTOR is None:
        _PREDICTOR = AdsorptionPredictor()
    return _PREDICTOR