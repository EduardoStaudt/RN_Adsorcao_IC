# -*- coding: utf-8 -*-
"""api/app.py

Servidor HTTP (estilo DRMS) para expor a predição da RNA de ADSORÇÃO.

Endpoints:
- GET  /health  -> status
- POST /predict -> recebe JSON com 22 inputs e devolve JSON com *_final e *_points

Como rodar (dev):
    python api/app.py

    Depois o Flutter (web) chama, por exemplo:
    POST http://localhost:8000/predict

    Obs.: este backend precisa do modelo e scalers em:
    models/adsorption/best_model.keras
    models/adsorption/scaler_input.save
    models/adsorption/scaler_output.save
"""

from __future__ import annotations

from typing import Any, Dict

from flask import Flask, jsonify, request
from flask_cors import CORS

from inference.predictor import get_predictor


app = Flask(__name__)
CORS(app)  # no MVP: libera CORS geral (Flutter Web em localhost)


@app.get("/health")
def health():
    return jsonify({"status": "ok"})


@app.post("/predict")
def predict():
    payload: Dict[str, Any] = request.get_json(silent=True) or {}

    try:
        predictor = get_predictor()
        result = predictor.predict(payload)
        return jsonify(result)

    except Exception as err:
        # erro amigável para o FRONT
        return jsonify({
            "error": str(err),
            "hint": "Verifique chaves/valores do JSON conforme contracts/prediction_contract.md"
        }), 400


if __name__ == "__main__":
    # host 0.0.0.0 permite acessar pela rede local; para uso local, localhost também serve.
    app.run(host="0.0.0.0", port=8000, debug=True)