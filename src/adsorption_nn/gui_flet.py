# -*- coding: utf-8 -*-
"""
GUI (Flet) para predição do modelo de ADSORÇÃO.

- 22 inputs
- modelo retorna 208 saídas:
  - 4 finais (mostra em texto)
  - 4 perfis (plota)
- paths corretos:
  models/adsorption/...
"""

import base64
import io
from pathlib import Path

import flet as ft
import numpy as np
import tensorflow as tf
import joblib
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[2]

MODEL_PATH = ROOT / "models" / "adsorption" / "best_model.keras"
SCALER_IN_PATH = ROOT / "models" / "adsorption" / "scaler_input.save"
SCALER_OUT_PATH = ROOT / "models" / "adsorption" / "scaler_output.save"

BLOCK_SIZE = 51

INPUT_COLS = [
    "L", "Nz", "eps", "rho_B", "u", "D_ax", "kL", "qmax", "b", "n",
    "lam_z", "rho_g", "cp_g", "cp_s", "D_col", "h_w", "T_wall", "dH",
    "dt", "t_end", "C_in", "T_in"
]

FINAL_COLS = ["C_out_final", "q_out_final", "T_out_final", "N_ads_final"]

# valores padrão (baseado numa linha do seu preview)
VALORES_PADRAO = [
    0.978, 51, 0.761, 381.785, 0.808, 0.000, 0.003, 0.136, 7.620, 1.065,
    0.772, 1.706, 1037.205, 1118.988, 0.049, 16.764, 306.493, 48910.134,
    0.333, 333.321, 1.691, 301.962
]


def gerar_grafico(x, y_true, y_pred, titulo):
    fig, ax = plt.subplots(figsize=(4.4, 3.2))
    ax.plot(x, y_true, label="true")
    ax.plot(x, y_pred, label="pred")
    ax.set_title(titulo)
    ax.grid(True)
    ax.legend()

    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=120, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)

    b64 = base64.b64encode(buf.read()).decode("utf-8")
    return f"data:image/png;base64,{b64}"


def split_208(y_vec):
    """
    y_vec: shape (208,)
    indices:
    - finais: 0..3
    - C_z:    4..54
    - q_z:    55..105
    - T_z:    106..156
    - Qtot_t: 157..207
    """
    finals = y_vec[0:4]
    Cz = y_vec[4:4 + BLOCK_SIZE]
    qz = y_vec[4 + BLOCK_SIZE:4 + 2 * BLOCK_SIZE]
    Tz = y_vec[4 + 2 * BLOCK_SIZE:4 + 3 * BLOCK_SIZE]
    Qt = y_vec[4 + 3 * BLOCK_SIZE:4 + 4 * BLOCK_SIZE]
    return finals, Cz, qz, Tz, Qt


# carregar modelo e scalers
if not MODEL_PATH.exists():
    raise FileNotFoundError(f"Modelo não encontrado: {MODEL_PATH}")
if not SCALER_IN_PATH.exists() or not SCALER_OUT_PATH.exists():
    raise FileNotFoundError("Scalers não encontrados em models/adsorption. Rode o train.py.")

model = tf.keras.models.load_model(str(MODEL_PATH), compile=False)
scaler_in = joblib.load(SCALER_IN_PATH)
scaler_out = joblib.load(SCALER_OUT_PATH)


def main(page: ft.Page):
    page.title = "Predição - Adsorção (22 -> 208)"
    page.scroll = "always"
    page.window.width = 1300
    page.window.height = 850

    campos = []
    for nome, valor in zip(INPUT_COLS, VALORES_PADRAO):
        campos.append(ft.TextField(label=nome, value=str(valor).replace(".", ","), width=140))

    status = ft.Text("")

    # textos para finais
    txt_finais = [ft.Text(f"{c}: -") for c in FINAL_COLS]

    # placeholder 1x1
    PLACEHOLDER_SRC = (
        "data:image/png;base64,"
        "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMB/6X8xQAAAABJRU5ErkJggg=="
    )

    img_C = ft.Image(src=PLACEHOLDER_SRC, width=420)
    img_q = ft.Image(src=PLACEHOLDER_SRC, width=420)
    img_T = ft.Image(src=PLACEHOLDER_SRC, width=420)
    img_Q = ft.Image(src=PLACEHOLDER_SRC, width=420)

    def rodar(_e):
        try:
            valores = []
            for campo in campos:
                valores.append(float(campo.value.replace(",", ".")))

            if len(valores) != 22:
                raise ValueError(f"Esperava 22 inputs, recebi {len(valores)}")

            X = np.array(valores, dtype=float).reshape(1, -1)

            expected = int(getattr(scaler_in, "n_features_in_", X.shape[1]))
            if X.shape[1] != expected:
                raise ValueError(f"Scaler espera {expected} features, mas você passou {X.shape[1]}.")

            Xn = scaler_in.transform(X)
            y_norm = model.predict(Xn, verbose=0)
            y = scaler_out.inverse_transform(y_norm).reshape(-1)

            if y.shape[0] != 208:
                raise ValueError(f"Modelo retornou {y.shape[0]} saídas, esperado 208.")

            finals, Cz, qz, Tz, Qt = split_208(y)

            # atualizar textos finais
            for i, name in enumerate(FINAL_COLS):
                txt_finais[i].value = f"{name}: {finals[i]:.6g}"
                txt_finais[i].update()

            # plota perfis "pred" contra "pred" (não tem true aqui)
            # (se quiser comparar true, precisa escolher uma amostra do dataset)
            x = np.arange(BLOCK_SIZE)
            img_C.src = gerar_grafico(x, Cz, Cz, "C(z) (pred)")
            img_q.src = gerar_grafico(x, qz, qz, "q(z) (pred)")
            img_T.src = gerar_grafico(x, Tz, Tz, "T(z) (pred)")
            img_Q.src = gerar_grafico(x, Qt, Qt, "Qtot(t) (pred)")

            img_C.update(); img_q.update(); img_T.update(); img_Q.update()

            status.value = "Predição realizada com sucesso."
            status.color = "green"
        except Exception as err:
            status.value = f"Erro: {err}"
            status.color = "red"

        status.update()

    # Compatibilidade flet (Button novo, ElevatedButton antigo)
    if hasattr(ft, "Button"):
        botao = ft.Button("Rodar Modelo", on_click=rodar)
    else:
        botao = ft.ElevatedButton("Rodar Modelo", on_click=rodar)

    page.add(
        ft.Column(
            [
                ft.Text("Entradas (22 parâmetros):"),
                ft.Row(campos, wrap=True),
                botao,
                status,
                ft.Divider(),
                ft.Text("Saídas finais (4):"),
                ft.Column(txt_finais),
                ft.Divider(),
                ft.Row([img_C, img_q], wrap=True),
                ft.Row([img_T, img_Q], wrap=True),
            ]
        )
    )


if __name__ == "__main__":
    # versões novas do Flet recomendam run()
    if hasattr(ft, "run"):
        ft.run(main)
    else:
        ft.app(target=main)
