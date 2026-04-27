# -*- coding: utf-8 -*-
from __future__ import annotations

import base64
import io
import json
import sys
from pathlib import Path

import flet as ft
import numpy as np
import tensorflow as tf
import joblib
import matplotlib.pyplot as plt
import pandas as pd


# -----------------------------------------------------------------------------
# Bootstrap: rodar via "uv run python src/adsorption_nn/gui_flet.py"
# garante import de adsorption_nn.config
# -----------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import adsorption_nn.config as cfg  # noqa: E402

cfg.ensure_dirs()

MODEL_PATH = cfg.ADS_BEST_MODEL
SCALER_IN_PATH = cfg.ADS_SCALER_IN
SCALER_OUT_PATH = cfg.ADS_SCALER_OUT
META_PATH = cfg.ADS_META

DATA_NPZ_PATH = cfg.ADS_FULL_NPZ
DATA_CSV_PATH = cfg.ADS_FULL_CSV

OUT_INFER_BASE = cfg.ADS_OUT_INFER
LATEST_TXT = OUT_INFER_BASE / "LATEST.txt"

BLOCK_SIZE_DEFAULT = 51

PARAM_COLS_FALLBACK = [
    "L", "Nz", "eps", "rho_B", "u", "D_ax", "kL", "qmax", "b", "n",
    "lam_z", "rho_g", "cp_g", "cp_s", "D_col", "h_w", "T_wall", "dH",
    "dt", "t_end", "C_in", "T_in"
]
FINAL_COLS_FALLBACK = ["C_out_final", "q_out_final", "T_out_final", "N_ads_final"]

VALORES_PADRAO_MAP = {
    "L": 0.978,
    "Nz": 51,
    "eps": 0.761,
    "rho_B": 381.785,
    "u": 0.808,
    "D_ax": 0.000,
    "kL": 0.003,
    "qmax": 0.136,
    "b": 7.620,
    "n": 1.065,
    "lam_z": 0.772,
    "rho_g": 1.706,
    "cp_g": 1037.205,
    "cp_s": 1118.988,
    "D_col": 0.049,
    "h_w": 16.764,
    "T_wall": 306.493,
    "dH": 48910.134,
    "dt": 0.333,
    "t_end": 333.321,
    "C_in": 1.691,
    "T_in": 301.962,
}


def load_meta():
    if META_PATH.exists():
        meta = json.loads(META_PATH.read_text(encoding="utf-8"))
        param_cols = meta.get("param_cols", PARAM_COLS_FALLBACK)
        final_cols = meta.get("final_cols", FINAL_COLS_FALLBACK)
        output_cols = meta.get("output_cols", [])
        block_size = int(meta.get("block_size", BLOCK_SIZE_DEFAULT))
        return param_cols, final_cols, output_cols, block_size
    return PARAM_COLS_FALLBACK, FINAL_COLS_FALLBACK, [], BLOCK_SIZE_DEFAULT


PARAM_COLS, FINAL_COLS, OUTPUT_COLS, BLOCK_SIZE = load_meta()

if not OUTPUT_COLS:
    OUTPUT_COLS = FINAL_COLS + (
        [f"C_z{i}" for i in range(BLOCK_SIZE)] +
        [f"q_z{i}" for i in range(BLOCK_SIZE)] +
        [f"T_z{i}" for i in range(BLOCK_SIZE)]
        # ["Qtot_final"] # tirei o for de plot
    )

LABELS = {
    "L":      "Comprimento Leito L (m)",
    "Nz":     "Malha Nz (-)",
    "eps":    "Porosidade ε (-)",
    "rho_B":  "Dens. Aparente ρb (kg/m³)",
    "u":      "Velocidade u (m/s)",
    "D_ax":   "Difusão axial Dax (m²/s)",
    "kL":     "Coef. Transferência Massa kL (1/s)",
    "qmax":   "Capacidade qmax (mol/kg)",
    "b":      "Const. Afinidade b (1/C)",
    "n":      "Exp. Freundlich n (-)",
    "lam_z":  "Disp. Térmica λz (W/m·K)",
    "rho_g":  "Dens. Gás ρg (kg/m³)",
    "cp_g":   "Capacidade Ter. Gás (J/kg·K)",
    "cp_s":   "Capacidade Ter. Sólido (J/kg·K)",
    "D_col":  "Diâmetro Interno Dcol (m)",
    "h_w":    "h Coef. Transf. Parede (W/m²·K)",
    "T_wall": "T parede (K)",
    "dH":     "Calor adsorvido ΔH (J/mol)",
    "dt":     "Passo dt (s)",
    "t_end":  "Tempo final (s)",
    "C_in":   "C entrada (mol/m³)",
    "T_in":   "T entrada (K)",
}


def b64_png(fig) -> str:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=140, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return "data:image/png;base64," + base64.b64encode(buf.read()).decode("utf-8")


def gerar_grafico(x, y_pred, titulo, eixo_x, eixo_y, y_true=None):
    fig, ax = plt.subplots(figsize=(3.4, 2)) # 4.6, 3.2 original
    if y_true is not None:
        ax.plot(x, y_true, label="true")
    ax.plot(x, y_pred, label="pred")
    ax.set_title(titulo)
    ax.set_xlabel(eixo_x, fontsize=12)     # eixo X
    ax.set_ylabel(eixo_y, fontsize=12)  # eixo Y
    ax.tick_params(axis="both", labelsize=12)
    ax.grid(True)
    ax.legend()
    return b64_png(fig)

def sumQtot_final(qz: np.ndarray, L: float, D_col: float, rho_B: float) -> float:
    qz = np.asarray(qz, dtype=float).reshape(-1)
    z = np.linspace(0.0, float(L), qz.size)
    A = np.pi * (float(D_col) ** 2) / 4.0
    integral = np.trapezoid(qz, z) if hasattr(np, "trapezoid") else np.trapz(qz, z)
    return float(A * float(rho_B) * integral)


def split_157(y_vec: np.ndarray):
    y_vec = np.asarray(y_vec, dtype=float).reshape(-1)
    finals = y_vec[0:4]
    Cz = y_vec[4:4 + BLOCK_SIZE]
    qz = y_vec[4 + BLOCK_SIZE:4 + 2 * BLOCK_SIZE]
    Tz = y_vec[4 + 2 * BLOCK_SIZE:4 + 3 * BLOCK_SIZE]
    # qtot_final = float(y_vec[4 + 3 * BLOCK_SIZE])
    return finals, Cz, qz, Tz # qtot_final

# -----------------------------------------------------------------------------
# LATEST: aceita path absoluto OU relativo ao ROOT
# -----------------------------------------------------------------------------
def get_latest_run_dir() -> Path | None:
    if LATEST_TXT.exists():
        raw = LATEST_TXT.read_text(encoding="utf-8").strip()
        if raw:
            p = Path(raw)
            if not p.is_absolute():
                p = cfg.ROOT / p
            if p.exists():
                return p

    if OUT_INFER_BASE.exists():
        dirs = [d for d in OUT_INFER_BASE.iterdir() if d.is_dir() and d.name[:8].isdigit()]
        if dirs:
            return sorted(dirs, key=lambda d: d.name)[-1]
    return None


# -----------------------------------------------------------------------------
# Métricas: primeiro tenta "release" (models/.../validation), depois "dev" (outputs)
# -----------------------------------------------------------------------------
def load_metrics(method: str):
    if method == "masked":
        blocks_csv = cfg.ADS_VAL_MASKED_BLOCKS
        finals_csv = cfg.ADS_VAL_MASKED_FINALS
        origin = f"Métricas do pacote do modelo: {cfg.ADS_VAL_MASKED_DIR}"
    else:
        blocks_csv = cfg.ADS_VAL_EPS_BLOCKS
        finals_csv = cfg.ADS_VAL_EPS_FINALS
        origin = f"Métricas do pacote do modelo: {cfg.ADS_VAL_EPS_DIR}"

    if blocks_csv.exists() and finals_csv.exists():
        return pd.read_csv(blocks_csv), pd.read_csv(finals_csv), origin

    run_dir = get_latest_run_dir()
    if run_dir is None:
        return None, None, (
            "Nenhuma métrica encontrada.\n"
            "- Usuário final: inclua models/adsorption/validation/ no repo.\n"
            "- Dev: rode validate_* para gerar outputs/..."
        )

    mdir = run_dir / method
    blocks_csv = mdir / "metricas_por_bloco_val.csv"
    finals_csv = mdir / "metricas_finais_individuais.csv"
    if not blocks_csv.exists() or not finals_csv.exists():
        return None, None, f"Não achei CSVs de métricas em: {mdir}"

    return pd.read_csv(blocks_csv), pd.read_csv(finals_csv), f"Métricas (dev): {mdir}"


def df_to_table(df: pd.DataFrame, max_rows: int = 12) -> ft.DataTable:
    df = df.copy()
    if len(df) > max_rows:
        df = df.iloc[:max_rows]

    def _fmt(v):
        if isinstance(v, float):
            return f"{v:.6g}"
        return str(v)

    cols = [ft.DataColumn(ft.Text(str(c))) for c in df.columns]
    rows = []
    for _, r in df.iterrows():
        cells = [ft.DataCell(ft.Text(_fmt(v))) for v in r.values]
        rows.append(ft.DataRow(cells=cells))
    return ft.DataTable(columns=cols, rows=rows, column_spacing=18, data_row_min_height=34)


# -----------------------------------------------------------------------------
# Dataset cache (opcional)
# -----------------------------------------------------------------------------
_DATA_CACHE = {"loaded": False, "X": None, "Y": None, "n": 0}


def load_dataset_cache():
    if _DATA_CACHE["loaded"]:
        return

    if DATA_NPZ_PATH.exists():
        data = np.load(str(DATA_NPZ_PATH), allow_pickle=True)
        if "data" not in data.files or "columns" not in data.files:
            raise ValueError("NPZ inválido: esperado chaves 'data' e 'columns'.")
        mat = data["data"]
        cols = [str(c) for c in data["columns"].tolist()]
        idx_x = [cols.index(c) for c in PARAM_COLS]
        idx_y = [cols.index(c) for c in OUTPUT_COLS]
        X = mat[:, idx_x].astype(np.float32, copy=False)
        Y = mat[:, idx_y].astype(np.float32, copy=False)
    elif DATA_CSV_PATH.exists():
        usecols = PARAM_COLS + OUTPUT_COLS
        df = pd.read_csv(DATA_CSV_PATH, usecols=usecols)
        X = df[PARAM_COLS].to_numpy(np.float32, copy=False)
        Y = df[OUTPUT_COLS].to_numpy(np.float32, copy=False)
    else:
        raise FileNotFoundError(
            "Dataset FULL não encontrado.\n"
            "Obs.: para usuário final isso é normal (o app roda sem dataset).\n"
            "Para treinar/validar do zero, coloque dataset em data/processed/adsorption/."
        )

    _DATA_CACHE["X"] = X
    _DATA_CACHE["Y"] = Y
    _DATA_CACHE["n"] = int(X.shape[0])
    _DATA_CACHE["loaded"] = True


# -----------------------------------------------------------------------------
# Modelo e scalers (obrigatórios para uso do app)
# -----------------------------------------------------------------------------
if not MODEL_PATH.exists():
    raise FileNotFoundError(f"Modelo não encontrado: {MODEL_PATH}")
if not SCALER_IN_PATH.exists() or not SCALER_OUT_PATH.exists():
    raise FileNotFoundError("Scalers não encontrados em models/adsorption (devem vir versionados).")

model = tf.keras.models.load_model(str(MODEL_PATH), compile=False)
scaler_in = joblib.load(SCALER_IN_PATH)
scaler_out = joblib.load(SCALER_OUT_PATH)


def main(page: ft.Page):
    page.title = "Predição - Adsorção (22 -> 157)"
    page.scroll = "always"
    page.window.width = 1500
    page.window.height = 930

    CARD_BG = "#151a1f"
    GREEN_ACTIVE = ft.Colors.GREEN_700
    BTN_H = 52
    INPUT_W = 230 # 220
    CARD_H = 350  # 455

    dataset_available = DATA_NPZ_PATH.exists() or DATA_CSV_PATH.exists()

    true_state = {"has_true": False, "idx": None, "x_true": None, "y_true": None}
    random_state = {"order": None, "pos": 0}

    # defaults alinhados com PARAM_COLS
    valores_padrao = [VALORES_PADRAO_MAP.get(nome, 0.0) for nome in PARAM_COLS]

    campos: list[ft.TextField] = []
    for nome, valor in zip(PARAM_COLS, valores_padrao):
        campos.append(
            ft.TextField(
                label=LABELS.get(nome, nome),
                value=str(valor).replace(".", ","),
                width=INPUT_W,
                text_size=21,
                label_style=ft.TextStyle(size=16),
            )
        )

    status_true = ft.Text(
        "TRUE: dataset FULL não está no pacote (OK)." if not dataset_available else "TRUE: (não carregado)",
        color="grey",
    )
    status_run = ft.Text("", color="grey")
    status_metrics = ft.Text("", color="grey")

    idx_field = ft.TextField(
        label="idx do dataset",
        value="",
        width=300,
        text_size=21,
        label_style=ft.TextStyle(size=16),
        disabled=not dataset_available,
    )

    def set_inputs_from_x(x_row: np.ndarray):
        for c, v in zip(campos, x_row):
            c.value = f"{float(v):.6g}".replace(".", ",")
        page.update()

    def carregar_idx(_e):
        try:
            load_dataset_cache()
            n = _DATA_CACHE["n"]
            idx = int(idx_field.value.strip())
            if idx < 0 or idx >= n:
                raise ValueError(f"idx fora do range: 0..{n-1}")
            x = _DATA_CACHE["X"][idx]
            y = _DATA_CACHE["Y"][idx].reshape(-1)
            set_inputs_from_x(x)
            true_state["has_true"] = True
            true_state["idx"] = idx
            true_state["x_true"] = x
            true_state["y_true"] = y
            status_true.value = f"TRUE: carregado (idx={idx})"
            status_true.color = "green"
        except Exception as err:
            true_state["has_true"] = False
            true_state["idx"] = None
            true_state["x_true"] = None
            true_state["y_true"] = None
            status_true.value = f"TRUE: erro -> {err}"
            status_true.color = "red"
        page.update()

    def carregar_random(_e):
        try:
            load_dataset_cache()
            n = _DATA_CACHE["n"]
            if random_state["order"] is None or random_state["pos"] >= n:
                rng = np.random.default_rng()
                order = np.arange(n)
                rng.shuffle(order)
                random_state["order"] = order
                random_state["pos"] = 0
            idx = int(random_state["order"][random_state["pos"]])
            random_state["pos"] += 1
            idx_field.value = str(idx)
            x = _DATA_CACHE["X"][idx]
            y = _DATA_CACHE["Y"][idx].reshape(-1)
            set_inputs_from_x(x)
            true_state["has_true"] = True
            true_state["idx"] = idx
            true_state["x_true"] = x
            true_state["y_true"] = y
            status_true.value = f"TRUE: carregado (idx={idx})"
            status_true.color = "green"
        except Exception as err:
            true_state["has_true"] = False
            true_state["idx"] = None
            true_state["x_true"] = None
            true_state["y_true"] = None
            status_true.value = f"TRUE: erro -> {err}"
            status_true.color = "red"
        page.update()

    PLACEHOLDER_SRC = (
        "data:image/png;base64,"
        "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMB/6X8xQAAAABJRU5ErkJggg=="
    )
    img_C = ft.Image(src=PLACEHOLDER_SRC, width=470)
    img_q = ft.Image(src=PLACEHOLDER_SRC, width=470)
    img_T = ft.Image(src=PLACEHOLDER_SRC, width=470)
    qtot_text = ft.Text("Qtot_final: -", size=20, weight=ft.FontWeight.BOLD)

    res_lines = {c: ft.Text(f"{c}: -", size=16) for c in FINAL_COLS}

    method_dd = ft.Dropdown(
        label="Método",
        value="masked",
        width=240,
        options=[ft.dropdown.Option("masked"), ft.dropdown.Option("eps")],
    )
    view_dd = ft.Dropdown(
        label="Ver",
        value="blocos",
        width=240,
        options=[ft.dropdown.Option("blocos"), ft.dropdown.Option("finais")],
    )
    table_holder = ft.Column([], spacing=8)
    df_blocks_cache = {"df": None}
    df_finals_cache = {"df": None}

    def render_metrics_table():
        table_holder.controls.clear()
        if view_dd.value == "blocos":
            dfb = df_blocks_cache["df"]
            table_holder.controls.append(df_to_table(dfb, 12) if dfb is not None else ft.Text("Sem métricas.", color="grey"))
        else:
            dff = df_finals_cache["df"]
            table_holder.controls.append(df_to_table(dff, 12) if dff is not None else ft.Text("Sem métricas.", color="grey"))
        page.update()

    def carregar_metricas(_e):
        dfb, dff, msg = load_metrics(method_dd.value)
        if dfb is None or dff is None:
            df_blocks_cache["df"] = None
            df_finals_cache["df"] = None
            status_metrics.value = msg
            status_metrics.color = "red"
        else:
            df_blocks_cache["df"] = dfb
            df_finals_cache["df"] = dff
            status_metrics.value = msg
            status_metrics.color = "green"
        render_metrics_table()

    view_dd.on_change = lambda e: render_metrics_table()

    btn_run = ft.Button("Rodar Modelo", height=BTN_H, width=280, style=ft.ButtonStyle(
        text_style=ft.TextStyle(size=25, weight=ft.FontWeight.BOLD)
    ))

    def set_run_button_state(state: str):
        if state == "ok":
            btn_run.text = "Rodado ✓"
            btn_run.bgcolor = ft.Colors.GREEN_700
            btn_run.color = ft.Colors.WHITE
        elif state == "err":
            btn_run.text = "Erro ✖"
            btn_run.bgcolor = ft.Colors.RED_700
            btn_run.color = ft.Colors.WHITE
        else:
            btn_run.text = "Rodar Modelo"
            btn_run.bgcolor = ft.Colors.BLUE_700
            btn_run.color = ft.Colors.WHITE
        page.update()

    def rodar_modelo(_e):
        try:
            valores = [float(c.value.replace(",", ".")) for c in campos]
            X = np.array(valores, dtype=float).reshape(1, -1)

            expected = int(getattr(scaler_in, "n_features_in_", X.shape[1]))
            if X.shape[1] != expected:
                raise ValueError(f"Scaler espera {expected} features, mas recebeu {X.shape[1]}.")

            Xn = scaler_in.transform(X)
            y_norm = model.predict(Xn, verbose=0)
            y_pred = scaler_out.inverse_transform(y_norm).reshape(-1)

            y_true_vec = true_state["y_true"] if true_state["has_true"] else None
            x_true_vec = true_state["x_true"] if true_state["has_true"] else None

            finals_pred, Cz_pred, qz_pred, Tz_pred = split_157(y_pred)

            L_pred = float(X[0, PARAM_COLS.index("L")])
            Dcol_pred = float(X[0, PARAM_COLS.index("D_col")])
            rhoB_pred = float(X[0, PARAM_COLS.index("rho_B")])
            qtot_pred = sumQtot_final(qz_pred, L=L_pred, D_col=Dcol_pred, rho_B=rhoB_pred)

            if y_true_vec is not None and x_true_vec is not None:
                finals_true, Cz_true, qz_true, Tz_true = split_157(y_true_vec)

                L_true = float(x_true_vec[PARAM_COLS.index("L")])
                Dcol_true = float(x_true_vec[PARAM_COLS.index("D_col")])
                rhoB_true = float(x_true_vec[PARAM_COLS.index("rho_B")])

                qtot_true = sumQtot_final(qz_true, L=L_true, D_col=Dcol_true, rho_B=rhoB_true)
            else:
                finals_true = Cz_true = qz_true = Tz_true = None
                qtot_true = None

            for i, name in enumerate(FINAL_COLS):
                if finals_true is not None:
                    res_lines[name].value = f"{name}: true={finals_true[i]:.6g} | pred={finals_pred[i]:.6g}"
                else:
                    res_lines[name].value = f"{name}: pred={finals_pred[i]:.6g}"

            x_leito = np.arange(BLOCK_SIZE)

            img_C.src = gerar_grafico(
                x_leito, Cz_pred, "C(z)",
                eixo_x="z (posição no leito)",
                eixo_y="C (mol/m³)",
                y_true=Cz_true
            )

            img_q.src = gerar_grafico(
                x_leito, qz_pred, "q(z)",
                eixo_x="z (posição no leito)",
                eixo_y="q (mol/kg)",
                y_true=qz_true
            )

            img_T.src = gerar_grafico(
                x_leito, Tz_pred, "T(z)",
                eixo_x="z (posição no leito)",
                eixo_y="T (K)",
                y_true=Tz_true
            )

            if qtot_true is not None:
                qtot_text.value = f"Qtot_final(calculado): true={qtot_true:.6g} | pred={qtot_pred:.6g}"
            else:
                qtot_text.value = f"Qtot_final(calculado): pred={qtot_pred:.6g}"

            status_run.value = "Predição realizada com sucesso."
            status_run.color = "green"
            set_run_button_state("ok")

        except Exception as err:
            status_run.value = f"Erro: {err}"
            status_run.color = "red"
            set_run_button_state("err")

        page.update()

    btn_run.on_click = rodar_modelo
    set_run_button_state("ready")

    graphs_view = ft.Column(
        [
            ft.Text("Gráficos (TRUE vs PRED)", size=35, weight=ft.FontWeight.BOLD),
            ft.Row([img_C, img_q], wrap=True, spacing=14),
            ft.Row([img_T], wrap=True, spacing=14),
            ft.Divider(height=10),
            qtot_text,
        ],
        spacing=4,
    )

    results_view = ft.Column(
        [
            ft.Text("Resultados numéricos", size=35, weight=ft.FontWeight.BOLD),
            ft.Text("Finais (true | pred) quando TRUE estiver carregado.", size=12, color="grey"),
            ft.Divider(height=8),
            res_lines[FINAL_COLS[0]],
            res_lines[FINAL_COLS[1]],
            res_lines[FINAL_COLS[2]],
            res_lines[FINAL_COLS[3]],
        ],
        spacing=8,
    )

    validations_view = ft.Column(
        [
            ft.Text("Validações (sem precisar do dataset)", size=35, weight=ft.FontWeight.BOLD),
            ft.Row(
                [
                    method_dd,
                    view_dd,
                    ft.Button("Carregar métricas", on_click=carregar_metricas, height=BTN_H),
                ],
                wrap=True,
                spacing=10,
            ),
            status_metrics,
            ft.Divider(height=8),
            table_holder,
        ],
        spacing=10,
    )

    content_holder = ft.Container(content=graphs_view, padding=10, expand=True)

    b_g = ft.Button("Gráficos", width=220, height=55, style=ft.ButtonStyle(
        text_style=ft.TextStyle(size=25, weight=ft.FontWeight.BOLD)
    ))
    b_r = ft.Button("Resultados", width=220, height=55, style=ft.ButtonStyle(
        text_style=ft.TextStyle(size=25, weight=ft.FontWeight.BOLD)
    ))
    b_v = ft.Button("Validações", width=220, height=55, style=ft.ButtonStyle(
        text_style=ft.TextStyle(size=25, weight=ft.FontWeight.BOLD)
    ))

    def set_sidebar_active(btn: ft.Button, active: bool):
        if active:
            btn.bgcolor = GREEN_ACTIVE
            btn.color = ft.Colors.WHITE
        else:
            btn.bgcolor = None
            btn.color = None

    def set_view(key: str):
        if key == "graficos":
            content_holder.content = graphs_view
        elif key == "resultados":
            content_holder.content = results_view
        else:
            content_holder.content = validations_view

        set_sidebar_active(b_g, key == "graficos")
        set_sidebar_active(b_r, key == "resultados")
        set_sidebar_active(b_v, key == "validacoes")
        page.update()

    b_g.on_click = lambda e: set_view("graficos")
    b_r.on_click = lambda e: set_view("resultados")
    b_v.on_click = lambda e: set_view("validacoes")

    sidebar = ft.Container(
        content=ft.Column(
            [
                ft.Text("Painel", size=40, weight=ft.FontWeight.BOLD),
                ft.Divider(height=8),
                b_g, b_r, b_v,
            ],
            spacing=10,
            horizontal_alignment="center",
        ),
        padding=10,
        width=220,
    )

    panel = ft.Container(
        content=ft.Row(
            [
                sidebar,
                ft.Container(width=12),
                ft.Container(content=content_holder, expand=True),
            ],
            spacing=0,
        ),
        padding=12,
        border_radius=12,
        bgcolor=CARD_BG,
    )

    btn_rand = ft.Button("Amostra aleatória", on_click=carregar_random, height=BTN_H, width=260, disabled=not dataset_available, style=ft.ButtonStyle(
        text_style=ft.TextStyle(size=25, weight=ft.FontWeight.BOLD)
    ))
    btn_load = ft.Button("Carregar idx", on_click=carregar_idx, height=BTN_H, width=260, disabled=not dataset_available, style=ft.ButtonStyle(
        text_style=ft.TextStyle(size=25, weight=ft.FontWeight.BOLD)
    ))

    actions_content = ft.Column(
        [
            ft.Text("Ações", size=40, weight=ft.FontWeight.BOLD),#30
            idx_field,
            ft.Row([btn_rand], alignment="center"),
            ft.Row([btn_load], alignment="center"),
            ft.Divider(22),
            status_true,
            status_run,
            ft.Divider(height=22),
            ft.Row([btn_run], alignment="center"),
            ft.Container(height=8),
        ],
        spacing=10,
        scroll="auto",
    )

    actions_card = ft.Container(
        content=actions_content,
        padding=14,
        border_radius=12,
        bgcolor=CARD_BG,
        width=330, # 330
        height=CARD_H,
    )

    inputs_card = ft.Container(
        content=ft.Column(
            [
                ft.Text("Entradas do modelo (22 parâmetros)", size=40, weight=ft.FontWeight.BOLD),
                ft.Row(campos, wrap=True, spacing=12, run_spacing=12),
            ],
            spacing=10,
            scroll="auto",
        ),
        padding=14,
        border_radius=12,
        bgcolor=CARD_BG,
        expand=1,
        height=CARD_H,
    )

    set_view("graficos")

    page.add(
        ft.Column(
            [
                ft.Row([actions_card, inputs_card], spacing=14, vertical_alignment="start"),
                panel,
            ],
            spacing=14,
        )
    )


if __name__ == "__main__":
    ft.app(target=main)