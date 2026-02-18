# -*- coding: utf-8 -*-
"""
Compara duas validações do ADSORPTION_NN (EPS vs MASKED) para o MESMO tag.

Esperado no disco (por padrão):
outputs/adsorption/inference/<TAG>/eps/
    metricas_por_bloco_val.csv
    metricas_finais_individuais.csv

outputs/adsorption/inference/<TAG>/masked/
    metricas_por_bloco_val.csv
    metricas_finais_individuais.csv

Gera:
- outputs/adsorption/inference/<TAG>/compare_validations_<TAG>.csv
- outputs/adsorption/inference/<TAG>/compare_summary_<TAG>.txt
"""

import sys
import argparse
from pathlib import Path

import pandas as pd


# =======================================================
# Bootstrap para importar config.py
# =======================================================
ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import adsorption_nn.config as cfg
cfg.ensure_dirs()


def _read_csv(p: Path) -> pd.DataFrame:
    if not p.exists():
        raise FileNotFoundError(f"Arquivo não encontrado: {p}")
    return pd.read_csv(p)


def _fmt(x) -> str:
    try:
        if pd.isna(x):
            return "NA"
    except Exception:
        pass
    try:
        return f"{float(x):.6g}"
    except Exception:
        return str(x)


def compare(tag: str, out_base: Path) -> tuple[pd.DataFrame, str]:
    """
    Retorna:
    - df_compare (tabela completa)
    - texto resumo
    """
    run_dir = out_base / tag
    eps_dir = run_dir / "eps"
    msk_dir = run_dir / "masked"

    # por bloco
    eps_blocks = _read_csv(eps_dir / "metricas_por_bloco_val.csv")
    msk_blocks = _read_csv(msk_dir / "metricas_por_bloco_val.csv")

    # finais individuais
    eps_finals = _read_csv(eps_dir / "metricas_finais_individuais.csv")
    msk_finals = _read_csv(msk_dir / "metricas_finais_individuais.csv")

    # padroniza chaves
    eps_blocks["metodo"] = "EPS"
    msk_blocks["metodo"] = "MASKED"

    eps_finals["metodo"] = "EPS"
    msk_finals["metodo"] = "MASKED"

    # junta em formato "longo"
    eps_blocks["tipo"] = "bloco"
    msk_blocks["tipo"] = "bloco"
    eps_blocks = eps_blocks.rename(columns={"bloco": "nome"})
    msk_blocks = msk_blocks.rename(columns={"bloco": "nome"})

    eps_finals["tipo"] = "final"
    msk_finals["tipo"] = "final"
    eps_finals = eps_finals.rename(columns={"variavel": "nome"})
    msk_finals = msk_finals.rename(columns={"variavel": "nome"})

    df_long = pd.concat([eps_blocks, msk_blocks, eps_finals, msk_finals], ignore_index=True)

    # métricas que normalmente existem nos CSVs
    metrics = [c for c in ["mse", "rmse", "mae", "mape_%", "acc_%", "r2", "mape_used", "mape_total"] if c in df_long.columns]

    # pivota -> colunas por método
    df_piv = df_long.pivot_table(
        index=["tipo", "nome"],
        columns="metodo",
        values=metrics,
        aggfunc="first",
    )

    # achata multiindex de colunas
    df_piv.columns = [f"{m}_{met}" for (m, met) in df_piv.columns]
    df_piv = df_piv.reset_index()

    # cria deltas (EPS - MASKED) pra ver diferença
    for m in ["mse", "rmse", "mae", "mape_%", "acc_%", "r2"]:
        c_eps = f"{m}_EPS"
        c_msk = f"{m}_MASKED"
        if c_eps in df_piv.columns and c_msk in df_piv.columns:
            df_piv[f"{m}_DELTA(EPS-MASKED)"] = df_piv[c_eps] - df_piv[c_msk]

    # resumo textual
    def pick(tipo: str, nome: str) -> dict:
        row = df_piv[(df_piv["tipo"] == tipo) & (df_piv["nome"] == nome)]
        if row.empty:
            return {}
        return row.iloc[0].to_dict()

    # bloco finais(4) geralmente existe
    blk_f = pick("bloco", "Finais(4)")
    summary_lines = []
    summary_lines.append(f"COMPARE VALIDATIONS - TAG={tag}")
    summary_lines.append("=" * 72)

    if blk_f:
        summary_lines.append("Resumo (bloco Finais(4)):")
        for m in ["rmse", "mae", "mape_%", "acc_%", "r2"]:
            e = blk_f.get(f"{m}_EPS", None)
            k = blk_f.get(f"{m}_MASKED", None)
            d = blk_f.get(f"{m}_DELTA(EPS-MASKED)", None)
            if e is not None and k is not None:
                summary_lines.append(f"  {m:>6}: EPS={_fmt(e)} | MASKED={_fmt(k)} | DELTA={_fmt(d)}")
        summary_lines.append("-" * 72)

    # finais individuais mais importantes
    for var in ["C_out_final", "q_out_final", "T_out_final", "N_ads_final"]:
        r = pick("final", var)
        if not r:
            continue
        summary_lines.append(f"Final: {var}")
        for m in ["rmse", "mae", "mape_%", "acc_%", "r2"]:
            e = r.get(f"{m}_EPS", None)
            k = r.get(f"{m}_MASKED", None)
            d = r.get(f"{m}_DELTA(EPS-MASKED)", None)
            if e is not None and k is not None:
                summary_lines.append(f"  {m:>6}: EPS={_fmt(e)} | MASKED={_fmt(k)} | DELTA={_fmt(d)}")
        summary_lines.append("-" * 72)

    # observação direta sobre o que você viu
    summary_lines.append("Observação prática:")
    summary_lines.append("- Se MAPE_EPS ficar gigantesco (tipo 1e+07%), é quase sempre por causa de y_true ~ 0.")
    summary_lines.append("- O MASKED_MAE/RMSE/R2 continuam comparáveis, e o MAPE_mask fica 'menos explosivo' porque ignora |y_true| < threshold.")
    summary_lines.append("=" * 72)

    return df_piv, "\n".join(summary_lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True, help="Ex: 20260216_200000")
    ap.add_argument("--out_base", default=str(cfg.ADS_OUT_INFER), help="Base do outputs/adsorption/inference")
    args = ap.parse_args()

    out_base = Path(args.out_base)
    df, txt = compare(args.tag, out_base)

    run_dir = out_base / args.tag
    run_dir.mkdir(parents=True, exist_ok=True)

    out_csv = run_dir / f"compare_validations_{args.tag}.csv"
    out_txt = run_dir / f"compare_summary_{args.tag}.txt"

    df.to_csv(out_csv, index=False, encoding="utf-8")
    out_txt.write_text(txt, encoding="utf-8")

    print(txt)
    print(f"[OK] CSV comparação salvo em: {out_csv}")
    print(f"[OK] TXT resumo salvo em:    {out_txt}")


if __name__ == "__main__":
    main()
