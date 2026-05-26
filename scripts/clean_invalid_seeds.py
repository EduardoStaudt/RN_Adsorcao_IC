"""
clean_invalid_seeds.py
Remove linhas onde N_ads_final ≈ 0 (usando np.isclose com atol=1e-9).
Seeds com N_ads_final próximo de zero indicam simulações degeneradas.

Uso:
    python scripts/clean_invalid_seeds.py [--dry-run]

Com --dry-run apenas conta e exibe, sem alterar arquivos.
"""

import sys
import argparse
import numpy as np
import pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

TARGETS = [
    ROOT / "data" / "processed" / "adsorption" / "dataset_FULL.csv",
    ROOT / "data" / "newdataset" / "SubSets" / "dataset_optuna_1000.csv",
    ROOT / "data" / "newdataset" / "SubSets" / "dataset_optuna_10000.csv",
    ROOT / "data" / "newdataset" / "SubSets" / "dataset_optuna_50000.csv",
]

ATOL = 1e-9
COL = "N_ads_final"
CHUNKSIZE = 50_000  # para o FULL.csv de 4 GB


def is_invalid(series: pd.Series) -> np.ndarray:
    return np.isclose(series.to_numpy(dtype=np.float64), 0.0, atol=ATOL)


def clean_large(path: Path, dry_run: bool) -> tuple[int, int]:
    """Processa o dataset_FULL.csv em chunks (evita OOM)."""
    total = 0
    kept_chunks = []

    for chunk in pd.read_csv(path, chunksize=CHUNKSIZE, low_memory=False):
        total += len(chunk)
        mask_invalid = is_invalid(chunk[COL])
        kept_chunks.append(chunk[~mask_invalid])

    df_clean = pd.concat(kept_chunks, ignore_index=True)
    removed = total - len(df_clean)

    if not dry_run:
        df_clean.to_csv(path, index=False)

    return total, removed


def clean_small(path: Path, dry_run: bool) -> tuple[int, int]:
    """Processa arquivos menores (subsets) de uma vez."""
    df = pd.read_csv(path, low_memory=False)
    total = len(df)
    mask_invalid = is_invalid(df[COL])
    df_clean = df[~mask_invalid].reset_index(drop=True)
    removed = total - len(df_clean)

    if not dry_run:
        df_clean.to_csv(path, index=False)

    return total, removed


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dry-run", action="store_true",
                        help="Apenas conta as linhas inválidas, sem alterar arquivos.")
    args = parser.parse_args()

    dry_run = args.dry_run
    mode = "[DRY-RUN]" if dry_run else "[EXECUTANDO]"
    print(f"\n{mode} Critério: np.isclose({COL}, 0, atol={ATOL})\n")
    print(f"{'Arquivo':<35} {'Total':>10} {'Removidas':>11} {'Restantes':>11} {'%':>7}")
    print("-" * 77)

    grand_total = grand_removed = 0

    for path in TARGETS:
        if not path.exists():
            print(f"{path.name:<35} {'NÃO ENCONTRADO':>40}")
            continue

        is_large = "FULL" in path.name
        fn = clean_large if is_large else clean_small

        print(f"{path.name:<35}", end="", flush=True)
        total, removed = fn(path, dry_run=dry_run)
        kept = total - removed
        pct = removed / total * 100 if total else 0

        grand_total += total
        grand_removed += removed

        print(f" {total:>10,} {removed:>11,} {kept:>11,} {pct:>6.3f}%")

    print("-" * 77)
    grand_kept = grand_total - grand_removed
    grand_pct = grand_removed / grand_total * 100 if grand_total else 0
    print(f"{'TOTAL':<35} {grand_total:>10,} {grand_removed:>11,} {grand_kept:>11,} {grand_pct:>6.3f}%")

    if dry_run:
        print("\n[DRY-RUN] Nenhum arquivo foi alterado.")
    else:
        print("\n[OK] Limpeza concluída. Arquivos sobrescritos sem as linhas inválidas.")


if __name__ == "__main__":
    main()
