# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
import sys
from pathlib import Path


# Seu arquivo do Drive (ID extraído do link)
DEFAULT_FILE_ID = "1N7GK1NneEwtyptZESoI1yD2Ek2RQOV0j"


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Baixa dataset_FULL.npz do Google Drive (cross-platform)."
    )
    ap.add_argument(
        "--file_id",
        default=DEFAULT_FILE_ID,
        help=f"ID do arquivo no Google Drive (default: {DEFAULT_FILE_ID})",
    )
    ap.add_argument(
        "--out",
        default="data/processed/adsorption/dataset_FULL.npz",
        help="Caminho de saída do .npz (default: data/processed/adsorption/dataset_FULL.npz)",
    )
    args = ap.parse_args()

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    try:
        import gdown
    except Exception:
        print("[ERRO] Dependência ausente: gdown")
        print("Instale com: pip install gdown")
        return 1

    # Link mais estável pro gdown (melhor que /file/d/.../view)
    url = f"https://drive.google.com/uc?id={args.file_id}"

    print("[INFO] Baixando do Google Drive...")
    print("[INFO] file_id :", args.file_id)
    print("[INFO] URL     :", url)
    print("[INFO] Saída   :", out.resolve())

    result = gdown.download(url, str(out), quiet=False, fuzzy=True)

    if (not result) or (not out.exists()) or out.stat().st_size == 0:
        print("[ERRO] Download falhou.")
        print("Causas comuns:")
        print("- O arquivo não está público: defina 'Anyone with the link' no Drive")
        print("- O file_id está errado")
        return 2

    mb = out.stat().st_size / (1024 * 1024)
    print(f"[OK] Download concluído ({mb:.2f} MB)")
    print("[OK] Arquivo:", out.resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
