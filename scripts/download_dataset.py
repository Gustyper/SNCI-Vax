#!/usr/bin/env python3
"""Script utilitário para baixar datasets do Hugging Face localmente."""

import argparse
from huggingface_hub import snapshot_download
import os

def main():
    parser = argparse.ArgumentParser(description="Baixa um dataset do Hugging Face Hub.")
    parser.add_argument("--repo_id", type=str, default="ozdentarikcan/DiffVaxDataset",
                        help="ID do repositório no Hugging Face (ex: ozdentarikcan/DiffVaxDataset)")
    parser.add_argument("--local_dir", type=str, default="data",
                        help="Pasta local onde os arquivos serão salvos")
    parser.add_argument("--token", type=str, default=None,
                        help="Seu token do Hugging Face (crie em huggingface.co/settings/tokens) para evitar bloqueio de 429 Too Many Requests")
    args = parser.parse_args()

    # Cria a pasta local caso não exista
    os.makedirs(args.local_dir, exist_ok=True)

    print(f"Iniciando download do dataset '{args.repo_id}'...")
    print(f"Os arquivos serão salvos na pasta local: '{args.local_dir}'")
    
    try:
        snapshot_download(
            repo_id=args.repo_id,
            repo_type="dataset",
            local_dir=args.local_dir,
            resume_download=True,  # Se cair, ele retoma de onde parou
            token=args.token
        )
        print("\nDownload concluído com sucesso!")
        print("Agora você pode fazer upload desta pasta para o seu Google Drive.")
    except Exception as e:
        print(f"\nOcorreu um erro durante o download: {e}")

if __name__ == "__main__":
    main()
