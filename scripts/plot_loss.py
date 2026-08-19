#!/usr/bin/env python3
"""Script utilitário para plotar a evolução das perdas durante o treinamento."""

import argparse
import pandas as pd
import matplotlib.pyplot as plt
import os

def main():
    parser = argparse.ArgumentParser(description="Plota gráficos de Loss a partir de um arquivo CSV.")
    parser.add_argument("--csv", type=str, default="outputs/loss_history.csv",
                        help="Caminho para o arquivo CSV de log")
    parser.add_argument("--output", type=str, default="outputs/loss_plot.png",
                        help="Caminho onde a imagem do gráfico será salva")
    args = parser.parse_args()

    if not os.path.exists(args.csv):
        print(f"Erro: Arquivo {args.csv} não encontrado.")
        print("Você precisa rodar o treinamento primeiro para gerar o histórico de perdas.")
        return

    try:
        df = pd.read_csv(args.csv)
        
        plt.figure(figsize=(10, 6))
        
        if 'total_loss' in df.columns:
            plt.plot(df['step'], df['total_loss'], label='Total Loss', color='black', linewidth=2)
            
        if 'sd15_loss' in df.columns:
            plt.plot(df['step'], df['sd15_loss'], label='SD 1.5 Loss', color='red', alpha=0.7)
            
        # Adicione novos surrogates aqui no futuro (ex: ip2p_loss)
        if 'ip2p_loss' in df.columns:
            plt.plot(df['step'], df['ip2p_loss'], label='InstructPix2Pix Loss', color='blue', alpha=0.7)
            
        plt.title("Evolução da Perda (Loss) de Treinamento - DiffVax Ensemble")
        plt.xlabel("Passos (Steps)")
        plt.ylabel("Loss")
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.6)
        
        # Salva o arquivo de imagem
        plt.savefig(args.output, dpi=300, bbox_inches='tight')
        print(f"Sucesso! Gráfico gerado e salvo em: {args.output}")
        
    except Exception as e:
        print(f"Ocorreu um erro ao gerar o gráfico: {e}")

if __name__ == "__main__":
    main()
