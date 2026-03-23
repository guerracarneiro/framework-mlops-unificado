from __future__ import annotations

"""
Gera os artefatos de referência da Fase 2D replicada do artigo.

Saídas:
- labels_ref_fase2c.csv
- config_melhor.json

Fonte:
- execução oficial consolidada da Fase 2C
- YAML oficial da Fase 2C
"""

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd
import yaml


def criar_parser_argumentos() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Gera labels_ref_fase2c.csv e config_melhor.json para a Fase 2D replicada."
    )
    parser.add_argument(
        "--config-fase2d",
        type=str,
        required=True,
        help="Caminho do YAML da fase2d_artigo_replicado.",
    )
    parser.add_argument(
        "--config-fase2c",
        type=str,
        default="experiments/clones/fase2c_config_final_artigo.yaml",
        help="Caminho do YAML oficial da Fase 2C.",
    )
    return parser


def carregar_yaml(caminho: str | Path) -> dict[str, Any]:
    caminho = Path(caminho)

    if not caminho.exists():
        raise FileNotFoundError(f"Arquivo YAML não encontrado: {caminho}")

    with caminho.open("r", encoding="utf-8") as arquivo:
        return yaml.safe_load(arquivo)


def garantir_pasta(caminho: str | Path) -> Path:
    pasta = Path(caminho)
    pasta.mkdir(parents=True, exist_ok=True)
    return pasta


def carregar_resultado_oficial(config_fase2d: dict[str, Any]) -> pd.DataFrame:
    pasta_execucao = Path(config_fase2d["base"]["pasta_execucao_oficial"])
    arquivo_resultado = pasta_execucao / "resultado_amostras.csv"

    if not arquivo_resultado.exists():
        raise FileNotFoundError(f"Arquivo não encontrado: {arquivo_resultado}")

    df_resultado = pd.read_csv(arquivo_resultado, encoding="utf-8-sig")
    return df_resultado


def gerar_labels_ref(df_resultado: pd.DataFrame) -> pd.DataFrame:
    colunas_obrigatorias = ["TT", "MATGEN", "cluster"]

    for coluna in colunas_obrigatorias:
        if coluna not in df_resultado.columns:
            raise ValueError(
                f"Coluna obrigatória ausente em resultado_amostras.csv: {coluna}"
            )

    df_ref = df_resultado[["TT", "MATGEN", "cluster"]].copy()
    df_ref["ordem_no_tt"] = df_ref.groupby("TT").cumcount()
    df_ref["row_id"] = range(len(df_ref))
    df_ref = df_ref.rename(columns={"cluster": "label_ref"})

    df_ref = df_ref[["TT", "ordem_no_tt", "MATGEN", "label_ref", "row_id"]].copy()
    return df_ref


def gerar_config_melhor(config_fase2c: dict[str, Any]) -> dict[str, Any]:
    """
    Gera uma estrutura simples e estável com a configuração oficial da Fase 2C.
    """
    preprocessamento = config_fase2c["preprocessamento"]
    reducer = config_fase2c["modelagem"]["grid_modelagem"]["reducer"]
    clusterer = config_fase2c["modelagem"]["grid_modelagem"]["clusterer"]

    config_execucao = {
        "normalizacao_tipo": preprocessamento["normalizacao"]["tipo"],
        "onehot_colunas": preprocessamento["onehot"]["colunas"],
        "imputacao_tipo": preprocessamento["imputacao"]["tipo"],
        "imputacao_params": preprocessamento["imputacao"].get("params", {}),
        "umap_params": {
            "n_neighbors": reducer["params"]["n_neighbors"][0],
            "n_components": reducer["params"]["n_components"][0],
            "min_dist": reducer["params"]["min_dist"][0],
            "metric": reducer["params"]["metric"][0],
            "random_state": reducer["params"]["random_state"][0],
        },
        "hdbscan_params": {
            "min_cluster_size": clusterer["params"]["min_cluster_size"][0],
            "min_samples": clusterer["params"]["min_samples"][0],
            "cluster_selection_epsilon": clusterer["params"]["cluster_selection_epsilon"][0],
            "metric": clusterer["params"]["metric"][0],
        },
    }

    return {
        "config_execucao": config_execucao
    }


def main() -> None:
    parser = criar_parser_argumentos()
    args = parser.parse_args()

    config_fase2d = carregar_yaml(args.config_fase2d)
    config_fase2c = carregar_yaml(args.config_fase2c)

    pasta_saida = garantir_pasta(config_fase2d["saidas"]["pasta_artefatos"])

    df_resultado = carregar_resultado_oficial(config_fase2d)
    df_ref = gerar_labels_ref(df_resultado)

    caminho_labels_ref = pasta_saida / "labels_ref_fase2c.csv"
    df_ref.to_csv(caminho_labels_ref, index=False, encoding="utf-8-sig")

    config_melhor = gerar_config_melhor(config_fase2c)
    caminho_config_melhor = pasta_saida / "config_melhor.json"
    with caminho_config_melhor.open("w", encoding="utf-8") as arquivo:
        json.dump(config_melhor, arquivo, indent=2, ensure_ascii=False)

    print("=" * 80)
    print("ARTEFATOS DE REFERÊNCIA GERADOS")
    print("=" * 80)
    print(f"labels_ref_csv: {caminho_labels_ref}")
    print(f"config_melhor_json: {caminho_config_melhor}")
    print()
    print("Amostra de labels_ref:")
    print(df_ref.head().to_string(index=False))
    print()
    print("Configuração oficial extraída:")
    print(json.dumps(config_melhor, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()