from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.river_level.sequencias import montar_dataset_sequencial
from src.utils.io_utils import garantir_pasta, ler_yaml, salvar_json


def carregar_base_features(caminho_arquivo: str | Path) -> pd.DataFrame:
    """
    Carrega a base oficial de features gerada na Fase 2.
    """
    caminho_arquivo = Path(caminho_arquivo)

    if not caminho_arquivo.exists():
        raise FileNotFoundError(f"Base de features não encontrada: {caminho_arquivo}")

    df = pd.read_csv(
        caminho_arquivo,
        parse_dates=["Data"],
        index_col="Data",
        encoding="utf-8-sig",
    )

    if not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError(
            "A base de features carregada não possui índice temporal válido."
        )

    return df.sort_index()


def salvar_dataset_sequencial(
    dataset: dict[str, Any],
    caminho_arquivo: str | Path,
) -> Path:
    """
    Salva o dataset sequencial em formato NPZ compactado.
    """
    caminho_arquivo = Path(caminho_arquivo)
    garantir_pasta(caminho_arquivo.parent)

    np.savez_compressed(
        caminho_arquivo,
        X=dataset["X"],
        y=dataset["y"],
        datas_inicio_janela=dataset["datas_inicio_janela"],
        datas_fim_janela=dataset["datas_fim_janela"],
        datas_alvo=dataset["datas_alvo"],
        colunas_entrada=dataset["colunas_entrada"],
        coluna_alvo=np.array([dataset["coluna_alvo"]]),
        indice_alvo_nas_entradas=np.array([dataset["indice_alvo_nas_entradas"]]),
        passos_entrada=np.array([dataset["passos_entrada"]]),
        horizonte_previsao=np.array([dataset["horizonte_previsao"]]),
    )

    return caminho_arquivo


def montar_resumo_execucao(
    config: dict[str, Any],
    df_features: pd.DataFrame,
    dataset: dict[str, Any],
    caminho_base_features: Path,
    caminho_dataset_sequencial: Path,
) -> dict[str, Any]:
    """
    Gera um resumo leve da Fase 3A para rastreabilidade.
    """
    X = dataset["X"]
    y = dataset["y"]

    return {
        "etapa": "preparacao_sequencias",
        "arquivo_configuracao": str(config.get("_caminho_config", "")),
        "base_features_entrada": str(caminho_base_features),
        "dataset_sequencial_saida": str(caminho_dataset_sequencial),
        "n_linhas_base_features": int(len(df_features)),
        "n_colunas_base_features": int(df_features.shape[1]),
        "coluna_alvo": dataset["coluna_alvo"],
        "n_colunas_entrada": int(len(dataset["colunas_entrada"])),
        "colunas_entrada": dataset["colunas_entrada"].tolist(),
        "indice_alvo_nas_entradas": int(dataset["indice_alvo_nas_entradas"]),
        "passos_entrada": int(dataset["passos_entrada"]),
        "horizonte_previsao": int(dataset["horizonte_previsao"]),
        "shape_X": list(X.shape),
        "shape_y": list(y.shape),
        "n_amostras": int(len(y)),
        "data_inicial_base_features": str(df_features.index.min().date()),
        "data_final_base_features": str(df_features.index.max().date()),
        "data_primeira_janela": str(dataset["datas_inicio_janela"][0]),
        "data_fim_primeira_janela": str(dataset["datas_fim_janela"][0]),
        "data_primeiro_alvo": str(dataset["datas_alvo"][0]),
        "data_ultima_janela": str(dataset["datas_inicio_janela"][-1]),
        "data_fim_ultima_janela": str(dataset["datas_fim_janela"][-1]),
        "data_ultimo_alvo": str(dataset["datas_alvo"][-1]),
    }


def executar_preparacao_sequencias(caminho_config: str | Path) -> None:
    """
    Executa a Fase 3A do Caso 2.
    """
    caminho_config = Path(caminho_config)
    config = ler_yaml(caminho_config)
    config["_caminho_config"] = str(caminho_config)

    sequencias_cfg = config["sequencias"]

    caminho_base_features = Path(
        sequencias_cfg.get(
            "caminho_base_features",
            "data/processed/river_level/dataset_features_baseline_lstm.csv",
        )
    )

    caminho_dataset_sequencial = Path(
        sequencias_cfg.get(
            "caminho_dataset_sequencial",
            "data/processed/river_level/dataset_sequencial_baseline_lstm.npz",
        )
    )

    caminho_resumo_execucao = Path(
        sequencias_cfg.get(
            "caminho_resumo_execucao",
            "artifacts/river_level/baseline_lstm/sequencias/resumo_sequencias_baseline_lstm.json",
        )
    )

    df_features = carregar_base_features(caminho_base_features)

    dataset = montar_dataset_sequencial(
        df_features=df_features,
        coluna_alvo=sequencias_cfg.get("coluna_alvo", "Nivel"),
        usar_todas_as_colunas_como_entrada=sequencias_cfg.get(
            "usar_todas_as_colunas_como_entrada",
            True,
        ),
        colunas_entrada=sequencias_cfg.get("colunas_entrada"),
        incluir_alvo_nas_entradas=sequencias_cfg.get(
            "incluir_alvo_nas_entradas",
            True,
        ),
        passos_entrada=sequencias_cfg["passos_entrada"],
        horizonte_previsao=sequencias_cfg.get("horizonte_previsao", 1),
    )

    salvar_dataset_sequencial(
        dataset=dataset,
        caminho_arquivo=caminho_dataset_sequencial,
    )

    resumo_execucao = montar_resumo_execucao(
        config=config,
        df_features=df_features,
        dataset=dataset,
        caminho_base_features=caminho_base_features,
        caminho_dataset_sequencial=caminho_dataset_sequencial,
    )

    salvar_json(
        dados=resumo_execucao,
        caminho_arquivo=caminho_resumo_execucao,
    )

    print("Preparação de sequências concluída com sucesso.")
    print(f"Base de features:       {caminho_base_features}")
    print(f"Dataset sequencial:     {caminho_dataset_sequencial}")
    print(f"Resumo JSON:            {caminho_resumo_execucao}")
    print(f"Shape X:                {dataset['X'].shape}")
    print(f"Shape y:                {dataset['y'].shape}")
    print(f"Primeiro alvo:          {dataset['datas_alvo'][0]}")
    print(f"Último alvo:            {dataset['datas_alvo'][-1]}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Executa a preparação de sequências do estudo de caso Rio Doce."
    )
    parser.add_argument(
        "--config",
        required=True,
        help="Caminho do arquivo YAML de configuração.",
    )

    args = parser.parse_args()
    executar_preparacao_sequencias(args.config)


if __name__ == "__main__":
    main()