from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from src.river_level.features import gerar_dataset_features
from src.utils.io_utils import garantir_pasta, ler_yaml, salvar_json


def carregar_base_processada(caminho_csv: str | Path) -> pd.DataFrame:
    """
    Carrega a base processada oficial da Fase 1.
    """
    caminho_csv = Path(caminho_csv)

    if not caminho_csv.exists():
        raise FileNotFoundError(f"Base processada não encontrada: {caminho_csv}")

    df = pd.read_csv(
        caminho_csv,
        sep=",",
        encoding="utf-8-sig",
        parse_dates=["Data"],
        index_col="Data",
    )

    if not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError(
            "A base processada carregada não possui índice temporal válido."
        )

    return df.sort_index()


def montar_resumo_execucao(
    config: dict,
    df_entrada: pd.DataFrame,
    df_saida: pd.DataFrame,
    caminho_entrada: Path,
    caminho_saida_csv: Path,
) -> dict:
    """
    Monta o resumo da execução para rastreabilidade local.
    """
    colunas_geradas = [col for col in df_saida.columns if col not in df_entrada.columns]

    return {
        "etapa": "engenharia_features",
        "arquivo_configuracao": str(config.get("_caminho_config", "")),
        "dataset_entrada": str(caminho_entrada),
        "dataset_saida": str(caminho_saida_csv),
        "n_linhas_entrada": int(len(df_entrada)),
        "n_linhas_saida": int(len(df_saida)),
        "n_colunas_entrada": int(df_entrada.shape[1]),
        "n_colunas_saida": int(df_saida.shape[1]),
        "data_inicial_entrada": str(df_entrada.index.min().date()),
        "data_final_entrada": str(df_entrada.index.max().date()),
        "data_inicial_saida": str(df_saida.index.min().date()),
        "data_final_saida": str(df_saida.index.max().date()),
        "colunas_entrada": list(df_entrada.columns),
        "colunas_saida": list(df_saida.columns),
        "colunas_geradas": colunas_geradas,
        "parametros_features": config["features"],
    }


def executar_engenharia_features(caminho_config: str | Path) -> None:
    """
    Executa a Fase 2 canônica do Caso 2.
    """
    caminho_config = Path(caminho_config)
    config = ler_yaml(caminho_config)
    config["_caminho_config"] = str(caminho_config)

    dados_cfg = config["dados"]
    features_cfg = config["features"]
    saidas_cfg = config["saidas"]

    caminho_entrada = Path(dados_cfg["caminho_base_processada"])
    caminho_saida_csv = Path(saidas_cfg["caminho_dataset_features"])
    caminho_saida_json = Path(saidas_cfg["caminho_resumo_execucao"])

    df_base_modelagem = carregar_base_processada(caminho_entrada)

    df_features = gerar_dataset_features(
        df_base_modelagem=df_base_modelagem,
        janelas_nivel=features_cfg["janelas_nivel"],
        defasagens=features_cfg["defasagens"],
        limiares_estiagem=features_cfg["limiares_estiagem"],
        fator_k_api=features_cfg["fator_k_api"],
        coluna_nivel=features_cfg.get("coluna_nivel", "Nivel"),
        coluna_precipitacao=features_cfg.get(
            "coluna_precipitacao",
            "Precip_Media_Estacoes",
        ),
        remover_nans_finais=features_cfg.get("remover_nans_finais", True),
    )

    garantir_pasta(caminho_saida_csv.parent)

    df_features.to_csv(
        caminho_saida_csv,
        index=True,
        index_label="Data",
        encoding="utf-8-sig",
    )

    resumo_execucao = montar_resumo_execucao(
        config=config,
        df_entrada=df_base_modelagem,
        df_saida=df_features,
        caminho_entrada=caminho_entrada,
        caminho_saida_csv=caminho_saida_csv,
    )

    salvar_json(
        dados=resumo_execucao,
        caminho_arquivo=caminho_saida_json,
    )

    print("Engenharia de features concluída com sucesso.")
    print(f"Base de entrada: {caminho_entrada}")
    print(f"Base de saída:   {caminho_saida_csv}")
    print(f"Resumo JSON:     {caminho_saida_json}")
    print(f"Linhas entrada:  {len(df_base_modelagem)}")
    print(f"Linhas saída:    {len(df_features)}")
    print(f"Colunas saída:   {df_features.shape[1]}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Executa a engenharia de features do estudo de caso Rio Doce."
    )
    parser.add_argument(
        "--config",
        required=True,
        help="Caminho do arquivo YAML de configuração.",
    )

    args = parser.parse_args()
    executar_engenharia_features(args.config)


if __name__ == "__main__":
    main()