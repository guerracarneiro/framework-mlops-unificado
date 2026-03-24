from __future__ import annotations

from typing import Iterable

import numpy as np
import pandas as pd

from src.river_level.preprocessamento import validar_indice_temporal


def validar_base_sequencial(
    df: pd.DataFrame,
    coluna_alvo: str,
) -> None:
    """
    Valida se a base de entrada está pronta para a montagem de sequências.
    """
    validar_indice_temporal(df)

    if coluna_alvo not in df.columns:
        raise ValueError(
            f"A coluna alvo '{coluna_alvo}' não existe na base de features."
        )

    if df.empty:
        raise ValueError("A base de features está vazia.")

    if df.isna().sum().sum() > 0:
        raise ValueError(
            "A base de features contém valores ausentes. "
            "A Fase 3A exige uma base sem NaN."
        )


def definir_colunas_entrada(
    df: pd.DataFrame,
    coluna_alvo: str,
    usar_todas_as_colunas_como_entrada: bool = True,
    colunas_entrada: Iterable[str] | None = None,
    incluir_alvo_nas_entradas: bool = True,
) -> list[str]:
    """
    Define quais colunas da base serão usadas como entrada do modelo.
    """
    if usar_todas_as_colunas_como_entrada:
        colunas_modelagem = list(df.columns)
    else:
        if colunas_entrada is None:
            raise ValueError(
                "Quando 'usar_todas_as_colunas_como_entrada' for False, "
                "é necessário informar 'colunas_entrada'."
            )

        colunas_modelagem = list(colunas_entrada)

    colunas_ausentes = [col for col in colunas_modelagem if col not in df.columns]
    if colunas_ausentes:
        raise ValueError(
            "Nem todas as colunas de entrada existem na base. "
            f"Colunas ausentes: {colunas_ausentes}"
        )

    if not incluir_alvo_nas_entradas:
        colunas_modelagem = [col for col in colunas_modelagem if col != coluna_alvo]

    if len(colunas_modelagem) == 0:
        raise ValueError("Nenhuma coluna de entrada foi definida para a modelagem.")

    return colunas_modelagem


def gerar_sequencias_supervisionadas(
    df: pd.DataFrame,
    coluna_alvo: str,
    colunas_entrada: list[str],
    passos_entrada: int,
    horizonte_previsao: int = 1,
) -> dict[str, np.ndarray | list[str] | str | int]:
    """
    Converte a base tabular em dataset supervisionado sequencial.

    Cada amostra de entrada possui o formato:
        [passos_entrada, n_features]

    O alvo corresponde ao valor da coluna alvo em:
        t + horizonte_previsao
    considerando o fim da janela de entrada como referência.
    """
    if passos_entrada <= 0:
        raise ValueError("'passos_entrada' deve ser maior que zero.")

    if horizonte_previsao <= 0:
        raise ValueError("'horizonte_previsao' deve ser maior que zero.")

    valores_entrada = df[colunas_entrada].to_numpy(dtype=np.float32)
    valores_alvo = df[coluna_alvo].to_numpy(dtype=np.float32)
    datas = df.index

    quantidade_amostras = len(df) - passos_entrada - horizonte_previsao + 1

    if quantidade_amostras <= 0:
        raise ValueError(
            "A base não possui linhas suficientes para gerar sequências com "
            f"passos_entrada={passos_entrada} e horizonte_previsao={horizonte_previsao}."
        )

    lista_X = []
    lista_y = []
    lista_data_inicio_janela = []
    lista_data_fim_janela = []
    lista_data_alvo = []

    for indice_inicial in range(quantidade_amostras):
        indice_fim_janela = indice_inicial + passos_entrada
        indice_alvo = indice_fim_janela + horizonte_previsao - 1

        sequencia_entrada = valores_entrada[indice_inicial:indice_fim_janela, :]
        valor_alvo = valores_alvo[indice_alvo]

        lista_X.append(sequencia_entrada)
        lista_y.append(valor_alvo)

        lista_data_inicio_janela.append(str(datas[indice_inicial].date()))
        lista_data_fim_janela.append(str(datas[indice_fim_janela - 1].date()))
        lista_data_alvo.append(str(datas[indice_alvo].date()))

    indice_alvo_nas_entradas = (
        colunas_entrada.index(coluna_alvo)
        if coluna_alvo in colunas_entrada
        else None
    )
    return {
        "X": np.array(lista_X, dtype=np.float32),
        "y": np.array(lista_y, dtype=np.float32),
        "datas_inicio_janela": np.array(lista_data_inicio_janela),
        "datas_fim_janela": np.array(lista_data_fim_janela),
        "datas_alvo": np.array(lista_data_alvo),
        "colunas_entrada": np.array(colunas_entrada),
        "coluna_alvo": coluna_alvo,
        "indice_alvo_nas_entradas": indice_alvo_nas_entradas,
        "passos_entrada": passos_entrada,
        "horizonte_previsao": horizonte_previsao,
    }


def montar_dataset_sequencial(
    df_features: pd.DataFrame,
    coluna_alvo: str = "Nivel",
    usar_todas_as_colunas_como_entrada: bool = True,
    colunas_entrada: Iterable[str] | None = None,
    incluir_alvo_nas_entradas: bool = True,
    passos_entrada: int = 180,
    horizonte_previsao: int = 1,
) -> dict[str, np.ndarray | list[str] | str | int]:
    """
    Executa a Fase 3A canônica do Caso 2.

    Etapas:
    1. valida a base
    2. define colunas de entrada
    3. gera as sequências supervisionadas
    """
    validar_base_sequencial(
        df=df_features,
        coluna_alvo=coluna_alvo,
    )

    colunas_modelagem = definir_colunas_entrada(
        df=df_features,
        coluna_alvo=coluna_alvo,
        usar_todas_as_colunas_como_entrada=usar_todas_as_colunas_como_entrada,
        colunas_entrada=colunas_entrada,
        incluir_alvo_nas_entradas=incluir_alvo_nas_entradas,
    )

    dataset_sequencial = gerar_sequencias_supervisionadas(
        df=df_features,
        coluna_alvo=coluna_alvo,
        colunas_entrada=colunas_modelagem,
        passos_entrada=passos_entrada,
        horizonte_previsao=horizonte_previsao,
    )

    return dataset_sequencial