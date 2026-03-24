from __future__ import annotations

from typing import Iterable

import numpy as np
import pandas as pd

from src.river_level.preprocessamento import validar_indice_temporal


def validar_colunas_minimas(
    df: pd.DataFrame,
    coluna_nivel: str = "Nivel",
    coluna_precipitacao: str = "Precip_Media_Estacoes",
) -> None:
    """
    Valida se a base possui as colunas mínimas esperadas para a
    engenharia de features do Caso 2.
    """
    validar_indice_temporal(df)

    colunas_obrigatorias = [coluna_nivel, coluna_precipitacao]
    colunas_ausentes = [col for col in colunas_obrigatorias if col not in df.columns]

    if colunas_ausentes:
        raise ValueError(
            "A base de entrada não possui todas as colunas mínimas esperadas. "
            f"Colunas ausentes: {colunas_ausentes}"
        )


def calcular_api(
    serie_precipitacao: pd.Series,
    fator_k: float = 0.92,
) -> pd.Series:
    """
    Calcula o Índice de Precipitação Antecedente (API).

    A implementação preserva a lógica observada no notebook original,
    mantendo a atualização iterativa baseada na precipitação do passo
    anterior e no acúmulo com decaimento exponencial.

    Parâmetros
    ----------
    serie_precipitacao : pd.Series
        Série de precipitação na escala original.
    fator_k : float
        Fator de decaimento do API.

    Retorno
    -------
    pd.Series
        Série do API alinhada ao índice temporal da precipitação.
    """
    if serie_precipitacao.empty:
        return pd.Series(dtype=float, index=serie_precipitacao.index, name="API")

    valores_api = np.zeros(len(serie_precipitacao), dtype=float)
    valores_precipitacao = serie_precipitacao.astype(float)

    valores_api[0] = valores_precipitacao.iloc[0]

    for indice in range(1, len(valores_precipitacao)):
        valores_api[indice] = (
            valores_precipitacao.iloc[indice - 1]
            + fator_k * valores_api[indice - 1]
        )

    return pd.Series(
        valores_api,
        index=serie_precipitacao.index,
        name="API",
    )


def adicionar_precipitacao_logaritmica(
    df: pd.DataFrame,
    coluna_precipitacao: str = "Precip_Media_Estacoes",
    nome_coluna_saida: str = "Precip_Log",
) -> pd.DataFrame:
    """
    Gera a transformação logarítmica da precipitação usando log(1 + x),
    conforme a lógica do notebook original.
    """
    df_saida = df.copy()
    df_saida[nome_coluna_saida] = np.log1p(df_saida[coluna_precipitacao].astype(float))
    return df_saida


def adicionar_api(
    df: pd.DataFrame,
    coluna_precipitacao: str = "Precip_Media_Estacoes",
    fator_k_api: float = 0.92,
    nome_coluna_saida: str = "API",
) -> pd.DataFrame:
    """
    Adiciona a coluna de API calculada a partir da precipitação diária.
    """
    df_saida = df.copy()
    df_saida[nome_coluna_saida] = calcular_api(
        serie_precipitacao=df_saida[coluna_precipitacao],
        fator_k=fator_k_api,
    )
    return df_saida


def adicionar_contadores_estiagem(
    df: pd.DataFrame,
    limiares_estiagem: Iterable[int | float],
    coluna_precipitacao: str = "Precip_Media_Estacoes",
) -> pd.DataFrame:
    """
    Adiciona contadores de dias consecutivos com precipitação abaixo
    dos limiares informados.

    A lógica preserva o comportamento do notebook original.
    """
    df_saida = df.copy()

    for limiar in limiares_estiagem:
        sem_chuva = df_saida[coluna_precipitacao] < limiar
        grupos_estiagem = (~sem_chuva).cumsum()
        nome_coluna = f"Dias_Estiagem_<{limiar}mm"
        df_saida[nome_coluna] = sem_chuva.groupby(grupos_estiagem).cumsum()

    return df_saida


def adicionar_features_nivel(
    df: pd.DataFrame,
    janelas_nivel: Iterable[int],
    coluna_nivel: str = "Nivel",
) -> pd.DataFrame:
    """
    Adiciona médias móveis e tendência do nível do rio.

    A média móvel utiliza janela temporal em dias para manter aderência
    ao notebook original.
    """
    df_saida = df.copy()

    for dias in janelas_nivel:
        df_saida[f"Nivel_Media_{dias}d"] = df_saida[coluna_nivel].rolling(
            window=f"{dias}D"
        ).mean()

        df_saida[f"Nivel_Tendencia_{dias}d"] = df_saida[coluna_nivel].diff(
            periods=dias
        )

    return df_saida


def adicionar_defasagens(
    df: pd.DataFrame,
    colunas_para_defasagem: Iterable[str],
    defasagens: Iterable[int],
) -> pd.DataFrame:
    """
    Adiciona defasagens explícitas para as colunas informadas.
    """
    df_saida = df.copy()

    colunas_para_defasagem = list(colunas_para_defasagem)
    colunas_ausentes = [col for col in colunas_para_defasagem if col not in df_saida.columns]

    if colunas_ausentes:
        raise ValueError(
            "Nem todas as colunas solicitadas para defasagem existem na base. "
            f"Colunas ausentes: {colunas_ausentes}"
        )

    for coluna in colunas_para_defasagem:
        for lag in defasagens:
            df_saida[f"{coluna}_Lag_{lag}d"] = df_saida[coluna].shift(lag)

    return df_saida


def adicionar_sazonalidade(
    df: pd.DataFrame,
    usar_ano_bissexto: bool = True,
) -> pd.DataFrame:
    """
    Adiciona codificação sazonal com seno e cosseno do dia do ano.
    """
    df_saida = df.copy()

    periodo = 365.25 if usar_ano_bissexto else 365.0
    dia_do_ano = df_saida.index.dayofyear

    df_saida["Seno_DiaAno"] = np.sin(2 * np.pi * dia_do_ano / periodo)
    df_saida["Cosseno_DiaAno"] = np.cos(2 * np.pi * dia_do_ano / periodo)

    return df_saida


def remover_registros_incompletos(df: pd.DataFrame) -> pd.DataFrame:
    """
    Remove linhas com valores ausentes resultantes da geração das features.
    """
    return df.dropna().copy()


def gerar_dataset_features(
    df_base_modelagem: pd.DataFrame,
    janelas_nivel: Iterable[int],
    defasagens: Iterable[int],
    limiares_estiagem: Iterable[int | float],
    fator_k_api: float = 0.92,
    coluna_nivel: str = "Nivel",
    coluna_precipitacao: str = "Precip_Media_Estacoes",
    remover_nans_finais: bool = True,
) -> pd.DataFrame:
    """
    Executa a geração canônica de features da Fase 2 do Caso 2.

    Ordem aplicada:
    1. valida base
    2. precipitação logarítmica
    3. API
    4. estiagem
    5. médias móveis e tendência do nível
    6. defasagens
    7. sazonalidade
    8. remoção final de NaN

    Retorno
    -------
    pd.DataFrame
        Base de features pronta para a próxima fase.
    """
    validar_colunas_minimas(
        df=df_base_modelagem,
        coluna_nivel=coluna_nivel,
        coluna_precipitacao=coluna_precipitacao,
    )

    df_features = df_base_modelagem.copy().sort_index()

    df_features = adicionar_precipitacao_logaritmica(
        df=df_features,
        coluna_precipitacao=coluna_precipitacao,
        nome_coluna_saida="Precip_Log",
    )

    df_features = adicionar_api(
        df=df_features,
        coluna_precipitacao=coluna_precipitacao,
        fator_k_api=fator_k_api,
        nome_coluna_saida="API",
    )

    df_features = adicionar_contadores_estiagem(
        df=df_features,
        limiares_estiagem=limiares_estiagem,
        coluna_precipitacao=coluna_precipitacao,
    )

    df_features = adicionar_features_nivel(
        df=df_features,
        janelas_nivel=janelas_nivel,
        coluna_nivel=coluna_nivel,
    )

    df_features = adicionar_defasagens(
        df=df_features,
        colunas_para_defasagem=[coluna_nivel, "Precip_Log", "API"],
        defasagens=defasagens,
    )

    df_features = adicionar_sazonalidade(
        df=df_features,
        usar_ano_bissexto=True,
    )

    if remover_nans_finais:
        df_features = remover_registros_incompletos(df_features)

    return df_features.sort_index()