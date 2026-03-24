from __future__ import annotations

import pandas as pd


def validar_indice_temporal(df: pd.DataFrame) -> None:
    """
    Valida se a base está indexada por data.

    Parâmetros
    ----------
    df : pd.DataFrame
        Base de entrada.

    Exceções
    --------
    ValueError
        Quando o índice não é do tipo datetime.
    """
    if not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError(
            "A base deve possuir um índice temporal do tipo DatetimeIndex."
        )


def filtrar_periodo_modelagem(
    df: pd.DataFrame,
    data_inicio: str | None = None,
    data_fim: str | None = None,
) -> pd.DataFrame:
    """
    Filtra a base pelo período de interesse usando o índice temporal.
    """
    validar_indice_temporal(df)

    df_filtrado = df.copy()

    if data_inicio is not None:
        df_filtrado = df_filtrado[df_filtrado.index >= pd.to_datetime(data_inicio)]

    if data_fim is not None:
        df_filtrado = df_filtrado[df_filtrado.index <= pd.to_datetime(data_fim)]

    return df_filtrado.sort_index()


def limitar_ate_ultimo_nivel_observado(
    df: pd.DataFrame,
    coluna_nivel: str = "Nivel",
) -> pd.DataFrame:
    """
    Limita a base até a última data com nível observado.
    """
    validar_indice_temporal(df)

    serie_nivel_valida = df.dropna(subset=[coluna_nivel])

    if serie_nivel_valida.empty:
        raise ValueError(
            f"A coluna '{coluna_nivel}' não possui valores válidos para limitar a base."
        )

    ultima_data_valida = serie_nivel_valida.index.max()
    df_limitado = df[df.index <= ultima_data_valida].copy()

    return df_limitado.sort_index()


def tratar_faltantes_iniciais(
    df: pd.DataFrame,
    colunas_tratar: list[str] | None = None,
) -> pd.DataFrame:
    """
    Aplica o tratamento inicial de faltantes reproduzindo o comportamento
    do notebook original: interpolação temporal, forward fill e backward fill.
    """
    validar_indice_temporal(df)

    df_tratado = df.copy().sort_index()

    colunas_tratar = colunas_tratar or ["Nivel", "Precip_Media_Estacoes"]

    for coluna in colunas_tratar:
        if coluna not in df_tratado.columns:
            continue

        df_tratado[coluna] = df_tratado[coluna].interpolate(method="time")
        df_tratado[coluna] = df_tratado[coluna].ffill()
        df_tratado[coluna] = df_tratado[coluna].bfill()

    return df_tratado


def tratar_evento_anomalo_novembro_2015(
    df: pd.DataFrame,
    coluna_nivel: str = "Nivel",
    data_inicio: str = "2015-11-05",
    data_fim: str = "2015-11-18",
) -> pd.DataFrame:
    """
    Trata o evento anômalo de novembro de 2015 reproduzindo o notebook original:
    substitui o nível por nulo no intervalo e reinterpola a série.
    """
    validar_indice_temporal(df)

    df_tratado = df.copy().sort_index()

    df_tratado.loc[data_inicio:data_fim, coluna_nivel] = pd.NA
    df_tratado[coluna_nivel] = df_tratado[coluna_nivel].interpolate(method="time")
    df_tratado[coluna_nivel] = df_tratado[coluna_nivel].ffill()
    df_tratado[coluna_nivel] = df_tratado[coluna_nivel].bfill()

    return df_tratado


def preparar_base_modelagem(
    df: pd.DataFrame,
    data_inicio: str | None = None,
    data_fim: str | None = None,
    tratar_evento_2015: bool = True,
    coluna_nivel: str = "Nivel",
    colunas_tratar: list[str] | None = None,
) -> pd.DataFrame:
    """
    Executa o fluxo mínimo de preparação da base para modelagem, alinhado
    ao comportamento do notebook original.

    Etapas:
    1. Ordena por data
    2. Filtra o período de modelagem
    3. Limita até o último nível observado
    4. Trata faltantes iniciais
    5. Trata o evento anômalo de novembro de 2015
    """
    validar_indice_temporal(df)

    df_modelagem = df.copy().sort_index()

    df_modelagem = filtrar_periodo_modelagem(
        df=df_modelagem,
        data_inicio=data_inicio,
        data_fim=data_fim,
    )

    df_modelagem = limitar_ate_ultimo_nivel_observado(
        df=df_modelagem,
        coluna_nivel=coluna_nivel,
    )

    df_modelagem = tratar_faltantes_iniciais(
        df=df_modelagem,
        colunas_tratar=colunas_tratar,
    )

    if tratar_evento_2015:
        df_modelagem = tratar_evento_anomalo_novembro_2015(
            df=df_modelagem,
            coluna_nivel=coluna_nivel,
        )

    return df_modelagem.sort_index()