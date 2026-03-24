from __future__ import annotations

from pathlib import Path
from typing import Iterable

import pandas as pd

from src.utils.io_utils import (
    converter_datas,
    converter_numerico_flexivel,
    ler_planilha_estruturada,
)


def carregar_dados_nivel(
    caminho_arquivo: str | Path,
    nome_planilha: str,
    usecols: list[int] | None = None,
    skiprows: int = 37,
    coluna_data: str = "Data",
    coluna_nivel: str = "Nivel",
) -> pd.DataFrame:
    """
    Lê a planilha de nível do rio de forma fiel ao comportamento validado
    diretamente no Excel.

    Esta leitura é mantida mais explícita porque a comparação entre a leitura
    direta do arquivo e a leitura refatorada mostrou que a série de nível pode
    ser distorcida quando se tenta generalizar demais esta etapa.

    Parâmetros
    ----------
    caminho_arquivo : str | Path
        Caminho do arquivo Excel/XLSM.
    nome_planilha : str
        Nome da planilha com os dados de nível.
    usecols : list[int] | None
        Colunas a serem lidas.
    skiprows : int
        Quantidade de linhas iniciais a ignorar.
    coluna_data : str
        Nome padronizado da coluna de data.
    coluna_nivel : str
        Nome padronizado da coluna de nível.

    Retorno
    -------
    pd.DataFrame
        Base de nível com colunas padronizadas.
    """
    caminho_arquivo = Path(caminho_arquivo)

    if not caminho_arquivo.exists():
        raise FileNotFoundError(f"Arquivo não encontrado: {caminho_arquivo}")

    usecols = usecols or [0, 1]

    # Faz a leitura direta da planilha do mesmo modo validado no notebook.
    df_nivel = pd.read_excel(
        caminho_arquivo,
        sheet_name=nome_planilha,
        skiprows=skiprows,
        usecols=usecols,
        header=0,
        engine="openpyxl",
    )

    # Mantém apenas as duas primeiras colunas lidas e padroniza os nomes.
    df_nivel = df_nivel.iloc[:, :2].copy()
    df_nivel.columns = [coluna_data, coluna_nivel]

    # Converte a data preservando o comportamento do notebook.
    if not pd.api.types.is_datetime64_any_dtype(df_nivel[coluna_data]):
        df_nivel[coluna_data] = converter_datas(
            df_nivel[coluna_data],
            dayfirst=True,
        )
    else:
        df_nivel[coluna_data] = pd.to_datetime(
            df_nivel[coluna_data],
            errors="coerce",
        )

    # Só aplica conversão flexível se a coluna não vier numérica.
    if pd.api.types.is_numeric_dtype(df_nivel[coluna_nivel]):
        df_nivel[coluna_nivel] = pd.to_numeric(
            df_nivel[coluna_nivel],
            errors="coerce",
        )
    else:
        df_nivel[coluna_nivel] = converter_numerico_flexivel(
            df_nivel[coluna_nivel]
        )

    df_nivel = df_nivel.dropna(subset=[coluna_data]).copy()
    df_nivel = df_nivel.sort_values(coluna_data).reset_index(drop=True)

    return df_nivel


def carregar_dados_climaticos(
    caminho_arquivo: str | Path,
    nome_planilha: str,
    usecols: list[int] | None = None,
    nomes_colunas: list[str] | None = None,
    skiprows: int = 1,
    coluna_estacao: str = "Estacao",
    coluna_data: str = "Data",
    coluna_precipitacao: str = "Precipitacao",
    coluna_radiacao: str = "Radiacao",
    formato_data: str = "%Y-%m-%d",
) -> pd.DataFrame:
    """
    Lê a planilha climática e padroniza as colunas principais.

    Parâmetros
    ----------
    caminho_arquivo : str | Path
        Caminho do arquivo Excel/XLSM.
    nome_planilha : str
        Nome da planilha climática.
    usecols : list[int] | None
        Colunas a serem lidas.
    nomes_colunas : list[str] | None
        Nomes padronizados das colunas.
    skiprows : int
        Quantidade de linhas iniciais a ignorar.
    coluna_estacao : str
        Nome padronizado da coluna de estação.
    coluna_data : str
        Nome padronizado da coluna de data.
    coluna_precipitacao : str
        Nome padronizado da coluna de precipitação.
    coluna_radiacao : str
        Nome padronizado da coluna de radiação.
    formato_data : str
        Formato preferencial das datas da aba clima.

    Retorno
    -------
    pd.DataFrame
        Base climática padronizada.
    """
    usecols = usecols or [0, 1, 2, 3]
    nomes_colunas = nomes_colunas or [
        coluna_estacao,
        coluna_data,
        coluna_precipitacao,
        coluna_radiacao,
    ]

    df_clima = ler_planilha_estruturada(
        caminho_arquivo=caminho_arquivo,
        nome_planilha=nome_planilha,
        usecols=usecols,
        nomes_colunas=nomes_colunas,
        skiprows=skiprows,
    )

    df_clima[coluna_data] = converter_datas(
        df_clima[coluna_data],
        formato_preferencial=formato_data,
        dayfirst=False,
    )

    df_clima[coluna_estacao] = converter_numerico_flexivel(
        df_clima[coluna_estacao]
    ).astype("Int64")

    df_clima[coluna_precipitacao] = converter_numerico_flexivel(
        df_clima[coluna_precipitacao]
    )

    df_clima[coluna_radiacao] = converter_numerico_flexivel(
        df_clima[coluna_radiacao]
    )

    df_clima = df_clima.dropna(subset=[coluna_data]).copy()
    df_clima = df_clima.sort_values([coluna_data, coluna_estacao]).reset_index(drop=True)

    return df_clima


def agregar_media_estacoes(
    df: pd.DataFrame,
    estacoes_interesse: Iterable[int | float],
    coluna_data: str,
    coluna_estacao: str,
    coluna_valor: str,
    nome_coluna_saida: str,
) -> pd.DataFrame:
    """
    Filtra as estações de interesse e calcula a média diária da variável informada.

    Parâmetros
    ----------
    df : pd.DataFrame
        Base climática padronizada.
    estacoes_interesse : Iterable[int | float]
        Estações utilizadas no cálculo da média.
    coluna_data : str
        Nome da coluna de data.
    coluna_estacao : str
        Nome da coluna de estação.
    coluna_valor : str
        Nome da variável climática a ser agregada.
    nome_coluna_saida : str
        Nome da coluna de saída.

    Retorno
    -------
    pd.DataFrame
        Base agregada por data.
    """
    estacoes_interesse = list(estacoes_interesse)

    df_filtrado = df[df[coluna_estacao].isin(estacoes_interesse)].copy()

    if df_filtrado.empty:
        raise ValueError(
            f"Nenhum dado encontrado para as estações informadas: {estacoes_interesse}"
        )

    df_pivot = df_filtrado.pivot_table(
        index=coluna_data,
        columns=coluna_estacao,
        values=coluna_valor,
        aggfunc="mean",
    )

    df_saida = pd.DataFrame(index=df_pivot.index)
    df_saida[nome_coluna_saida] = df_pivot.mean(axis=1, skipna=True)

    return df_saida.reset_index()


def consolidar_base_nivel_precipitacao(
    caminho_arquivo: str | Path,
    nome_planilha_nivel: str,
    nome_planilha_clima: str,
    estacoes_interesse: Iterable[int | float],
    skiprows_nivel: int = 37,
    skiprows_clima: int = 1,
    usecols_nivel: list[int] | None = None,
    usecols_clima: list[int] | None = None,
    coluna_data: str = "Data",
    coluna_nivel: str = "Nivel",
    coluna_estacao: str = "Estacao",
    coluna_precipitacao: str = "Precipitacao",
    coluna_radiacao: str = "Radiacao",
    nome_coluna_precipitacao_media: str = "Precip_Media_Estacoes",
) -> pd.DataFrame:
    """
    Consolida a base de nível com a precipitação média das estações selecionadas.

    O retorno já mantém a data como índice temporal, alinhando a estrutura ao
    comportamento do notebook original e facilitando as próximas etapas.

    Retorno
    -------
    pd.DataFrame
        Base consolidada com índice temporal.
    """
    df_nivel = carregar_dados_nivel(
        caminho_arquivo=caminho_arquivo,
        nome_planilha=nome_planilha_nivel,
        usecols=usecols_nivel,
        skiprows=skiprows_nivel,
        coluna_data=coluna_data,
        coluna_nivel=coluna_nivel,
    )

    df_clima = carregar_dados_climaticos(
        caminho_arquivo=caminho_arquivo,
        nome_planilha=nome_planilha_clima,
        usecols=usecols_clima,
        nomes_colunas=[coluna_estacao, coluna_data, coluna_precipitacao, coluna_radiacao],
        skiprows=skiprows_clima,
        coluna_estacao=coluna_estacao,
        coluna_data=coluna_data,
        coluna_precipitacao=coluna_precipitacao,
        coluna_radiacao=coluna_radiacao,
    )

    df_precip = agregar_media_estacoes(
        df=df_clima,
        estacoes_interesse=estacoes_interesse,
        coluna_data=coluna_data,
        coluna_estacao=coluna_estacao,
        coluna_valor=coluna_precipitacao,
        nome_coluna_saida=nome_coluna_precipitacao_media,
    )

    df_consolidado = pd.merge(
        df_nivel,
        df_precip,
        on=coluna_data,
        how="left",
    )

    df_consolidado = df_consolidado.sort_values(coluna_data).reset_index(drop=True)
    df_consolidado = df_consolidado.set_index(coluna_data)

    return df_consolidado