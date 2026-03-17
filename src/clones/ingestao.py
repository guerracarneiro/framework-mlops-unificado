from __future__ import annotations

"""
Módulo de ingestão de dados do Estudo de Caso 1.

Responsável por:
- carregar o dataset bruto a partir de arquivo Excel;
- separar colunas de identificação das colunas utilizadas na modelagem.

Este módulo não realiza transformações nos dados.
Qualquer etapa de limpeza, filtragem ou engenharia de atributos
é tratado no módulo de preprocessamento.
"""

from pathlib import Path
from typing import Tuple

import pandas as pd


def carregar_dataset_excel(
    caminho_excel: str | Path,
    aba: str,
) -> pd.DataFrame:
    """
    Carrega um dataset a partir de um arquivo Excel.

    Parâmetros
    ----------
    caminho_excel : str | Path
        Caminho do arquivo Excel.
    aba : str
        Nome da aba a ser lida.

    Retorno
    -------
    pd.DataFrame
        DataFrame contendo os dados carregados.

    Exceções
    --------
    FileNotFoundError
        Quando o arquivo não é encontrado.
    ValueError
        Quando a leitura do Excel falha.
    """
    caminho_excel = Path(caminho_excel)

    if not caminho_excel.exists():
        raise FileNotFoundError(f"Arquivo não encontrado: {caminho_excel}")

    try:
        df = pd.read_excel(
            caminho_excel,
            sheet_name=aba,
            engine="openpyxl",
        )
    except Exception as e:
        raise ValueError(f"Erro ao ler o arquivo Excel: {e}")

    if df.empty:
        raise ValueError("O dataset carregado está vazio.")

    return df


def separar_colunas_identificacao(
    df: pd.DataFrame,
    colunas_id: list[str],
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Separa colunas de identificação das colunas de modelagem.

    As colunas de identificação são mantidas para rastreabilidade,
    mas não são utilizadas diretamente na construção da matriz de features.

    Parâmetros
    ----------
    df : pd.DataFrame
        DataFrame original.
    colunas_id : list[str]
        Lista de colunas consideradas como identificadores.

    Retorno
    -------
    Tuple[pd.DataFrame, pd.DataFrame]
        - DataFrame contendo apenas colunas de identificação
        - DataFrame contendo apenas colunas de modelagem
    """
    colunas_existentes = [col for col in colunas_id if col in df.columns]

    df_id = df[colunas_existentes].copy()
    df_modelagem = df.drop(columns=colunas_existentes, errors="ignore").copy()

    return df_id, df_modelagem


def selecionar_colunas_modelagem(
    df: pd.DataFrame,
    colunas_excluir: list[str] | None = None,
) -> pd.DataFrame:
    """
    Remove colunas específicas antes do preprocessamento.

    Esta função é útil para eliminar colunas que não devem participar
    da modelagem, como identificadores redundantes ou atributos irrelevantes.

    Parâmetros
    ----------
    df : pd.DataFrame
        DataFrame de entrada.
    colunas_excluir : list[str] | None
        Lista de colunas a serem removidas.

    Retorno
    -------
    pd.DataFrame
        DataFrame com colunas removidas.
    """
    if not colunas_excluir:
        return df.copy()

    colunas_validas = [col for col in colunas_excluir if col in df.columns]

    df_filtrado = df.drop(columns=colunas_validas, errors="ignore").copy()

    return df_filtrado


def converter_para_numerico(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Converte colunas para tipo numérico sempre que possível.

    Valores inválidos são convertidos para NaN, permitindo posterior imputação.

    Parâmetros
    ----------
    df : pd.DataFrame
        DataFrame de entrada.

    Retorno
    -------
    pd.DataFrame
        DataFrame com colunas convertidas para tipo numérico quando aplicável.
    """
    df_convertido = df.copy()

    for coluna in df_convertido.columns:
        df_convertido[coluna] = pd.to_numeric(
            df_convertido[coluna],
            errors="ignore",
        )

    return df_convertido