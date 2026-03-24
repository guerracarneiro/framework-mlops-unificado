from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable

import json

import pandas as pd
import yaml


def garantir_pasta(caminho: str | Path) -> Path:
    """
    Garante que a pasta exista e retorna o caminho como Path.
    """
    caminho = Path(caminho)
    caminho.mkdir(parents=True, exist_ok=True)
    return caminho


def ler_yaml(caminho_arquivo: str | Path) -> dict[str, Any]:
    """
    Lê um arquivo YAML e retorna seu conteúdo como dicionário.
    """
    caminho_arquivo = Path(caminho_arquivo)

    if not caminho_arquivo.exists():
        raise FileNotFoundError(f"Arquivo YAML não encontrado: {caminho_arquivo}")

    with open(caminho_arquivo, "r", encoding="utf-8") as arquivo:
        conteudo = yaml.safe_load(arquivo)

    if conteudo is None:
        return {}

    if not isinstance(conteudo, dict):
        raise ValueError(
            f"O arquivo YAML deve conter um dicionário na raiz: {caminho_arquivo}"
        )

    return conteudo


def salvar_json(dados: dict[str, Any], caminho_arquivo: str | Path) -> Path:
    """
    Salva um dicionário em arquivo JSON.
    """
    caminho_arquivo = Path(caminho_arquivo)
    garantir_pasta(caminho_arquivo.parent)

    with open(caminho_arquivo, "w", encoding="utf-8") as arquivo:
        json.dump(dados, arquivo, ensure_ascii=False, indent=4)

    return caminho_arquivo


def ler_planilha_estruturada(
    caminho_arquivo: str | Path,
    nome_planilha: str,
    usecols: Iterable[int | str],
    nomes_colunas: list[str],
    skiprows: int = 0,
    engine: str = "openpyxl",
) -> pd.DataFrame:
    """
    Lê uma planilha Excel/XLSM com estrutura conhecida e aplica
    nomes padronizados às colunas.
    """
    caminho_arquivo = Path(caminho_arquivo)

    if not caminho_arquivo.exists():
        raise FileNotFoundError(f"Arquivo não encontrado: {caminho_arquivo}")

    return pd.read_excel(
        caminho_arquivo,
        sheet_name=nome_planilha,
        usecols=list(usecols),
        names=nomes_colunas,
        skiprows=skiprows,
        engine=engine,
    )


def converter_datas(
    serie_datas: pd.Series,
    formato_preferencial: str | None = None,
    dayfirst: bool = True,
) -> pd.Series:
    """
    Converte uma série para datetime usando primeiro um formato
    preferencial e depois fallback genérico.
    """
    datas_texto = serie_datas.astype(str).str.strip()

    if formato_preferencial is not None:
        datas_convertidas = pd.to_datetime(
            datas_texto,
            format=formato_preferencial,
            errors="coerce",
        )
    else:
        datas_convertidas = pd.to_datetime(
            datas_texto,
            errors="coerce",
            dayfirst=dayfirst,
        )

    mascara_invalidas = datas_convertidas.isna()

    if mascara_invalidas.any():
        datas_convertidas.loc[mascara_invalidas] = pd.to_datetime(
            datas_texto[mascara_invalidas],
            errors="coerce",
            dayfirst=dayfirst,
        )

    return datas_convertidas


def converter_numerico_flexivel(serie_valores: pd.Series) -> pd.Series:
    """
    Converte uma série para numérico, tratando valores com vírgula decimal,
    ponto decimal e valores vazios.
    """
    serie_texto = serie_valores.astype(str).str.strip()

    serie_texto = serie_texto.replace(
        {
            "": pd.NA,
            "nan": pd.NA,
            "NaN": pd.NA,
            "None": pd.NA,
            "-": pd.NA,
        }
    )

    mascara_virgula = serie_texto.str.contains(",", na=False)

    serie_texto.loc[mascara_virgula] = (
        serie_texto.loc[mascara_virgula]
        .str.replace(".", "", regex=False)
        .str.replace(",", ".", regex=False)
    )

    return pd.to_numeric(serie_texto, errors="coerce")