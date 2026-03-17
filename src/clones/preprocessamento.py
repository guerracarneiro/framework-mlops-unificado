from __future__ import annotations

"""
Módulo de preprocessamento do Estudo de Caso 1.

Responsável por:
- aplicar padronizações em colunas categóricas;
- filtrar registros conforme regras do experimento;
- aplicar codificação one-hot;
- aplicar estratégias de imputação;
- aplicar normalização dos dados numéricos.

Este módulo foi estruturado para apoiar a Fase 1 do estudo,
na qual diferentes estratégias de preprocessamento serão comparadas
de forma rastreável e reproduzível.
"""

from typing import Any

import numpy as np
import pandas as pd
from sklearn.impute import KNNImputer, SimpleImputer
from sklearn.preprocessing import StandardScaler


def aplicar_padronizacao_especie(
    df: pd.DataFrame,
    especie_replace: dict[str, str] | None = None,
    coluna_especie: str = "ESPECIE",
) -> pd.DataFrame:
    """
    Aplica padronização dos valores da coluna de espécie.

    Parâmetros
    ----------
    df : pd.DataFrame
        DataFrame de entrada.
    especie_replace : dict[str, str] | None
        Dicionário com substituições a serem aplicadas.
    coluna_especie : str
        Nome da coluna de espécie.

    Retorno
    -------
    pd.DataFrame
        DataFrame com padronização aplicada.
    """
    df_saida = df.copy()

    if not especie_replace:
        return df_saida

    if coluna_especie not in df_saida.columns:
        return df_saida

    df_saida[coluna_especie] = df_saida[coluna_especie].replace(especie_replace)

    return df_saida


def filtrar_especies(
    df: pd.DataFrame,
    especies_para_excluir: list[str] | None = None,
    coluna_especie: str = "ESPECIE",
) -> pd.DataFrame:
    """
    Remove registros pertencentes às espécies informadas para exclusão.

    Parâmetros
    ----------
    df : pd.DataFrame
        DataFrame de entrada.
    especies_para_excluir : list[str] | None
        Lista de espécies a serem removidas.
    coluna_especie : str
        Nome da coluna de espécie.

    Retorno
    -------
    pd.DataFrame
        DataFrame filtrado.
    """
    df_saida = df.copy()

    if not especies_para_excluir:
        return df_saida

    if coluna_especie not in df_saida.columns:
        return df_saida

    df_saida = df_saida[~df_saida[coluna_especie].isin(especies_para_excluir)].copy()

    return df_saida.reset_index(drop=True)


def converter_coluna_idade_para_numerico(
    df: pd.DataFrame,
    coluna_idade: str = "IDADE",
) -> pd.DataFrame:
    """
    Converte a coluna de idade para formato numérico.

    A função trata cenários em que a coluna venha como texto com vírgula decimal.

    Parâmetros
    ----------
    df : pd.DataFrame
        DataFrame de entrada.
    coluna_idade : str
        Nome da coluna de idade.

    Retorno
    -------
    pd.DataFrame
        DataFrame com a coluna de idade convertida quando aplicável.
    """
    df_saida = df.copy()

    if coluna_idade not in df_saida.columns:
        return df_saida

    serie_idade = df_saida[coluna_idade]

    if pd.api.types.is_numeric_dtype(serie_idade):
        df_saida[coluna_idade] = pd.to_numeric(serie_idade, errors="coerce")
        return df_saida

    serie_idade = (
        serie_idade.astype(str)
        .str.strip()
        .str.replace(",", ".", regex=False)
    )

    df_saida[coluna_idade] = pd.to_numeric(serie_idade, errors="coerce")

    return df_saida


def filtrar_idade(
    df: pd.DataFrame,
    idade_min: float | None = None,
    idade_max: float | None = None,
    coluna_idade: str = "IDADE",
) -> pd.DataFrame:
    """
    Filtra registros com base em intervalo mínimo e máximo da idade.

    Parâmetros
    ----------
    df : pd.DataFrame
        DataFrame de entrada.
    idade_min : float | None
        Limite inferior de idade.
    idade_max : float | None
        Limite superior de idade.
    coluna_idade : str
        Nome da coluna de idade.

    Retorno
    -------
    pd.DataFrame
        DataFrame filtrado por idade.
    """
    df_saida = df.copy()

    if coluna_idade not in df_saida.columns:
        return df_saida

    df_saida = converter_coluna_idade_para_numerico(
        df_saida,
        coluna_idade=coluna_idade,
    )

    df_saida = df_saida.dropna(subset=[coluna_idade]).copy()

    if idade_min is not None:
        df_saida = df_saida[df_saida[coluna_idade] >= idade_min].copy()

    if idade_max is not None:
        df_saida = df_saida[df_saida[coluna_idade] <= idade_max].copy()

    return df_saida.reset_index(drop=True)


def aplicar_filtros_base(
    df: pd.DataFrame,
    especie_replace: dict[str, str] | None = None,
    especies_para_excluir: list[str] | None = None,
    idade_min: float | None = None,
    idade_max: float | None = None,
) -> pd.DataFrame:
    """
    Aplica o conjunto básico de filtros e padronizações do experimento.

    Etapas aplicadas:
    - padronização da coluna de espécie;
    - remoção de espécies indesejadas;
    - filtro por faixa de idade.

    Parâmetros
    ----------
    df : pd.DataFrame
        DataFrame de entrada.
    especie_replace : dict[str, str] | None
        Substituições da coluna de espécie.
    especies_para_excluir : list[str] | None
        Lista de espécies a excluir.
    idade_min : float | None
        Idade mínima permitida.
    idade_max : float | None
        Idade máxima permitida.

    Retorno
    -------
    pd.DataFrame
        DataFrame filtrado e padronizado.
    """
    df_saida = df.copy()

    df_saida = aplicar_padronizacao_especie(
        df_saida,
        especie_replace=especie_replace,
        coluna_especie="ESPECIE",
    )

    df_saida = filtrar_especies(
        df_saida,
        especies_para_excluir=especies_para_excluir,
        coluna_especie="ESPECIE",
    )

    df_saida = filtrar_idade(
        df_saida,
        idade_min=idade_min,
        idade_max=idade_max,
        coluna_idade="IDADE",
    )

    return df_saida.reset_index(drop=True)


def aplicar_onehot(
    df: pd.DataFrame,
    colunas: list[str] | None = None,
) -> pd.DataFrame:
    """
    Aplica codificação one-hot nas colunas categóricas informadas.

    Quando a lista de colunas estiver vazia, retorna um DataFrame vazio
    com o mesmo índice do DataFrame de entrada.

    Parâmetros
    ----------
    df : pd.DataFrame
        DataFrame de entrada.
    colunas : list[str] | None
        Lista de colunas a serem codificadas.

    Retorno
    -------
    pd.DataFrame
        DataFrame contendo apenas as colunas one-hot geradas.
    """
    if not colunas:
        return pd.DataFrame(index=df.index)

    colunas_validas = [col for col in colunas if col in df.columns]

    if not colunas_validas:
        return pd.DataFrame(index=df.index)

    df_categorico = df[colunas_validas].copy()

    df_saida = pd.get_dummies(
        df_categorico,
        columns=colunas_validas,
        drop_first=False,
        dtype=float,
    )

    return df_saida


def selecionar_colunas_numericas(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Seleciona apenas colunas numéricas do DataFrame.

    Retorno
    -------
    pd.DataFrame
        DataFrame contendo apenas colunas numéricas.
    """
    df_saida = df.select_dtypes(include=[np.number]).copy()
    return df_saida


def imputar_sem_imputacao(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Remove linhas contendo valores ausentes.

    Esta estratégia representa o cenário sem imputação explícita,
    mantendo apenas registros completos para a modelagem.

    Os índices originais são preservados para permitir alinhamento posterior
    com os identificadores das amostras.

    Parâmetros
    ----------
    df : pd.DataFrame
        DataFrame de entrada.

    Retorno
    -------
    pd.DataFrame
        DataFrame sem valores ausentes, preservando os índices originais.
    """
    df_saida = df.dropna(axis=0).copy()
    return df_saida


def imputar_mediana(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Aplica imputação simples por mediana nas colunas numéricas.

    Parâmetros
    ----------
    df : pd.DataFrame
        DataFrame de entrada contendo colunas numéricas.

    Retorno
    -------
    pd.DataFrame
        DataFrame imputado.
    """
    imputador = SimpleImputer(strategy="median")
    matriz_imputada = imputador.fit_transform(df)

    df_saida = pd.DataFrame(
        matriz_imputada,
        columns=df.columns,
        index=df.index,
    )

    return df_saida


def imputar_media70_mais_knn(
    df: pd.DataFrame,
    serie_grupo: pd.Series | None = None,
    limiar_preenchimento_grupo: float = 0.70,
    n_vizinhos_knn: int = 5,
) -> pd.DataFrame:
    """
    Aplica imputação em duas etapas:
    1. preenche valores ausentes com média por grupo quando a cobertura do grupo
       atinge o limiar mínimo definido;
    2. aplica KNNImputer para valores ainda ausentes.

    A série de agrupamento é fornecida externamente, o que permite utilizar
    colunas de identificação sem misturá-las à matriz de modelagem.

    Parâmetros
    ----------
    df : pd.DataFrame
        DataFrame de entrada contendo apenas colunas numéricas.
    serie_grupo : pd.Series | None
        Série com o agrupamento das amostras, alinhada ao índice do DataFrame.
    limiar_preenchimento_grupo : float
        Cobertura mínima do grupo para uso da média do grupo.
    n_vizinhos_knn : int
        Número de vizinhos do KNNImputer.

    Retorno
    -------
    pd.DataFrame
        DataFrame imputado, preservando os índices originais.
    """
    df_saida = df.copy()

    if serie_grupo is None:
        imputador_knn = KNNImputer(n_neighbors=n_vizinhos_knn)
        matriz_imputada = imputador_knn.fit_transform(df_saida)

        return pd.DataFrame(
            matriz_imputada,
            columns=df_saida.columns,
            index=df_saida.index,
        )

    serie_grupo = serie_grupo.loc[df_saida.index]

    for coluna in df_saida.columns:
        if not pd.api.types.is_numeric_dtype(df_saida[coluna]):
            continue

        estatisticas_grupo = (
            pd.DataFrame(
                {
                    "grupo": serie_grupo,
                    "valor": df_saida[coluna],
                },
                index=df_saida.index,
            )
            .groupby("grupo")["valor"]
            .agg(["mean", "count", "size"])
            .rename(columns={"mean": "media_grupo", "count": "qtd_validos", "size": "qtd_total"})
        )

        estatisticas_grupo["cobertura"] = (
            estatisticas_grupo["qtd_validos"] / estatisticas_grupo["qtd_total"]
        )

        grupos_validos = estatisticas_grupo[
            estatisticas_grupo["cobertura"] >= limiar_preenchimento_grupo
        ]["media_grupo"]

        mascara_nan = df_saida[coluna].isna()

        if mascara_nan.any():
            medias_map = serie_grupo.map(grupos_validos)
            df_saida.loc[mascara_nan, coluna] = medias_map.loc[mascara_nan]

    imputador_knn = KNNImputer(n_neighbors=n_vizinhos_knn)
    matriz_imputada = imputador_knn.fit_transform(df_saida)

    df_imputado = pd.DataFrame(
        matriz_imputada,
        columns=df_saida.columns,
        index=df_saida.index,
    )

    return df_imputado


def aplicar_imputacao(
    df: pd.DataFrame,
    config_imputacao: dict[str, Any],
    serie_grupo: pd.Series | None = None,
) -> pd.DataFrame:
    """
    Aplica a estratégia de imputação definida na configuração.

    Estratégias suportadas:
    - nenhuma
    - mediana
    - media70_mais_knn

    Parâmetros
    ----------
    df : pd.DataFrame
        DataFrame de entrada.
    config_imputacao : dict[str, Any]
        Configuração da estratégia de imputação.
    serie_grupo : pd.Series | None
        Série de agrupamento utilizada na estratégia media70_mais_knn.

    Retorno
    -------
    pd.DataFrame
        DataFrame imputado.
    """
    tipo_imputacao = config_imputacao.get("tipo", "nenhuma")
    params = config_imputacao.get("params", {})

    if tipo_imputacao == "nenhuma":
        return imputar_sem_imputacao(df)

    if tipo_imputacao == "mediana":
        return imputar_mediana(df)

    if tipo_imputacao == "media70_mais_knn":
        return imputar_media70_mais_knn(
            df,
            serie_grupo=serie_grupo,
            limiar_preenchimento_grupo=params.get("limiar_preenchimento_grupo", 0.70),
            n_vizinhos_knn=params.get("n_vizinhos_knn", 5),
        )

    raise ValueError(f"Tipo de imputação não suportado: {tipo_imputacao}")


def normalizar_dados(
    df: pd.DataFrame,
    tipo: str = "standard",
) -> tuple[pd.DataFrame, Any]:
    """
    Aplica normalização nas colunas numéricas contínuas.

    Estratégias suportadas:
    - standard

    Parâmetros
    ----------
    df : pd.DataFrame
        DataFrame numérico de entrada.
    tipo : str
        Tipo de normalização.

    Retorno
    -------
    tuple[pd.DataFrame, Any]
        - DataFrame normalizado
        - objeto do normalizador ajustado
    """
    if tipo != "standard":
        raise ValueError(f"Tipo de normalização não suportado: {tipo}")

    normalizador = StandardScaler()
    matriz_normalizada = normalizador.fit_transform(df)

    df_normalizado = pd.DataFrame(
        matriz_normalizada,
        columns=df.columns,
        index=df.index,
    )

    return df_normalizado, normalizador


def concatenar_blocos_modelagem(
    df_numerico: pd.DataFrame,
    df_onehot: pd.DataFrame,
) -> pd.DataFrame:
    """
    Concatena a base numérica tratada com as colunas one-hot.

    A concatenação preserva o índice e mantém as colunas one-hot sem normalização,
    reproduzindo a lógica do projeto original.

    Parâmetros
    ----------
    df_numerico : pd.DataFrame
        Base numérica já imputada e normalizada.
    df_onehot : pd.DataFrame
        Base one-hot gerada a partir das colunas categóricas selecionadas.

    Retorno
    -------
    pd.DataFrame
        Matriz final de features para modelagem.
    """
    if df_onehot.empty:
        return df_numerico.copy()

    df_saida = pd.concat(
        [
            df_numerico,
            df_onehot.loc[df_numerico.index].copy(),
        ],
        axis=1,
    )

    return df_saida


def alinhar_identificadores_apos_imputacao(
    df_id: pd.DataFrame,
    df_modelagem_resultante: pd.DataFrame,
) -> pd.DataFrame:
    """
    Alinha o DataFrame de identificação ao resultado final do preprocessamento.

    Esta função utiliza os índices preservados no DataFrame resultante para
    recuperar corretamente os identificadores correspondentes às amostras
    mantidas após a imputação.

    Parâmetros
    ----------
    df_id : pd.DataFrame
        DataFrame de identificação original.
    df_modelagem_resultante : pd.DataFrame
        DataFrame de modelagem após preprocessamento.

    Retorno
    -------
    pd.DataFrame
        DataFrame de identificação alinhado ao conjunto final.
    """
    indices_resultantes = df_modelagem_resultante.index
    df_id_alinhado = df_id.loc[indices_resultantes].copy()

    return df_id_alinhado


def executar_preprocessamento_fase1(
    df_modelagem: pd.DataFrame,
    config_preprocessamento: dict[str, Any],
    serie_grupo_imputacao: pd.Series | None = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """
    Executa o preprocessamento da Fase 1 e retorna a matriz final para modelagem.

    Etapas aplicadas:
    - separação da base numérica;
    - imputação da base numérica;
    - normalização da base numérica;
    - geração das colunas one-hot em paralelo;
    - concatenação final dos blocos.

    Os índices originais são preservados ao longo de todo o fluxo para permitir
    alinhamento correto com o DataFrame de identificação.

    Parâmetros
    ----------
    df_modelagem : pd.DataFrame
        DataFrame de modelagem após aplicação dos filtros base.
    config_preprocessamento : dict[str, Any]
        Configuração do preprocessamento da execução.
    serie_grupo_imputacao : pd.Series | None
        Série utilizada como agrupamento na estratégia media70_mais_knn.

    Retorno
    -------
    tuple[pd.DataFrame, dict[str, Any]]
        - matriz final de features
        - metadados resumidos do preprocessamento
    """
    df_trabalho = df_modelagem.copy()

    colunas_onehot = config_preprocessamento.get("onehot", {}).get("colunas", [])

    # Bloco numérico: apenas colunas numéricas da base original.
    df_numerico = selecionar_colunas_numericas(df_trabalho)

    if serie_grupo_imputacao is not None:
        serie_grupo_imputacao = serie_grupo_imputacao.loc[df_numerico.index]

    config_imputacao = config_preprocessamento.get("imputacao", {"tipo": "nenhuma"})
    df_numerico_imputado = aplicar_imputacao(
        df_numerico,
        config_imputacao=config_imputacao,
        serie_grupo=serie_grupo_imputacao,
    )

    tipo_normalizacao = config_preprocessamento.get("normalizacao", {}).get("tipo", "standard")
    df_numerico_normalizado, normalizador = normalizar_dados(
        df_numerico_imputado,
        tipo=tipo_normalizacao,
    )

    # Bloco categórico: one-hot gerado separadamente e sem normalização.
    df_onehot = aplicar_onehot(df_trabalho, colunas=colunas_onehot)

    # Alinha as colunas one-hot ao subconjunto final após eventual remoção de linhas.
    if not df_onehot.empty:
        df_onehot = df_onehot.loc[df_numerico_normalizado.index].copy()

    df_features = concatenar_blocos_modelagem(
        df_numerico=df_numerico_normalizado,
        df_onehot=df_onehot,
    )

    metadados = {
        "n_linhas_entrada": len(df_modelagem),
        "n_linhas_saida": len(df_features),
        "n_colunas_saida": df_features.shape[1],
        "n_colunas_numericas": df_numerico_normalizado.shape[1],
        "n_colunas_onehot": df_onehot.shape[1],
        "imputacao_tipo": config_imputacao.get("tipo", "nenhuma"),
        "onehot_colunas": colunas_onehot,
        "normalizacao_tipo": tipo_normalizacao,
    }

    return df_features, metadados