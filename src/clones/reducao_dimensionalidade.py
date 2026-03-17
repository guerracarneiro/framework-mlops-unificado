from __future__ import annotations

"""
Módulo de redução de dimensionalidade do Estudo de Caso 1.

Responsável por:
- aplicar PCA;
- aplicar UMAP;
- selecionar dinamicamente o redutor configurado no experimento.

Este módulo foi estruturado para apoiar comparações entre diferentes
estratégias de redução de dimensionalidade de forma reproduzível.
"""

from typing import Any

import numpy as np
import pandas as pd
import umap
from sklearn.decomposition import PCA


def aplicar_pca(
    df_features: pd.DataFrame,
    n_components: int | float,
) -> tuple[np.ndarray, Any]:
    """
    Aplica redução de dimensionalidade com PCA.

    Parâmetros
    ----------
    df_features : pd.DataFrame
        DataFrame de entrada contendo apenas features numéricas.
    n_components : int | float
        Número de componentes ou fração da variância explicada desejada.

    Retorno
    -------
    tuple[np.ndarray, Any]
        - matriz reduzida
        - modelo PCA ajustado
    """
    modelo_pca = PCA(n_components=n_components)
    matriz_reduzida = modelo_pca.fit_transform(df_features)

    return matriz_reduzida, modelo_pca


def aplicar_umap(
    df_features: pd.DataFrame,
    n_neighbors: int = 15,
    n_components: int = 2,
    min_dist: float = 0.1,
    metric: str = "euclidean",
    random_state: int | None = 30,
) -> tuple[np.ndarray, Any]:
    """
    Aplica redução de dimensionalidade com UMAP.

    Parâmetros
    ----------
    df_features : pd.DataFrame
        DataFrame de entrada contendo apenas features numéricas.
    n_neighbors : int
        Número de vizinhos considerados pelo UMAP.
    n_components : int
        Número de dimensões do embedding de saída.
    min_dist : float
        Distância mínima entre pontos no embedding.
    metric : str
        Métrica utilizada no cálculo das distâncias.
    random_state : int | None
        Semente aleatória para reprodutibilidade.

    Retorno
    -------
    tuple[np.ndarray, Any]
        - matriz reduzida
        - modelo UMAP ajustado
    """
    modelo_umap = umap.UMAP(
        n_neighbors=n_neighbors,
        n_components=n_components,
        min_dist=min_dist,
        metric=metric,
        random_state=random_state,
    )

    matriz_reduzida = modelo_umap.fit_transform(df_features)

    return matriz_reduzida, modelo_umap


def aplicar_reducao_dimensionalidade(
    df_features: pd.DataFrame,
    config_reducer: dict[str, Any],
) -> tuple[np.ndarray, Any, dict[str, Any]]:
    """
    Aplica a estratégia de redução de dimensionalidade definida na configuração.

    Estratégias suportadas:
    - pca
    - umap

    Parâmetros
    ----------
    df_features : pd.DataFrame
        DataFrame de entrada contendo apenas features numéricas.
    config_reducer : dict[str, Any]
        Configuração do redutor.

    Retorno
    -------
    tuple[np.ndarray, Any, dict[str, Any]]
        - matriz reduzida
        - modelo ajustado
        - metadados resumidos da redução
    """
    nome_reducer = config_reducer.get("name", "").lower()
    params = config_reducer.get("params", {})

    if nome_reducer == "pca":
        n_components = params.get("n_components", 2)

        matriz_reduzida, modelo_reducer = aplicar_pca(
            df_features=df_features,
            n_components=n_components,
        )

        metadados = {
            "reducer_name": "pca",
            "n_amostras": matriz_reduzida.shape[0],
            "n_componentes_saida": matriz_reduzida.shape[1],
            "param_n_components": n_components,
        }

        if hasattr(modelo_reducer, "explained_variance_ratio_"):
            metadados["variancia_explicada_total"] = float(
                modelo_reducer.explained_variance_ratio_.sum()
            )

        return matriz_reduzida, modelo_reducer, metadados

    if nome_reducer == "umap":
        n_neighbors = params.get("n_neighbors", 15)
        n_components = params.get("n_components", 2)
        min_dist = params.get("min_dist", 0.1)
        metric = params.get("metric", "euclidean")
        random_state = params.get("random_state", 30)

        matriz_reduzida, modelo_reducer = aplicar_umap(
            df_features=df_features,
            n_neighbors=n_neighbors,
            n_components=n_components,
            min_dist=min_dist,
            metric=metric,
            random_state=random_state,
        )

        metadados = {
            "reducer_name": "umap",
            "n_amostras": matriz_reduzida.shape[0],
            "n_componentes_saida": matriz_reduzida.shape[1],
            "param_n_neighbors": n_neighbors,
            "param_n_components": n_components,
            "param_min_dist": min_dist,
            "param_metric": metric,
            "param_random_state": random_state,
        }

        return matriz_reduzida, modelo_reducer, metadados

    raise ValueError(f"Redutor não suportado: {nome_reducer}")


def converter_embedding_para_dataframe(
    matriz_reduzida: np.ndarray,
    prefixo_coluna: str = "dim",
) -> pd.DataFrame:
    """
    Converte a matriz reduzida em DataFrame com nomes padronizados de colunas.

    Parâmetros
    ----------
    matriz_reduzida : np.ndarray
        Matriz resultante da redução de dimensionalidade.
    prefixo_coluna : str
        Prefixo utilizado no nome das colunas.

    Retorno
    -------
    pd.DataFrame
        DataFrame contendo o embedding reduzido.
    """
    quantidade_componentes = matriz_reduzida.shape[1]
    nomes_colunas = [f"{prefixo_coluna}_{i + 1}" for i in range(quantidade_componentes)]

    df_embedding = pd.DataFrame(
        matriz_reduzida,
        columns=nomes_colunas,
    )

    return df_embedding