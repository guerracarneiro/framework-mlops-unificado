from __future__ import annotations

"""
Módulo de clusterização do Estudo de Caso 1.

Responsável por:
- aplicar KMeans;
- aplicar Agglomerative Clustering;
- aplicar HDBSCAN;
- selecionar dinamicamente o clusterizador configurado no experimento.

Este módulo foi estruturado para apoiar comparações entre diferentes
estratégias de agrupamento de forma reproduzível.
"""

from typing import Any

import hdbscan
import numpy as np
from sklearn.cluster import AgglomerativeClustering, KMeans


def aplicar_kmeans(
    matriz_reduzida: np.ndarray,
    n_clusters: int,
    random_state: int | None = 30,
    n_init: int = 10,
) -> tuple[np.ndarray, Any]:
    """
    Aplica clusterização com KMeans.

    Parâmetros
    ----------
    matriz_reduzida : np.ndarray
        Matriz de entrada após redução de dimensionalidade.
    n_clusters : int
        Número de clusters desejado.
    random_state : int | None
        Semente aleatória para reprodutibilidade.
    n_init : int
        Número de inicializações do algoritmo.

    Retorno
    -------
    tuple[np.ndarray, Any]
        - vetor de rótulos de cluster
        - modelo KMeans ajustado
    """
    modelo_kmeans = KMeans(
        n_clusters=n_clusters,
        random_state=random_state,
        n_init=n_init,
    )

    labels = modelo_kmeans.fit_predict(matriz_reduzida)

    return labels, modelo_kmeans


def aplicar_agglomerative(
    matriz_reduzida: np.ndarray,
    n_clusters: int,
    linkage: str = "ward",
    metric: str = "euclidean",
) -> tuple[np.ndarray, Any]:
    """
    Aplica clusterização hierárquica aglomerativa.

    Parâmetros
    ----------
    matriz_reduzida : np.ndarray
        Matriz de entrada após redução de dimensionalidade.
    n_clusters : int
        Número de clusters desejado.
    linkage : str
        Critério de ligação entre grupos.
    metric : str
        Métrica de distância utilizada.

    Retorno
    -------
    tuple[np.ndarray, Any]
        - vetor de rótulos de cluster
        - modelo Agglomerative ajustado
    """
    # Na implementação do scikit-learn, linkage="ward" exige metric="euclidean".
    if linkage == "ward":
        metric = "euclidean"

    modelo_agglomerative = AgglomerativeClustering(
        n_clusters=n_clusters,
        linkage=linkage,
        metric=metric,
    )

    labels = modelo_agglomerative.fit_predict(matriz_reduzida)

    return labels, modelo_agglomerative


def aplicar_hdbscan(
    matriz_reduzida: np.ndarray,
    min_cluster_size: int = 5,
    min_samples: int | None = None,
    cluster_selection_epsilon: float = 0.0,
    metric: str = "euclidean",
) -> tuple[np.ndarray, Any]:
    """
    Aplica clusterização com HDBSCAN.

    Parâmetros
    ----------
    matriz_reduzida : np.ndarray
        Matriz de entrada após redução de dimensionalidade.
    min_cluster_size : int
        Tamanho mínimo de cluster.
    min_samples : int | None
        Número mínimo de amostras por região densa.
    cluster_selection_epsilon : float
        Valor de tolerância para seleção de clusters.
    metric : str
        Métrica de distância utilizada.

    Retorno
    -------
    tuple[np.ndarray, Any]
        - vetor de rótulos de cluster
        - modelo HDBSCAN ajustado
    """
    modelo_hdbscan = hdbscan.HDBSCAN(
        min_cluster_size=min_cluster_size,
        min_samples=min_samples,
        cluster_selection_epsilon=cluster_selection_epsilon,
        metric=metric,
    )

    labels = modelo_hdbscan.fit_predict(matriz_reduzida)

    return labels, modelo_hdbscan


def calcular_resumo_labels(
    labels: np.ndarray,
) -> dict[str, Any]:
    """
    Calcula um resumo simples dos rótulos gerados pela clusterização.

    O valor -1 é interpretado como ruído, conforme convenção do HDBSCAN.

    Parâmetros
    ----------
    labels : np.ndarray
        Vetor de rótulos produzidos pelo clusterizador.

    Retorno
    -------
    dict[str, Any]
        Resumo contendo quantidade de clusters, ruído e distribuição básica.
    """
    labels = np.asarray(labels)

    quantidade_ruido = int((labels == -1).sum())
    quantidade_total = int(len(labels))

    labels_validos = labels[labels != -1]
    clusters_unicos = sorted(np.unique(labels_validos).tolist())
    quantidade_clusters = int(len(clusters_unicos))

    percentual_ruido = 0.0
    if quantidade_total > 0:
        percentual_ruido = float(quantidade_ruido / quantidade_total)

    resumo = {
        "n_amostras": quantidade_total,
        "n_clusters": quantidade_clusters,
        "n_ruido": quantidade_ruido,
        "percentual_ruido": percentual_ruido,
        "clusters_unicos": clusters_unicos,
    }

    return resumo


def aplicar_clusterizacao(
    matriz_reduzida: np.ndarray,
    config_clusterer: dict[str, Any],
) -> tuple[np.ndarray, Any, dict[str, Any]]:
    """
    Aplica a estratégia de clusterização definida na configuração.

    Estratégias suportadas:
    - kmeans
    - agglomerative
    - hdbscan

    Parâmetros
    ----------
    matriz_reduzida : np.ndarray
        Matriz de entrada após redução de dimensionalidade.
    config_clusterer : dict[str, Any]
        Configuração do clusterizador.

    Retorno
    -------
    tuple[np.ndarray, Any, dict[str, Any]]
        - vetor de rótulos
        - modelo ajustado
        - metadados resumidos da clusterização
    """
    nome_clusterer = config_clusterer.get("name", "").lower()
    params = config_clusterer.get("params", {})

    if nome_clusterer == "kmeans":
        n_clusters = params.get("n_clusters", 5)
        random_state = params.get("random_state", 30)
        n_init = params.get("n_init", 10)

        labels, modelo_clusterer = aplicar_kmeans(
            matriz_reduzida=matriz_reduzida,
            n_clusters=n_clusters,
            random_state=random_state,
            n_init=n_init,
        )

        resumo = calcular_resumo_labels(labels)

        metadados = {
            "clusterer_name": "kmeans",
            "param_n_clusters": n_clusters,
            "param_random_state": random_state,
            "param_n_init": n_init,
            **resumo,
        }

        return labels, modelo_clusterer, metadados

    if nome_clusterer == "agglomerative":
        n_clusters = params.get("n_clusters", 5)
        linkage = params.get("linkage", "ward")
        metric = params.get("metric", "euclidean")

        labels, modelo_clusterer = aplicar_agglomerative(
            matriz_reduzida=matriz_reduzida,
            n_clusters=n_clusters,
            linkage=linkage,
            metric=metric,
        )

        resumo = calcular_resumo_labels(labels)

        metadados = {
            "clusterer_name": "agglomerative",
            "param_n_clusters": n_clusters,
            "param_linkage": linkage,
            "param_metric": metric,
            **resumo,
        }

        return labels, modelo_clusterer, metadados

    if nome_clusterer == "hdbscan":
        min_cluster_size = params.get("min_cluster_size", 5)
        min_samples = params.get("min_samples", None)
        cluster_selection_epsilon = params.get("cluster_selection_epsilon", 0.0)
        metric = params.get("metric", "euclidean")

        labels, modelo_clusterer = aplicar_hdbscan(
            matriz_reduzida=matriz_reduzida,
            min_cluster_size=min_cluster_size,
            min_samples=min_samples,
            cluster_selection_epsilon=cluster_selection_epsilon,
            metric=metric,
        )

        resumo = calcular_resumo_labels(labels)

        metadados = {
            "clusterer_name": "hdbscan",
            "param_min_cluster_size": min_cluster_size,
            "param_min_samples": min_samples,
            "param_cluster_selection_epsilon": cluster_selection_epsilon,
            "param_metric": metric,
            **resumo,
        }

        return labels, modelo_clusterer, metadados

    raise ValueError(f"Clusterizador não suportado: {nome_clusterer}")