from __future__ import annotations

"""
Módulo de avaliação do Estudo de Caso 1.

Responsável por:
- calcular métricas internas de clusterização;
- tratar métricas em cenários com ruído;
- calcular DBCV quando aplicável;
- calcular score composto final;
- consolidar resultados de avaliação em estrutura simples e rastreável.
"""

from typing import Any

import numpy as np
from sklearn.metrics import calinski_harabasz_score, davies_bouldin_score, silhouette_score

try:
    from hdbscan.validity import validity_index
except Exception:
    validity_index = None


def contar_clusters_validos(
    labels: np.ndarray,
) -> int:
    """
    Conta a quantidade de clusters válidos, desconsiderando o rótulo de ruído (-1).

    Parâmetros
    ----------
    labels : np.ndarray
        Vetor de rótulos da clusterização.

    Retorno
    -------
    int
        Quantidade de clusters válidos.
    """
    labels = np.asarray(labels)
    labels_validos = labels[labels != -1]
    return int(len(np.unique(labels_validos)))


def calcular_percentual_ruido(
    labels: np.ndarray,
) -> float:
    """
    Calcula o percentual de amostras marcadas como ruído.

    Parâmetros
    ----------
    labels : np.ndarray
        Vetor de rótulos da clusterização.

    Retorno
    -------
    float
        Percentual de ruído no intervalo [0, 1].
    """
    labels = np.asarray(labels)

    if len(labels) == 0:
        return 0.0

    return float((labels == -1).sum() / len(labels))


def filtrar_ruido(
    matriz_reduzida: np.ndarray,
    labels: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Remove amostras rotuladas como ruído e retorna a matriz e os rótulos filtrados.

    Parâmetros
    ----------
    matriz_reduzida : np.ndarray
        Matriz de entrada após redução de dimensionalidade.
    labels : np.ndarray
        Vetor de rótulos da clusterização.

    Retorno
    -------
    tuple[np.ndarray, np.ndarray]
        - matriz sem ruído
        - labels sem ruído
    """
    labels = np.asarray(labels)
    mascara_validos = labels != -1

    matriz_filtrada = matriz_reduzida[mascara_validos]
    labels_filtrados = labels[mascara_validos]

    return matriz_filtrada, labels_filtrados


def existe_quantidade_minima_para_metricas(
    labels: np.ndarray,
) -> bool:
    """
    Verifica se existe quantidade mínima de clusters para cálculo das métricas internas.

    As métricas silhouette, Davies-Bouldin e Calinski-Harabasz exigem pelo menos
    dois clusters válidos.

    Parâmetros
    ----------
    labels : np.ndarray
        Vetor de rótulos da clusterização.

    Retorno
    -------
    bool
        Indica se o cálculo das métricas é viável.
    """
    quantidade_clusters = contar_clusters_validos(labels)
    return quantidade_clusters >= 2


def calcular_metricas_internas(
    matriz_reduzida: np.ndarray,
    labels: np.ndarray,
    remover_ruido: bool = True,
) -> dict[str, float]:
    """
    Calcula métricas internas de clusterização.

    Quando `remover_ruido=True`, remove amostras com rótulo -1 antes do cálculo.
    Isso é especialmente útil para resultados do HDBSCAN.

    Métricas calculadas:
    - silhouette
    - davies_bouldin
    - calinski_harabasz
    - k
    - noise_pct

    Parâmetros
    ----------
    matriz_reduzida : np.ndarray
        Matriz de entrada após redução de dimensionalidade.
    labels : np.ndarray
        Vetor de rótulos da clusterização.
    remover_ruido : bool
        Indica se amostras com rótulo -1 devem ser removidas antes do cálculo.

    Retorno
    -------
    dict[str, float]
        Dicionário com métricas internas.
    """
    labels = np.asarray(labels)

    noise_pct = calcular_percentual_ruido(labels)
    k_total = contar_clusters_validos(labels)

    matriz_avaliacao = matriz_reduzida
    labels_avaliacao = labels

    if remover_ruido:
        matriz_avaliacao, labels_avaliacao = filtrar_ruido(matriz_reduzida, labels)

    resultado = {
        "silhouette": np.nan,
        "davies_bouldin": np.nan,
        "calinski_harabasz": np.nan,
        "k": float(k_total),
        "noise_pct": float(noise_pct),
    }

    if len(labels_avaliacao) == 0:
        return resultado

    if len(np.unique(labels_avaliacao)) < 2:
        return resultado

    try:
        resultado["silhouette"] = float(
            silhouette_score(matriz_avaliacao, labels_avaliacao)
        )
    except Exception:
        resultado["silhouette"] = np.nan

    try:
        resultado["davies_bouldin"] = float(
            davies_bouldin_score(matriz_avaliacao, labels_avaliacao)
        )
    except Exception:
        resultado["davies_bouldin"] = np.nan

    try:
        resultado["calinski_harabasz"] = float(
            calinski_harabasz_score(matriz_avaliacao, labels_avaliacao)
        )
    except Exception:
        resultado["calinski_harabasz"] = np.nan

    return resultado


def calcular_dbcv(
    matriz_reduzida: np.ndarray,
    labels: np.ndarray,
    tipo_clusterizador: str,
) -> float:
    """
    Calcula o índice DBCV quando aplicável.

    O DBCV é mais coerente para métodos baseados em densidade, especialmente HDBSCAN.
    Quando o cálculo não for possível, retorna NaN.

    Parâmetros
    ----------
    matriz_reduzida : np.ndarray
        Matriz de entrada após redução de dimensionalidade.
    labels : np.ndarray
        Vetor de rótulos da clusterização.
    tipo_clusterizador : str
        Nome do clusterizador utilizado.

    Retorno
    -------
    float
        Valor do DBCV ou NaN.
    """
    if validity_index is None:
        return np.nan

    if tipo_clusterizador.lower() != "hdbscan":
        return np.nan

    labels = np.asarray(labels)

    try:
        k = contar_clusters_validos(labels)
        if k < 2:
            return np.nan

        valor_dbcv = validity_index(matriz_reduzida, labels)
        return float(valor_dbcv)
    except Exception:
        return np.nan


def calcular_inverso_davies_bouldin(
    davies_bouldin: float,
) -> float:
    """
    Calcula o inverso suavizado do índice Davies-Bouldin.

    A transformação adotada é:
        1 / (1 + davies_bouldin)

    Parâmetros
    ----------
    davies_bouldin : float
        Valor do índice Davies-Bouldin.

    Retorno
    -------
    float
        Valor transformado ou NaN.
    """
    if davies_bouldin is None or np.isnan(davies_bouldin):
        return np.nan

    return float(1.0 / (1.0 + davies_bouldin))


def calcular_penalidade_ruido_hdbscan(
    noise_pct: float,
    cfg_score: dict[str, Any],
) -> float:
    """
    Calcula penalidade associada ao percentual de ruído no HDBSCAN.

    A fórmula considera:
    - limiar tolerado;
    - peso da penalidade;
    - expoente da penalidade.

    Quando o percentual de ruído estiver abaixo do limiar, a penalidade é zero.

    Parâmetros
    ----------
    noise_pct : float
        Percentual de ruído no intervalo [0, 1].
    cfg_score : dict[str, Any]
        Configuração do score final.

    Retorno
    -------
    float
        Penalidade calculada.
    """
    penal_cfg = cfg_score.get("penalidade_ruido", {})

    limiar = penal_cfg.get("limiar", 0.0)
    peso = penal_cfg.get("peso", 0.0)
    expoente = penal_cfg.get("expoente", 2)

    excesso = max(0.0, noise_pct - limiar)
    penalidade = peso * (excesso**expoente)

    return float(penalidade)


def calcular_penalidade_k(
    k: float,
    cfg_score: dict[str, Any],
) -> float:
    """
    Calcula penalidade associada à quantidade de clusters fora da faixa ideal.

    Quando o valor de k estiver dentro do intervalo ideal, a penalidade é zero.
    Fora desse intervalo, aplica penalidade proporcional à distância até a faixa.

    Parâmetros
    ----------
    k : float
        Quantidade de clusters válidos.
    cfg_score : dict[str, Any]
        Configuração do score final.

    Retorno
    -------
    float
        Penalidade calculada.
    """
    penal_cfg = cfg_score.get("penalidade_k", {})

    faixa_ideal = penal_cfg.get("faixa_ideal", [3, 8])
    peso = penal_cfg.get("peso", 0.0)

    if not isinstance(faixa_ideal, list) or len(faixa_ideal) != 2:
        return 0.0

    minimo, maximo = faixa_ideal

    if k < minimo:
        distancia = minimo - k
    elif k > maximo:
        distancia = k - maximo
    else:
        distancia = 0.0

    penalidade = peso * distancia
    return float(penalidade)


def calcular_score_final(
    metricas: dict[str, float],
    cfg_score: dict[str, Any],
    tipo_clusterizador: str,
) -> float:
    """
    Calcula o score composto final do experimento.

    O score combina:
    - silhouette
    - DBCV
    - inverso de Davies-Bouldin

    E aplica penalidades:
    - percentual de ruído, quando clusterizador for HDBSCAN;
    - quantidade de clusters fora da faixa ideal.

    Parâmetros
    ----------
    metricas : dict[str, float]
        Dicionário com métricas calculadas.
    cfg_score : dict[str, Any]
        Configuração do score final.
    tipo_clusterizador : str
        Nome do clusterizador utilizado.

    Retorno
    -------
    float
        Score final calculado.
    """
    pesos = cfg_score.get("pesos", {})

    peso_silhouette = pesos.get("silhouette", 1.0)
    peso_dbcv = pesos.get("dbcv", 1.0)
    peso_db_inv = pesos.get("db_inv", 1.0)

    silhouette = metricas.get("silhouette", np.nan)
    dbcv = metricas.get("dbcv", np.nan)
    davies_bouldin = metricas.get("davies_bouldin", np.nan)
    db_inv = calcular_inverso_davies_bouldin(davies_bouldin)

    score_base = 0.0

    if not np.isnan(silhouette):
        score_base += peso_silhouette * silhouette

    if not np.isnan(dbcv):
        score_base += peso_dbcv * dbcv

    if not np.isnan(db_inv):
        score_base += peso_db_inv * db_inv

    noise_pct = metricas.get("noise_pct", 0.0)
    k = metricas.get("k", 0.0)

    penalidade_ruido = 0.0
    if tipo_clusterizador.lower() == "hdbscan":
        penalidade_ruido = calcular_penalidade_ruido_hdbscan(noise_pct, cfg_score)

    penalidade_k = calcular_penalidade_k(k, cfg_score)

    score_final = score_base - penalidade_ruido - penalidade_k

    return float(score_final)


def avaliar_resultado_clusterizacao(
    matriz_reduzida: np.ndarray,
    labels: np.ndarray,
    tipo_clusterizador: str,
    cfg_score: dict[str, Any],
) -> dict[str, float]:
    """
    Avalia o resultado da clusterização e retorna métricas consolidadas.

    Métricas retornadas:
    - silhouette
    - davies_bouldin
    - calinski_harabasz
    - dbcv
    - k
    - noise_pct
    - db_inv
    - score_final

    Parâmetros
    ----------
    matriz_reduzida : np.ndarray
        Matriz de entrada após redução de dimensionalidade.
    labels : np.ndarray
        Vetor de rótulos da clusterização.
    tipo_clusterizador : str
        Nome do clusterizador utilizado.
    cfg_score : dict[str, Any]
        Configuração do score final.

    Retorno
    -------
    dict[str, float]
        Dicionário consolidado de métricas.
    """
    remover_ruido = tipo_clusterizador.lower() == "hdbscan"

    metricas = calcular_metricas_internas(
        matriz_reduzida=matriz_reduzida,
        labels=labels,
        remover_ruido=remover_ruido,
    )

    metricas["dbcv"] = calcular_dbcv(
        matriz_reduzida=matriz_reduzida,
        labels=labels,
        tipo_clusterizador=tipo_clusterizador,
    )

    metricas["db_inv"] = calcular_inverso_davies_bouldin(
        metricas.get("davies_bouldin", np.nan)
    )

    metricas["score_final"] = calcular_score_final(
        metricas=metricas,
        cfg_score=cfg_score,
        tipo_clusterizador=tipo_clusterizador,
    )

    return metricas