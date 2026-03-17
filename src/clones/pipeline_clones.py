from __future__ import annotations

"""
Pipeline principal do Estudo de Caso 1.

Responsável por orquestrar:
- ingestão dos dados;
- aplicação dos filtros base;
- preprocessamento da execução;
- redução de dimensionalidade;
- clusterização;
- avaliação dos resultados;
- montagem dos artefatos intermediários e finais.

Esta primeira versão mantém a execução simples e local, servindo como base
para a integração com MLflow e posterior expansão para grids estruturados.
"""

from typing import Any

import numpy as np
import pandas as pd

from src.clones.avaliacao import avaliar_resultado_clusterizacao
from src.clones.clusterizacao import aplicar_clusterizacao
from src.clones.ingestao import (
    carregar_dataset_excel,
    selecionar_colunas_modelagem,
    separar_colunas_identificacao,
)
from src.clones.preprocessamento import (
    alinhar_identificadores_apos_imputacao,
    aplicar_filtros_base,
    executar_preprocessamento_fase1,
)
from src.clones.reducao_dimensionalidade import (
    aplicar_reducao_dimensionalidade,
    converter_embedding_para_dataframe,
)


def montar_dataframe_resultado_amostras(
    df_id: pd.DataFrame,
    df_embedding: pd.DataFrame,
    labels: np.ndarray,
) -> pd.DataFrame:
    """
    Monta o DataFrame final por amostra, unindo identificadores, embedding e rótulos.

    Parâmetros
    ----------
    df_id : pd.DataFrame
        DataFrame com colunas de identificação alinhadas ao conjunto final.
    df_embedding : pd.DataFrame
        DataFrame com dimensões reduzidas.
    labels : np.ndarray
        Vetor de rótulos da clusterização.

    Retorno
    -------
    pd.DataFrame
        DataFrame consolidado por amostra.
    """
    df_resultado = pd.concat(
        [
            df_id.reset_index(drop=True),
            df_embedding.reset_index(drop=True),
            pd.DataFrame({"cluster": labels}).reset_index(drop=True),
        ],
        axis=1,
    )

    return df_resultado

def extrair_parametros_execucao(
    config_unitaria: dict[str, Any],
) -> dict[str, Any]:
    """
    Extrai os principais parâmetros da execução em formato plano.

    Esta estrutura será útil para:
    - logs;
    - MLflow;
    - tabelas de resumo;
    - auditoria das execuções.

    Parâmetros
    ----------
    config_unitaria : dict[str, Any]
        Configuração unitária da execução.

    Retorno
    -------
    dict[str, Any]
        Dicionário plano com parâmetros relevantes.
    """
    preprocessamento = config_unitaria.get("preprocessamento", {})
    modelagem = config_unitaria.get("modelagem", {})

    imputacao_cfg = preprocessamento.get("imputacao", {})
    onehot_cfg = preprocessamento.get("onehot", {})
    normalizacao_cfg = preprocessamento.get("normalizacao", {})

    reducer_cfg = modelagem.get("reducer", {})
    clusterer_cfg = modelagem.get("clusterer", {})

    parametros = {
        "indice_execucao": config_unitaria.get("execucao", {}).get("indice_execucao"),
        "nome_execucao": config_unitaria.get("execucao", {}).get("nome_execucao"),
        "fase_experimental": config_unitaria.get("execucao", {}).get("fase_experimental"),
        "imputacao_tipo": imputacao_cfg.get("tipo"),
        "onehot_colunas": ",".join(onehot_cfg.get("colunas", [])),
        "normalizacao_tipo": normalizacao_cfg.get("tipo"),
        "reducer_name": reducer_cfg.get("name"),
        "clusterer_name": clusterer_cfg.get("name"),
    }

    for chave, valor in reducer_cfg.get("params", {}).items():
        parametros[f"reducer__{chave}"] = valor

    for chave, valor in clusterer_cfg.get("params", {}).items():
        parametros[f"clusterer__{chave}"] = valor

    for chave, valor in imputacao_cfg.get("params", {}).items():
        parametros[f"imputacao__{chave}"] = valor

    return parametros


def executar_pipeline_clones(
    config_unitaria: dict[str, Any],
    config_score: dict[str, Any],
) -> dict[str, Any]:
    """
    Executa o pipeline completo do Estudo de Caso 1 para uma configuração unitária.

    Etapas executadas:
    - leitura do dataset bruto;
    - filtros base e padronizações;
    - separação de identificadores e colunas de modelagem;
    - preprocessamento da execução;
    - redução de dimensionalidade;
    - clusterização;
    - avaliação dos resultados;
    - montagem do resultado final por amostra.

    Parâmetros
    ----------
    config_unitaria : dict[str, Any]
        Configuração unitária da execução.
    config_score : dict[str, Any]
        Configuração do score final.

    Retorno
    -------
    dict[str, Any]
        Dicionário com artefatos, métricas, parâmetros e resultados da execução.
    """
    # -----------------------------------------------------
    # 1. Ingestão
    # -----------------------------------------------------
    config_dados = config_unitaria["dados"]
    df_bruto = carregar_dataset_excel(
        caminho_excel=config_dados["caminho_excel"],
        aba=config_dados["aba"],
    )

    # -----------------------------------------------------
    # 2. Filtros base
    # -----------------------------------------------------
    config_preprocessamento = config_unitaria["preprocessamento"]

    df_filtrado = aplicar_filtros_base(
        df=df_bruto,
        especie_replace=config_preprocessamento.get("padronizacoes", {}).get("especie_replace"),
        especies_para_excluir=config_preprocessamento.get("filtros", {}).get("especies_para_excluir"),
        idade_min=config_preprocessamento.get("filtros", {}).get("idade_min"),
        idade_max=config_preprocessamento.get("filtros", {}).get("idade_max"),
    )

    # -----------------------------------------------------
    # 3. Separação de identificadores e base de modelagem
    # -----------------------------------------------------
    colunas_id = config_dados.get("id_columns", [])
    df_id, df_modelagem = separar_colunas_identificacao(
        df=df_filtrado,
        colunas_id=colunas_id,
    )

    df_modelagem = selecionar_colunas_modelagem(
        df=df_modelagem,
        colunas_excluir=config_preprocessamento.get("colunas_para_remover", []),
    )

    serie_grupo_imputacao = None
    if "TT" in df_id.columns:
        serie_grupo_imputacao = df_id["TT"].copy()

    # -----------------------------------------------------
    # 4. Preprocessamento da execução
    # -----------------------------------------------------
    df_features, metadados_preprocessamento = executar_preprocessamento_fase1(
        df_modelagem=df_modelagem,
        config_preprocessamento=config_preprocessamento,
        serie_grupo_imputacao=serie_grupo_imputacao,
    )

    df_id_alinhado = alinhar_identificadores_apos_imputacao(
        df_id=df_id,
        df_modelagem_resultante=df_features,
    )

    # -----------------------------------------------------
    # 5. Redução de dimensionalidade
    # -----------------------------------------------------
    config_reducer = config_unitaria["modelagem"]["reducer"]

    matriz_reduzida, modelo_reducer, metadados_reducer = aplicar_reducao_dimensionalidade(
        df_features=df_features,
        config_reducer=config_reducer,
    )

    df_embedding = converter_embedding_para_dataframe(
        matriz_reduzida=matriz_reduzida,
        prefixo_coluna="dim",
    )

    # -----------------------------------------------------
    # 6. Clusterização
    # -----------------------------------------------------
    config_clusterer = config_unitaria["modelagem"]["clusterer"]

    labels, modelo_clusterer, metadados_clusterizacao = aplicar_clusterizacao(
        matriz_reduzida=matriz_reduzida,
        config_clusterer=config_clusterer,
    )

    # -----------------------------------------------------
    # 7. Avaliação
    # -----------------------------------------------------
    nome_clusterer = config_clusterer.get("name", "")

    metricas_avaliacao = avaliar_resultado_clusterizacao(
        matriz_reduzida=matriz_reduzida,
        labels=labels,
        tipo_clusterizador=nome_clusterer,
        cfg_score=config_score,
    )

    # -----------------------------------------------------
    # 8. Resultado por amostra
    # -----------------------------------------------------
    df_resultado_amostras = montar_dataframe_resultado_amostras(
        df_id=df_id_alinhado,
        df_embedding=df_embedding,
        labels=labels,
    )

    # -----------------------------------------------------
    # 9. Resumo consolidado da execução
    # -----------------------------------------------------
    parametros_execucao = extrair_parametros_execucao(config_unitaria)
    colunas_features_finais = df_features.columns.tolist()

    resumo_execucao = {
        **parametros_execucao,
        **metadados_preprocessamento,
        **metadados_reducer,
        **metadados_clusterizacao,
        **metricas_avaliacao,
        "n_amostras_bruto": len(df_bruto),
        "n_amostras_filtrado": len(df_filtrado),
        "n_amostras_resultado": len(df_resultado_amostras),
        "n_colunas_features_finais": len(colunas_features_finais),
        "colunas_features_finais": colunas_features_finais,
    }

    return {
        "config_unitaria": config_unitaria,
        "parametros_execucao": parametros_execucao,
        "metadados_preprocessamento": metadados_preprocessamento,
        "metadados_reducer": metadados_reducer,
        "metadados_clusterizacao": metadados_clusterizacao,
        "metricas_avaliacao": metricas_avaliacao,
        "resumo_execucao": resumo_execucao,
        "df_bruto": df_bruto,
        "df_filtrado": df_filtrado,
        "df_id_alinhado": df_id_alinhado,
        "df_features": df_features,
        "matriz_reduzida": matriz_reduzida,
        "df_embedding": df_embedding,
        "labels": labels,
        "df_resultado_amostras": df_resultado_amostras,
        "modelo_reducer": modelo_reducer,
        "modelo_clusterer": modelo_clusterer,
    }