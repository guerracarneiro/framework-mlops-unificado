from __future__ import annotations

"""
Runner da Fase 2D: sensibilidade à multicolinearidade.

Objetivo:
- partir da base tratada oficial da Fase 2C;
- aplicar poda de multicolinearidade no bloco numérico;
- reconstruir a matriz final de modelagem com o one-hot oficial;
- rodar a configuração oficial de UMAP + HDBSCAN;
- comparar cada partição obtida com a partição oficial via AMI/NMI.

Estratégia:
- correlação absoluta por limiar;
- componentes conexos de variáveis correlacionadas;
- seleção determinística de 1 representante por componente;
- múltiplas seeds do UMAP por limiar.
"""

import argparse
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Any

import mlflow
import numpy as np
import pandas as pd
import yaml
from sklearn.metrics import adjusted_mutual_info_score, normalized_mutual_info_score

from src.clones.avaliacao import avaliar_resultado_clusterizacao
from src.clones.clusterizacao import aplicar_clusterizacao
from src.clones.preprocessamento import (
    aplicar_onehot,
    concatenar_blocos_modelagem,
    normalizar_dados,
    selecionar_colunas_numericas,
)
from src.clones.reducao_dimensionalidade import (
    aplicar_reducao_dimensionalidade,
    converter_embedding_para_dataframe,
)
from src.clones.run_experimento_clones import (
    garantir_pasta,
    preparar_experimento_mlflow,
    registrar_artefatos_mlflow,
    registrar_metricas_mlflow,
    registrar_parametros_mlflow,
    salvar_artefatos_locais_execucao,
)


def criar_parser_argumentos() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Executa a Fase 2D de sensibilidade à multicolinearidade."
    )
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Caminho do arquivo YAML da Fase 2D.",
    )
    return parser


def carregar_yaml(caminho_yaml: str | Path) -> dict[str, Any]:
    caminho_yaml = Path(caminho_yaml)

    if not caminho_yaml.exists():
        raise FileNotFoundError(f"Arquivo YAML não encontrado: {caminho_yaml}")

    with caminho_yaml.open("r", encoding="utf-8") as arquivo:
        return yaml.safe_load(arquivo)


def carregar_config_score(caminho_score: str | Path) -> dict[str, Any]:
    caminho_score = Path(caminho_score)

    if not caminho_score.exists():
        raise FileNotFoundError(f"Arquivo de score não encontrado: {caminho_score}")

    with caminho_score.open("r", encoding="utf-8") as arquivo:
        return yaml.safe_load(arquivo)


def carregar_execucao_oficial(config: dict[str, Any]) -> tuple[pd.DataFrame, pd.DataFrame]:
    cfg_ref = config["referencia"]

    pasta_execucao = Path(cfg_ref["pasta_execucao_oficial"])
    arquivo_base = pasta_execucao / cfg_ref["arquivo_base_tratada"]
    arquivo_resultado = pasta_execucao / cfg_ref["arquivo_resultado_amostras"]

    if not pasta_execucao.exists():
        raise FileNotFoundError(f"Pasta da execução oficial não encontrada: {pasta_execucao}")

    if not arquivo_base.exists():
        raise FileNotFoundError(f"Arquivo base_analitica_tratada não encontrado: {arquivo_base}")

    if not arquivo_resultado.exists():
        raise FileNotFoundError(f"Arquivo resultado_amostras não encontrado: {arquivo_resultado}")

    df_base_tratada = pd.read_csv(arquivo_base, encoding="utf-8-sig")
    df_resultado_amostras = pd.read_csv(arquivo_resultado, encoding="utf-8-sig")

    return df_base_tratada, df_resultado_amostras


def validar_estrutura_referencia(
    df_base_tratada: pd.DataFrame,
    df_resultado_amostras: pd.DataFrame,
    colunas_id: list[str],
    coluna_cluster: str,
) -> None:
    if len(df_base_tratada) != len(df_resultado_amostras):
        raise ValueError(
            "A base tratada e o resultado oficial possuem quantidades de linhas diferentes. "
            f"base={len(df_base_tratada)} | resultado={len(df_resultado_amostras)}"
        )

    for coluna_id in colunas_id:
        if coluna_id not in df_resultado_amostras.columns:
            raise ValueError(
                f"Coluna de identificação '{coluna_id}' não encontrada em resultado_amostras.csv"
            )

    if coluna_cluster not in df_resultado_amostras.columns:
        raise ValueError(
            f"Coluna de cluster '{coluna_cluster}' não encontrada em resultado_amostras.csv"
        )


def calcular_correlacao_absoluta(
    df_numerico: pd.DataFrame,
    metodo: str,
) -> pd.DataFrame:
    metodo = str(metodo).strip().lower()
    df_corr = df_numerico.corr(method=metodo)
    return df_corr.abs()


def identificar_componentes_correlacionados(
    df_corr_abs: pd.DataFrame,
    limiar_correlacao_abs: float,
) -> list[list[str]]:
    colunas = list(df_corr_abs.columns)
    adjacencias: dict[str, set[str]] = {coluna: set() for coluna in colunas}

    for i, coluna_i in enumerate(colunas):
        for j in range(i + 1, len(colunas)):
            coluna_j = colunas[j]
            valor_corr = df_corr_abs.iat[i, j]

            if pd.isna(valor_corr):
                continue

            if float(valor_corr) >= float(limiar_correlacao_abs):
                adjacencias[coluna_i].add(coluna_j)
                adjacencias[coluna_j].add(coluna_i)

    visitados: set[str] = set()
    componentes: list[list[str]] = []

    for coluna in colunas:
        if coluna in visitados:
            continue

        fila = [coluna]
        visitados.add(coluna)
        componente = [coluna]

        while fila:
            atual = fila.pop()
            for vizinha in adjacencias[atual]:
                if vizinha not in visitados:
                    visitados.add(vizinha)
                    fila.append(vizinha)
                    componente.append(vizinha)

        componentes.append(sorted(componente))

    return componentes


def selecionar_representantes_deterministicos(
    df_corr_abs: pd.DataFrame,
    componentes: list[list[str]],
) -> tuple[list[str], pd.DataFrame]:
    """
    Mantém 1 representante por componente.

    Regra:
    - componentes de tamanho 1: mantém a própria variável;
    - componentes maiores: mantém a variável com menor redundância média;
    - em empate, mantém a de nome alfabético menor.
    """
    representantes: list[str] = []
    registros_componentes: list[dict[str, Any]] = []

    for indice_componente, componente in enumerate(componentes, start=1):
        if len(componente) == 1:
            representante = componente[0]
            redundancia_media = 0.0
        else:
            df_subcorr = df_corr_abs.loc[componente, componente].copy()

            redundancias: list[tuple[str, float]] = []
            for coluna in componente:
                soma_corr = float(df_subcorr.loc[coluna].sum()) - 1.0
                redundancia_media = soma_corr / float(len(componente) - 1)
                redundancias.append((coluna, redundancia_media))

            redundancias_ordenadas = sorted(
                redundancias,
                key=lambda item: (item[1], item[0]),
            )

            representante = redundancias_ordenadas[0][0]
            redundancia_media = redundancias_ordenadas[0][1]

        representantes.append(representante)

        for coluna in componente:
            registros_componentes.append(
                {
                    "indice_componente": indice_componente,
                    "tamanho_componente": len(componente),
                    "feature": coluna,
                    "representante": representante,
                    "mantida": int(coluna == representante),
                    "redundancia_media_representante": redundancia_media,
                }
            )

    df_componentes = pd.DataFrame(registros_componentes)
    representantes = sorted(representantes)

    return representantes, df_componentes


def extrair_pares_correlacionados(
    df_corr_abs: pd.DataFrame,
    limiar_correlacao_abs: float,
) -> pd.DataFrame:
    registros: list[dict[str, Any]] = []
    colunas = list(df_corr_abs.columns)

    for i, coluna_i in enumerate(colunas):
        for j in range(i + 1, len(colunas)):
            coluna_j = colunas[j]
            valor_corr = df_corr_abs.iat[i, j]

            if pd.isna(valor_corr):
                continue

            if float(valor_corr) >= float(limiar_correlacao_abs):
                registros.append(
                    {
                        "feature_1": coluna_i,
                        "feature_2": coluna_j,
                        "correlacao_abs": float(valor_corr),
                        "limiar_correlacao_abs": float(limiar_correlacao_abs),
                    }
                )

    return pd.DataFrame(registros)


def podar_multicolinearidade(
    df_numerico: pd.DataFrame,
    metodo_correlacao: str,
    limiar_correlacao_abs: float | None,
) -> tuple[pd.DataFrame, list[str], list[str], pd.DataFrame, pd.DataFrame]:
    """
    Aplica poda de multicolinearidade apenas no bloco numérico.
    """
    if limiar_correlacao_abs is None:
        features_mantidas = sorted(df_numerico.columns.tolist())
        features_removidas: list[str] = []

        df_componentes = pd.DataFrame(
            [
                {
                    "indice_componente": i + 1,
                    "tamanho_componente": 1,
                    "feature": coluna,
                    "representante": coluna,
                    "mantida": 1,
                    "redundancia_media_representante": 0.0,
                }
                for i, coluna in enumerate(features_mantidas)
            ]
        )

        df_pares = pd.DataFrame(
            columns=["feature_1", "feature_2", "correlacao_abs", "limiar_correlacao_abs"]
        )

        return (
            df_numerico.copy(),
            features_mantidas,
            features_removidas,
            df_componentes,
            df_pares,
        )

    df_corr_abs = calcular_correlacao_absoluta(
        df_numerico=df_numerico,
        metodo=metodo_correlacao,
    )

    componentes = identificar_componentes_correlacionados(
        df_corr_abs=df_corr_abs,
        limiar_correlacao_abs=float(limiar_correlacao_abs),
    )

    features_mantidas, df_componentes = selecionar_representantes_deterministicos(
        df_corr_abs=df_corr_abs,
        componentes=componentes,
    )

    features_removidas = sorted(
        [coluna for coluna in df_numerico.columns if coluna not in set(features_mantidas)]
    )

    df_pares = extrair_pares_correlacionados(
        df_corr_abs=df_corr_abs,
        limiar_correlacao_abs=float(limiar_correlacao_abs),
    )

    df_numerico_podado = df_numerico[features_mantidas].copy()

    return (
        df_numerico_podado,
        features_mantidas,
        features_removidas,
        df_componentes,
        df_pares,
    )


def montar_features_com_poda(
    df_base_tratada: pd.DataFrame,
    colunas_id: list[str],
    colunas_onehot: list[str],
    tipo_normalizacao: str,
    metodo_correlacao: str,
    limiar_correlacao_abs: float | None,
) -> tuple[pd.DataFrame, pd.DataFrame, list[str], list[str], pd.DataFrame, pd.DataFrame]:
    df_trabalho = df_base_tratada.copy()

    colunas_id_presentes = [col for col in colunas_id if col in df_trabalho.columns]
    if colunas_id_presentes:
        df_trabalho = df_trabalho.drop(columns=colunas_id_presentes)

    df_numerico = selecionar_colunas_numericas(df_trabalho)

    (
        df_numerico_podado,
        features_mantidas,
        features_removidas,
        df_componentes,
        df_pares,
    ) = podar_multicolinearidade(
        df_numerico=df_numerico,
        metodo_correlacao=metodo_correlacao,
        limiar_correlacao_abs=limiar_correlacao_abs,
    )

    df_numerico_normalizado, _ = normalizar_dados(
        df_numerico_podado,
        tipo=tipo_normalizacao,
    )

    df_onehot = aplicar_onehot(
        df_trabalho,
        colunas=colunas_onehot,
    )

    df_features = concatenar_blocos_modelagem(
        df_numerico=df_numerico_normalizado,
        df_onehot=df_onehot,
    )

    return (
        df_features,
        df_numerico_podado,
        features_mantidas,
        features_removidas,
        df_componentes,
        df_pares,
    )


def executar_modelagem_oficial(
    df_features: pd.DataFrame,
    config: dict[str, Any],
    seed_umap: int,
    config_score: dict[str, Any],
) -> tuple[pd.DataFrame, np.ndarray, dict[str, Any], dict[str, Any], dict[str, Any]]:
    cfg_reducer = config["modelagem_oficial"]["reducer"].copy()
    cfg_clusterer = config["modelagem_oficial"]["clusterer"].copy()

    cfg_reducer["params"] = dict(cfg_reducer.get("params", {}))
    cfg_reducer["params"]["random_state"] = int(seed_umap)

    matriz_reduzida, _, metadados_reducer = aplicar_reducao_dimensionalidade(
        df_features=df_features,
        config_reducer=cfg_reducer,
    )

    labels, _, metadados_clusterizacao = aplicar_clusterizacao(
        matriz_reduzida=matriz_reduzida,
        config_clusterer=cfg_clusterer,
    )

    metricas_avaliacao = avaliar_resultado_clusterizacao(
        matriz_reduzida=matriz_reduzida,
        labels=labels,
        tipo_clusterizador=cfg_clusterer["name"],
        cfg_score=config_score,
    )

    df_embedding = converter_embedding_para_dataframe(matriz_reduzida)

    return (
        df_embedding,
        labels,
        metricas_avaliacao,
        metadados_reducer,
        metadados_clusterizacao,
    )


def calcular_ami_nmi(
    labels_referencia: np.ndarray,
    labels_preditos: np.ndarray,
) -> dict[str, float]:
    return {
        "ami": float(adjusted_mutual_info_score(labels_referencia, labels_preditos)),
        "nmi": float(normalized_mutual_info_score(labels_referencia, labels_preditos)),
    }


def montar_resultado_amostras(
    df_resultado_referencia: pd.DataFrame,
    df_embedding: pd.DataFrame,
    labels_preditos: np.ndarray,
    colunas_id: list[str],
) -> pd.DataFrame:
    df_ids = df_resultado_referencia[colunas_id].reset_index(drop=True).copy()

    df_resultado = pd.concat(
        [
            df_ids,
            df_embedding.reset_index(drop=True),
            pd.DataFrame({"cluster": labels_preditos}).reset_index(drop=True),
        ],
        axis=1,
    )

    return df_resultado


def salvar_artefatos_complementares(
    pasta_execucao: Path,
    features_mantidas: list[str],
    features_removidas: list[str],
    df_componentes: pd.DataFrame,
    df_pares: pd.DataFrame,
    limiar_correlacao_abs: float | None,
) -> dict[str, Path]:
    caminhos: dict[str, Path] = {}

    df_features_mantidas = pd.DataFrame(
        {
            "limiar_correlacao_abs": [limiar_correlacao_abs] * len(features_mantidas),
            "feature": features_mantidas,
            "status": ["mantida"] * len(features_mantidas),
        }
    )
    caminho_features_mantidas = pasta_execucao / "features_mantidas.csv"
    df_features_mantidas.to_csv(caminho_features_mantidas, index=False, encoding="utf-8-sig")
    caminhos["features_mantidas_csv"] = caminho_features_mantidas

    df_features_removidas = pd.DataFrame(
        {
            "limiar_correlacao_abs": [limiar_correlacao_abs] * len(features_removidas),
            "feature": features_removidas,
            "status": ["removida"] * len(features_removidas),
        }
    )
    caminho_features_removidas = pasta_execucao / "features_removidas.csv"
    df_features_removidas.to_csv(caminho_features_removidas, index=False, encoding="utf-8-sig")
    caminhos["features_removidas_csv"] = caminho_features_removidas

    caminho_componentes = pasta_execucao / "componentes_correlacionados.csv"
    df_componentes.to_csv(caminho_componentes, index=False, encoding="utf-8-sig")
    caminhos["componentes_correlacionados_csv"] = caminho_componentes

    caminho_pares = pasta_execucao / "pares_correlacionados.csv"
    df_pares.to_csv(caminho_pares, index=False, encoding="utf-8-sig")
    caminhos["pares_correlacionados_csv"] = caminho_pares

    return caminhos


def consolidar_resumo_por_limiar(
    df_resumo_execucoes: pd.DataFrame,
) -> pd.DataFrame:
    colunas_metricas = [
        "ami",
        "nmi",
        "score_final",
        "noise_pct",
        "n_clusters",
        "n_features_modelagem",
        "n_features_numericas_apos_poda",
        "n_features_removidas",
        "n_pares_correlacionados",
    ]

    df_trabalho = df_resumo_execucoes.copy()

    for coluna in colunas_metricas:
        if coluna in df_trabalho.columns:
            df_trabalho[coluna] = pd.to_numeric(df_trabalho[coluna], errors="coerce")

    agrupado = (
        df_trabalho.groupby("limiar_correlacao_abs", dropna=False)[colunas_metricas]
        .agg(["mean", "std", "min", "max"])
        .reset_index()
    )

    agrupado.columns = [
        "_".join([parte for parte in coluna if parte]).strip("_")
        if isinstance(coluna, tuple)
        else coluna
        for coluna in agrupado.columns
    ]

    return agrupado

def normalizar_metodo_correlacao(valor: str) -> str:
    """
    Normaliza o nome do método de correlação e aceita aliases simples.
    """
    metodo = str(valor).strip().lower()

    aliases = {
        "pearson": "pearson",
        "spearman": "spearman",
        "sperman": "spearman",
    }

    if metodo not in aliases:
        raise ValueError(
            "Método de correlação não suportado. "
            f"Valor recebido: {valor}. "
            "Valores aceitos: 'pearson' ou 'spearman'."
        )

    return aliases[metodo]


def validar_configuracao_fase2d(config: dict[str, Any]) -> None:
    """
    Valida os campos principais da configuração da Fase 2D antes de iniciar a execução.
    """
    tipo_normalizacao = (
        config["preprocessamento"]["normalizacao"]["tipo"]
    )

    if str(tipo_normalizacao).strip().lower() != "standard":
        raise ValueError(
            "Tipo de normalização inválido na Fase 2D. "
            f"Valor recebido: {tipo_normalizacao}. "
            "No projeto atual, o valor suportado é apenas 'standard'."
        )

    metodo_correlacao = config["sensibilidade_multicolinearidade"]["metodo_correlacao"]
    config["sensibilidade_multicolinearidade"]["metodo_correlacao"] = (
        normalizar_metodo_correlacao(metodo_correlacao)
    )


def formatar_limiar_para_rotulo(limiar_correlacao_abs: float | None) -> str:
    """
    Formata o limiar para uso em rótulos e nomes de arquivos.
    """
    if limiar_correlacao_abs is None:
        return "sem_poda"

    return str(limiar_correlacao_abs).replace(".", "p")


def gerar_grafico_ami_por_limiar(
    df_resumo_execucoes: pd.DataFrame,
    pasta_reports: Path,
) -> dict[str, Path]:
    """
    Gera gráfico de AMI por limiar usando média e desvio-padrão entre seeds.
    """
    df_plot = df_resumo_execucoes.copy()
    df_plot["limiar_plot"] = df_plot["limiar_correlacao_abs"].apply(
        lambda x: -1.0 if pd.isna(x) else float(x)
    )

    df_plot["rotulo_limiar"] = df_plot["limiar_correlacao_abs"].apply(
        lambda x: "sem_poda" if pd.isna(x) else str(x)
    )

    df_resumo = (
        df_plot.groupby(["limiar_plot", "rotulo_limiar"], as_index=False)["ami"]
        .agg(["mean", "std", "min", "max"])
        .reset_index()
    )

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.errorbar(
        df_resumo["limiar_plot"],
        df_resumo["mean"],
        yerr=df_resumo["std"].fillna(0.0),
        marker="o",
        capsize=4,
    )

    ax.set_xticks(df_resumo["limiar_plot"])
    ax.set_xticklabels(df_resumo["rotulo_limiar"])
    ax.set_xlabel("Limiar de correlação absoluta")
    ax.set_ylabel("AMI")
    ax.set_title("Sensibilidade da partição oficial por limiar - AMI")
    ax.grid(True, alpha=0.3)

    caminho_png = pasta_reports / "grafico_ami_por_limiar.png"
    caminho_pdf = pasta_reports / "grafico_ami_por_limiar.pdf"

    fig.tight_layout()
    fig.savefig(caminho_png, dpi=300, bbox_inches="tight")
    fig.savefig(caminho_pdf, bbox_inches="tight")
    plt.close(fig)

    return {
        "grafico_ami_png": caminho_png,
        "grafico_ami_pdf": caminho_pdf,
    }


def gerar_grafico_nmi_por_limiar(
    df_resumo_execucoes: pd.DataFrame,
    pasta_reports: Path,
) -> dict[str, Path]:
    """
    Gera gráfico de NMI por limiar usando média e desvio-padrão entre seeds.
    """
    df_plot = df_resumo_execucoes.copy()
    df_plot["limiar_plot"] = df_plot["limiar_correlacao_abs"].apply(
        lambda x: -1.0 if pd.isna(x) else float(x)
    )

    df_plot["rotulo_limiar"] = df_plot["limiar_correlacao_abs"].apply(
        lambda x: "sem_poda" if pd.isna(x) else str(x)
    )

    df_resumo = (
        df_plot.groupby(["limiar_plot", "rotulo_limiar"], as_index=False)["nmi"]
        .agg(["mean", "std", "min", "max"])
        .reset_index()
    )

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.errorbar(
        df_resumo["limiar_plot"],
        df_resumo["mean"],
        yerr=df_resumo["std"].fillna(0.0),
        marker="o",
        capsize=4,
    )

    ax.set_xticks(df_resumo["limiar_plot"])
    ax.set_xticklabels(df_resumo["rotulo_limiar"])
    ax.set_xlabel("Limiar de correlação absoluta")
    ax.set_ylabel("NMI")
    ax.set_title("Sensibilidade da partição oficial por limiar - NMI")
    ax.grid(True, alpha=0.3)

    caminho_png = pasta_reports / "grafico_nmi_por_limiar.png"
    caminho_pdf = pasta_reports / "grafico_nmi_por_limiar.pdf"

    fig.tight_layout()
    fig.savefig(caminho_png, dpi=300, bbox_inches="tight")
    fig.savefig(caminho_pdf, bbox_inches="tight")
    plt.close(fig)

    return {
        "grafico_nmi_png": caminho_png,
        "grafico_nmi_pdf": caminho_pdf,
    }


def gerar_grafico_score_final_por_limiar(
    df_resumo_execucoes: pd.DataFrame,
    pasta_reports: Path,
) -> dict[str, Path]:
    """
    Gera gráfico de score_final por limiar.
    """
    df_plot = df_resumo_execucoes.copy()
    df_plot["limiar_plot"] = df_plot["limiar_correlacao_abs"].apply(
        lambda x: -1.0 if pd.isna(x) else float(x)
    )

    df_plot["rotulo_limiar"] = df_plot["limiar_correlacao_abs"].apply(
        lambda x: "sem_poda" if pd.isna(x) else str(x)
    )

    df_resumo = (
        df_plot.groupby(["limiar_plot", "rotulo_limiar"], as_index=False)["score_final"]
        .agg(["mean", "std", "min", "max"])
        .reset_index()
    )

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.errorbar(
        df_resumo["limiar_plot"],
        df_resumo["mean"],
        yerr=df_resumo["std"].fillna(0.0),
        marker="o",
        capsize=4,
    )

    ax.set_xticks(df_resumo["limiar_plot"])
    ax.set_xticklabels(df_resumo["rotulo_limiar"])
    ax.set_xlabel("Limiar de correlação absoluta")
    ax.set_ylabel("score_final")
    ax.set_title("Sensibilidade da qualidade interna por limiar - score_final")
    ax.grid(True, alpha=0.3)

    caminho_png = pasta_reports / "grafico_score_final_por_limiar.png"
    caminho_pdf = pasta_reports / "grafico_score_final_por_limiar.pdf"

    fig.tight_layout()
    fig.savefig(caminho_png, dpi=300, bbox_inches="tight")
    fig.savefig(caminho_pdf, bbox_inches="tight")
    plt.close(fig)

    return {
        "grafico_score_final_png": caminho_png,
        "grafico_score_final_pdf": caminho_pdf,
    }


def gerar_grafico_noise_clusters_por_limiar(
    df_resumo_execucoes: pd.DataFrame,
    pasta_reports: Path,
) -> dict[str, Path]:
    """
    Gera gráficos complementares de noise_pct e n_clusters por limiar.
    """
    df_plot = df_resumo_execucoes.copy()
    df_plot["limiar_plot"] = df_plot["limiar_correlacao_abs"].apply(
        lambda x: -1.0 if pd.isna(x) else float(x)
    )

    df_plot["rotulo_limiar"] = df_plot["limiar_correlacao_abs"].apply(
        lambda x: "sem_poda" if pd.isna(x) else str(x)
    )

    df_resumo = (
        df_plot.groupby(["limiar_plot", "rotulo_limiar"], as_index=False)[["noise_pct", "n_clusters"]]
        .agg(["mean", "std", "min", "max"])
        .reset_index()
    )

    df_resumo.columns = [
        "_".join([parte for parte in coluna if parte]).strip("_")
        if isinstance(coluna, tuple)
        else coluna
        for coluna in df_resumo.columns
    ]

    # gráfico noise_pct
    fig1, ax1 = plt.subplots(figsize=(10, 6))
    ax1.errorbar(
        df_resumo["limiar_plot"],
        df_resumo["noise_pct_mean"],
        yerr=df_resumo["noise_pct_std"].fillna(0.0),
        marker="o",
        capsize=4,
    )
    ax1.set_xticks(df_resumo["limiar_plot"])
    ax1.set_xticklabels(df_resumo["rotulo_limiar"])
    ax1.set_xlabel("Limiar de correlação absoluta")
    ax1.set_ylabel("noise_pct")
    ax1.set_title("Sensibilidade do ruído por limiar")
    ax1.grid(True, alpha=0.3)

    caminho_noise_png = pasta_reports / "grafico_noise_pct_por_limiar.png"
    caminho_noise_pdf = pasta_reports / "grafico_noise_pct_por_limiar.pdf"

    fig1.tight_layout()
    fig1.savefig(caminho_noise_png, dpi=300, bbox_inches="tight")
    fig1.savefig(caminho_noise_pdf, bbox_inches="tight")
    plt.close(fig1)

    # gráfico n_clusters
    fig2, ax2 = plt.subplots(figsize=(10, 6))
    ax2.errorbar(
        df_resumo["limiar_plot"],
        df_resumo["n_clusters_mean"],
        yerr=df_resumo["n_clusters_std"].fillna(0.0),
        marker="o",
        capsize=4,
    )
    ax2.set_xticks(df_resumo["limiar_plot"])
    ax2.set_xticklabels(df_resumo["rotulo_limiar"])
    ax2.set_xlabel("Limiar de correlação absoluta")
    ax2.set_ylabel("n_clusters")
    ax2.set_title("Sensibilidade do número de grupos por limiar")
    ax2.grid(True, alpha=0.3)

    caminho_clusters_png = pasta_reports / "grafico_n_clusters_por_limiar.png"
    caminho_clusters_pdf = pasta_reports / "grafico_n_clusters_por_limiar.pdf"

    fig2.tight_layout()
    fig2.savefig(caminho_clusters_png, dpi=300, bbox_inches="tight")
    fig2.savefig(caminho_clusters_pdf, bbox_inches="tight")
    plt.close(fig2)

    return {
        "grafico_noise_pct_png": caminho_noise_png,
        "grafico_noise_pct_pdf": caminho_noise_pdf,
        "grafico_n_clusters_png": caminho_clusters_png,
        "grafico_n_clusters_pdf": caminho_clusters_pdf,
    }


def executar_fase2d(
    caminho_config: str | Path,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    config = carregar_yaml(caminho_config)
    validar_configuracao_fase2d(config)
    config_score = carregar_config_score(config["criterios"]["caminho_score_final"])

    colunas_id = config["referencia"]["colunas_id"]
    coluna_cluster = config["referencia"]["coluna_cluster_referencia"]
    colunas_onehot = config["preprocessamento"]["onehot"]["colunas"]
    tipo_normalizacao = config["preprocessamento"]["normalizacao"]["tipo"]

    metodo_correlacao = config["sensibilidade_multicolinearidade"]["metodo_correlacao"]
    limiares = list(config["sensibilidade_multicolinearidade"]["limiares_correlacao_abs"])
    incluir_referencia_sem_poda = bool(
        config["sensibilidade_multicolinearidade"]["incluir_referencia_sem_poda"]
    )

    seeds_umap = [int(seed) for seed in config["robustez"]["seeds_umap"]]

    df_base_tratada, df_resultado_referencia = carregar_execucao_oficial(config)

    validar_estrutura_referencia(
        df_base_tratada=df_base_tratada,
        df_resultado_amostras=df_resultado_referencia,
        colunas_id=colunas_id,
        coluna_cluster=coluna_cluster,
    )

    labels_referencia = df_resultado_referencia[coluna_cluster].to_numpy()

    lista_limiares: list[float | None] = []
    if incluir_referencia_sem_poda:
        lista_limiares.append(None)
    lista_limiares.extend([float(valor) for valor in limiares])

    preparar_experimento_mlflow(
        tracking_uri=config["projeto"]["tracking_uri"],
        nome_experimento=config["projeto"]["nome_experimento_mlflow"],
    )

    pasta_artefatos_base = garantir_pasta(config["saidas"]["pasta_artefatos"])
    pasta_reports = garantir_pasta(config["saidas"]["pasta_reports"])

    resumos_execucoes: list[dict[str, Any]] = []
    registros_features_mantidas: list[dict[str, Any]] = []
    registros_features_removidas: list[dict[str, Any]] = []

    print("=" * 80)
    print("INÍCIO DA EXECUÇÃO - FASE 2D: SENSIBILIDADE À MULTICOLINEARIDADE")
    print("=" * 80)
    print(f"Arquivo de configuração: {caminho_config}")
    print(f"Limiar(es): {lista_limiares}")
    print(f"Seeds UMAP: {seeds_umap}")
    print()

    indice_execucao = 0

    for limiar_correlacao_abs in lista_limiares:
        for seed_umap in seeds_umap:
            indice_execucao += 1

            nome_limiar = "sem_poda" if limiar_correlacao_abs is None else f"tau_{str(limiar_correlacao_abs).replace('.', 'p')}"
            nome_execucao = f"exec_{indice_execucao:03d}__{nome_limiar}__seed{seed_umap}"

            print("-" * 80)
            print(f"Execução {indice_execucao:03d}")
            print(f"Nome: {nome_execucao}")
            print(f"Limiar: {limiar_correlacao_abs}")
            print(f"Seed UMAP: {seed_umap}")

            (
                df_features,
                df_numerico_podado,
                features_mantidas,
                features_removidas,
                df_componentes,
                df_pares,
            ) = montar_features_com_poda(
                df_base_tratada=df_base_tratada,
                colunas_id=colunas_id,
                colunas_onehot=colunas_onehot,
                tipo_normalizacao=tipo_normalizacao,
                metodo_correlacao=metodo_correlacao,
                limiar_correlacao_abs=limiar_correlacao_abs,
            )

            (
                df_embedding,
                labels_preditos,
                metricas_avaliacao,
                metadados_reducer,
                metadados_clusterizacao,
            ) = executar_modelagem_oficial(
                df_features=df_features,
                config=config,
                seed_umap=seed_umap,
                config_score=config_score,
            )

            metricas_concordancia = calcular_ami_nmi(
                labels_referencia=labels_referencia,
                labels_preditos=labels_preditos,
            )

            df_resultado_amostras = montar_resultado_amostras(
                df_resultado_referencia=df_resultado_referencia,
                df_embedding=df_embedding,
                labels_preditos=labels_preditos,
                colunas_id=colunas_id,
            )

            resumo_execucao = {
                "fase_experimental": config["execucao"]["fase_experimental"],
                "nome_execucao": nome_execucao,
                "indice_execucao": indice_execucao,
                "seed_umap": seed_umap,
                "limiar_correlacao_abs": limiar_correlacao_abs,
                "metodo_correlacao": metodo_correlacao,
                "n_amostras": int(len(df_features)),
                "n_features_modelagem": int(df_features.shape[1]),
                "n_features_numericas_apos_poda": int(df_numerico_podado.shape[1]),
                "n_features_mantidas": int(len(features_mantidas)),
                "n_features_removidas": int(len(features_removidas)),
                "n_pares_correlacionados": int(len(df_pares)),
                **metricas_avaliacao,
                **metricas_concordancia,
                **metadados_reducer,
                **metadados_clusterizacao,
            }

            resultado_execucao = {
                "df_resultado_amostras": df_resultado_amostras,
                "df_embedding": df_embedding,
                "df_base_analitica_tratada": df_base_tratada,
                "resumo_execucao": resumo_execucao,
            }

            pasta_execucao = pasta_artefatos_base / nome_execucao

            with mlflow.start_run(run_name=nome_execucao):
                mlflow.set_tag("fase_experimental", config["execucao"]["fase_experimental"])
                mlflow.set_tag("nome_execucao", nome_execucao)
                mlflow.set_tag("limiar_correlacao_abs", str(limiar_correlacao_abs))
                mlflow.set_tag("seed_umap", seed_umap)

                parametros_execucao = {
                    "fase_experimental": config["execucao"]["fase_experimental"],
                    "limiar_correlacao_abs": limiar_correlacao_abs,
                    "metodo_correlacao": metodo_correlacao,
                    "seed_umap": seed_umap,
                    "normalizacao_tipo": tipo_normalizacao,
                    "onehot_colunas": ",".join(colunas_onehot),
                    "reducer_name": config["modelagem_oficial"]["reducer"]["name"],
                    "clusterer_name": config["modelagem_oficial"]["clusterer"]["name"],
                    "n_features_mantidas": len(features_mantidas),
                    "n_features_removidas": len(features_removidas),
                }

                for chave, valor in config["modelagem_oficial"]["reducer"]["params"].items():
                    parametros_execucao[f"reducer__{chave}"] = valor

                for chave, valor in config["modelagem_oficial"]["clusterer"]["params"].items():
                    parametros_execucao[f"clusterer__{chave}"] = valor

                registrar_parametros_mlflow(parametros_execucao)

                registrar_metricas_mlflow(
                    metricas_avaliacao=metricas_avaliacao,
                    metadados_preprocessamento={
                        "n_colunas_saida": int(df_features.shape[1]),
                    },
                    metadados_reducer=metadados_reducer,
                    metadados_clusterizacao=metadados_clusterizacao,
                )

                mlflow.log_metric("ami", metricas_concordancia["ami"])
                mlflow.log_metric("nmi", metricas_concordancia["nmi"])
                mlflow.log_metric("n_features_mantidas", float(len(features_mantidas)))
                mlflow.log_metric("n_features_removidas", float(len(features_removidas)))
                mlflow.log_metric("n_pares_correlacionados", float(len(df_pares)))

                caminhos_artefatos = salvar_artefatos_locais_execucao(
                    resultado_execucao=resultado_execucao,
                    pasta_execucao=pasta_execucao,
                )

                caminhos_complementares = salvar_artefatos_complementares(
                    pasta_execucao=pasta_execucao,
                    features_mantidas=features_mantidas,
                    features_removidas=features_removidas,
                    df_componentes=df_componentes,
                    df_pares=df_pares,
                    limiar_correlacao_abs=limiar_correlacao_abs,
                )

                caminhos_artefatos.update(caminhos_complementares)
                registrar_artefatos_mlflow(caminhos_artefatos)

            resumos_execucoes.append(resumo_execucao)

            registros_features_mantidas.extend(
                [
                    {
                        "limiar_correlacao_abs": limiar_correlacao_abs,
                        "seed_umap": seed_umap,
                        "feature": feature,
                        "status": "mantida",
                    }
                    for feature in features_mantidas
                ]
            )

            registros_features_removidas.extend(
                [
                    {
                        "limiar_correlacao_abs": limiar_correlacao_abs,
                        "seed_umap": seed_umap,
                        "feature": feature,
                        "status": "removida",
                    }
                    for feature in features_removidas
                ]
            )

            print(f"AMI: {metricas_concordancia['ami']:.6f}")
            print(f"NMI: {metricas_concordancia['nmi']:.6f}")
            print(f"n_clusters: {metadados_clusterizacao.get('n_clusters')}")
            print(f"noise_pct: {metricas_avaliacao.get('noise_pct')}")
            print(f"score_final: {metricas_avaliacao.get('score_final')}")
            print(f"Features mantidas: {len(features_mantidas)}")
            print(f"Features removidas: {len(features_removidas)}")
            print()

    df_resumo_execucoes = pd.DataFrame(resumos_execucoes)
    df_resumo_por_limiar = consolidar_resumo_por_limiar(df_resumo_execucoes)

    df_features_mantidas = pd.DataFrame(registros_features_mantidas)
    df_features_removidas = pd.DataFrame(registros_features_removidas)

    caminho_resumo_execucoes = pasta_artefatos_base / "resumo_fase2d_execucoes.csv"
    caminho_resumo_por_limiar = pasta_reports / "resumo_fase2d_por_limiar.csv"
    caminho_features_mantidas = pasta_reports / "features_mantidas_por_limiar.csv"
    caminho_features_removidas = pasta_reports / "features_removidas_por_limiar.csv"

    df_resumo_execucoes.to_csv(caminho_resumo_execucoes, index=False, encoding="utf-8-sig")
    df_resumo_por_limiar.to_csv(caminho_resumo_por_limiar, index=False, encoding="utf-8-sig")
    df_features_mantidas.to_csv(caminho_features_mantidas, index=False, encoding="utf-8-sig")
    df_features_removidas.to_csv(caminho_features_removidas, index=False, encoding="utf-8-sig")

    caminhos_graficos: dict[str, Path] = {}
    caminhos_graficos.update(
        gerar_grafico_ami_por_limiar(
            df_resumo_execucoes=df_resumo_execucoes,
            pasta_reports=pasta_reports,
        )
    )
    caminhos_graficos.update(
        gerar_grafico_nmi_por_limiar(
            df_resumo_execucoes=df_resumo_execucoes,
            pasta_reports=pasta_reports,
        )
    )
    caminhos_graficos.update(
        gerar_grafico_score_final_por_limiar(
            df_resumo_execucoes=df_resumo_execucoes,
            pasta_reports=pasta_reports,
        )
    )
    caminhos_graficos.update(
        gerar_grafico_noise_clusters_por_limiar(
            df_resumo_execucoes=df_resumo_execucoes,
            pasta_reports=pasta_reports,
        )
    )

    

    print("=" * 80)
    print("EXECUÇÃO FINALIZADA - FASE 2D")
    print("=" * 80)
    print(f"Resumo detalhado salvo em: {caminho_resumo_execucoes}")
    print(f"Resumo por limiar salvo em: {caminho_resumo_por_limiar}")
    print(f"Features mantidas salvas em: {caminho_features_mantidas}")
    print(f"Features removidas salvas em: {caminho_features_removidas}")
    print()

    print(f"Gráficos salvos em: {pasta_reports}")
    print()
    for nome_arquivo, caminho_arquivo in caminhos_graficos.items():
        print(f"{nome_arquivo}: {caminho_arquivo}")

    return df_resumo_execucoes, df_resumo_por_limiar


def main() -> None:
    parser = criar_parser_argumentos()
    args = parser.parse_args()
    executar_fase2d(args.config)


if __name__ == "__main__":
    main()