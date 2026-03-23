from __future__ import annotations

"""
Runner da replicação da análise de sensibilidade à multicolinearidade
nos moldes do artigo, adaptado ao framework do projeto.

Objetivo:
- usar a base tratada oficial da Fase 2C;
- usar a partição de referência fixada em labels_ref_fase2c.csv;
- aplicar poda por multicolinearidade com correlação por limiar tau;
- manter 1 representante por componente correlacionado, com seleção
  aleatória controlada por seed;
- executar UMAP + HDBSCAN com a configuração oficial;
- comparar cada rodada com a partição de referência via AMI/NMI;
- resumir resultados por tau com IC bootstrap e estabilidade pareada;
- salvar artefatos locais e registrar execuções no MLflow.
"""

import argparse
import json
import random
from itertools import combinations
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
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
from src.clones.run_experimento_clones import garantir_pasta, preparar_experimento_mlflow


def criar_parser_argumentos() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Executa a replicação da análise de multicolinearidade do artigo."
    )
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Caminho do YAML da fase2d_artigo_replicado.",
    )
    return parser


def carregar_yaml(caminho_yaml: str | Path) -> dict[str, Any]:
    caminho_yaml = Path(caminho_yaml)

    if not caminho_yaml.exists():
        raise FileNotFoundError(f"Arquivo YAML não encontrado: {caminho_yaml}")

    with caminho_yaml.open("r", encoding="utf-8") as arquivo:
        return yaml.safe_load(arquivo)


def carregar_json(caminho_json: str | Path) -> dict[str, Any]:
    caminho_json = Path(caminho_json)

    if not caminho_json.exists():
        raise FileNotFoundError(f"Arquivo JSON não encontrado: {caminho_json}")

    with caminho_json.open("r", encoding="utf-8") as arquivo:
        return json.load(arquivo)


def carregar_cfg_score_padrao() -> dict[str, Any]:
    """
    Carrega a configuração padrão do score final do projeto.

    Quando o arquivo não existir, usa uma configuração mínima compatível.
    """
    caminho_score = Path("experiments/clones/criterios/score_final.yaml")

    if caminho_score.exists():
        with caminho_score.open("r", encoding="utf-8") as arquivo:
            return yaml.safe_load(arquivo)

    return {
        "pesos": {
            "silhouette": 1.0,
            "dbcv": 1.0,
            "db_inv": 1.0,
        },
        "penalidade_ruido": {
            "limiar": 0.20,
            "peso": 1.0,
            "expoente": 2,
        },
        "penalidade_k": {
            "faixa_ideal": [3, 8],
            "peso": 0.05,
        },
    }


def normalizar_metodo_correlacao(valor: str) -> str:
    metodo = str(valor).strip().lower()

    aliases = {
        "pearson": "pearson",
        "spearman": "spearman",
        "sperman": "spearman",
    }

    if metodo not in aliases:
        raise ValueError(
            f"Método de correlação não suportado: {valor}. "
            "Valores aceitos: 'pearson' ou 'spearman'."
        )

    return aliases[metodo]


def validar_configuracao(config: dict[str, Any]) -> None:
    config["suites"]["corr_pruning_sensitivity"]["metodo"] = normalizar_metodo_correlacao(
        config["suites"]["corr_pruning_sensitivity"]["metodo"]
    )

    if "saidas" not in config:
        raise ValueError("Bloco 'saidas' não encontrado no YAML.")

    if "base" not in config:
        raise ValueError("Bloco 'base' não encontrado no YAML.")

    if "referencia" not in config:
        raise ValueError("Bloco 'referencia' não encontrado no YAML.")


def carregar_base_tratada(config: dict[str, Any]) -> pd.DataFrame:
    pasta_execucao = Path(config["base"]["pasta_execucao_oficial"])
    arquivo_base = pasta_execucao / config["base"]["arquivo_base_tratada"]

    if not arquivo_base.exists():
        raise FileNotFoundError(f"Arquivo base tratada não encontrado: {arquivo_base}")

    return pd.read_csv(arquivo_base, encoding="utf-8-sig")


def carregar_labels_ref(config: dict[str, Any]) -> pd.DataFrame:
    caminho_labels = Path(config["referencia"]["labels_ref_csv"])

    if not caminho_labels.exists():
        raise FileNotFoundError(f"Arquivo labels_ref não encontrado: {caminho_labels}")

    df_labels = pd.read_csv(caminho_labels, encoding="utf-8-sig")

    colunas_obrigatorias = ["TT", "ordem_no_tt", "MATGEN", "label_ref", "row_id"]
    for coluna in colunas_obrigatorias:
        if coluna not in df_labels.columns:
            raise ValueError(f"Coluna obrigatória ausente em labels_ref_csv: {coluna}")

    return df_labels


def carregar_config_melhor(config: dict[str, Any]) -> dict[str, Any]:
    caminho_config = Path(config["referencia"]["config_melhor_json"])
    return carregar_json(caminho_config)


def montar_base_alinhada_artigo(
    df_base_tratada: pd.DataFrame,
    df_labels_ref: pd.DataFrame,
) -> pd.DataFrame:
    """
    Monta uma base alinhada ao protocolo do artigo.

    Como a base tratada oficial não contém os identificadores, os identificadores
    e a partição de referência são acoplados pela ordem, que já foi preservada
    a partir da execução oficial usada para gerar labels_ref_fase2c.csv.
    """
    if len(df_base_tratada) != len(df_labels_ref):
        raise ValueError(
            "A base tratada e labels_ref possuem quantidades de linhas diferentes. "
            f"base={len(df_base_tratada)} | labels_ref={len(df_labels_ref)}"
        )

    df_base = df_base_tratada.reset_index(drop=True).copy()
    df_ref = df_labels_ref.reset_index(drop=True).copy()

    df_saida = pd.concat(
        [
            df_ref[["TT", "ordem_no_tt", "MATGEN", "row_id", "label_ref"]].copy(),
            df_base,
        ],
        axis=1,
    )

    serie_ordem_calculada = df_saida.groupby("TT").cumcount()
    if not serie_ordem_calculada.equals(df_saida["ordem_no_tt"]):
        raise ValueError(
            "Falha no alinhamento por TT + ordem_no_tt. "
            "A ordem reconstruída não coincide com labels_ref."
        )

    return df_saida


def obter_colunas_excluidas_do_bloco_numerico(
    id_columns: list[str],
    colunas_onehot: list[str],
) -> list[str]:
    colunas_excluir = set(id_columns)
    colunas_excluir.update(colunas_onehot)
    colunas_excluir.update(["ordem_no_tt", "row_id", "label_ref"])
    return sorted(colunas_excluir)


def extrair_bloco_numerico_modelagem(
    df_base_alinhada: pd.DataFrame,
    colunas_excluir: list[str],
) -> pd.DataFrame:
    df_trabalho = df_base_alinhada.drop(
        columns=[col for col in colunas_excluir if col in df_base_alinhada.columns],
        errors="ignore",
    ).copy()

    df_numerico = selecionar_colunas_numericas(df_trabalho)

    if df_numerico.empty:
        raise ValueError("O bloco numérico de modelagem ficou vazio.")

    return df_numerico


def calcular_matriz_correlacao_absoluta(
    df_numerico: pd.DataFrame,
    metodo: str,
) -> pd.DataFrame:
    df_corr = df_numerico.corr(method=metodo)
    return df_corr.abs()


def identificar_componentes_correlacionados(
    df_corr_abs: pd.DataFrame,
    tau: float,
) -> list[list[str]]:
    colunas = list(df_corr_abs.columns)
    adjacencias: dict[str, set[str]] = {coluna: set() for coluna in colunas}

    for i, coluna_i in enumerate(colunas):
        for j in range(i + 1, len(colunas)):
            coluna_j = colunas[j]
            valor = df_corr_abs.iat[i, j]

            if pd.isna(valor):
                continue

            if float(valor) >= float(tau):
                adjacencias[coluna_i].add(coluna_j)
                adjacencias[coluna_j].add(coluna_i)

    componentes: list[list[str]] = []
    visitados: set[str] = set()

    for coluna in colunas:
        if coluna in visitados:
            continue

        fila = [coluna]
        visitados.add(coluna)
        componente = [coluna]

        while fila:
            atual = fila.pop()
            for vizinha in sorted(adjacencias[atual]):
                if vizinha not in visitados:
                    visitados.add(vizinha)
                    fila.append(vizinha)
                    componente.append(vizinha)

        componentes.append(sorted(componente))

    return componentes


def selecionar_representantes_por_seed(
    componentes: list[list[str]],
    seed_selecao: int,
) -> tuple[list[str], pd.DataFrame]:
    """
    Mantém 1 representante por componente.

    Regra:
    - componente de tamanho 1: mantém a própria variável;
    - componente maior: escolhe 1 representante aleatoriamente com seed controlada.
    """
    gerador = random.Random(seed_selecao)

    representantes: list[str] = []
    registros: list[dict[str, Any]] = []

    for indice_componente, componente in enumerate(componentes, start=1):
        if len(componente) == 1:
            representante = componente[0]
        else:
            representante = gerador.choice(sorted(componente))

        representantes.append(representante)

        for feature in componente:
            registros.append(
                {
                    "indice_componente": indice_componente,
                    "tamanho_componente": len(componente),
                    "feature": feature,
                    "representante": representante,
                    "mantida": int(feature == representante),
                    "seed_selecao": seed_selecao,
                }
            )

    df_componentes = pd.DataFrame(registros)
    representantes = sorted(representantes)

    return representantes, df_componentes


def extrair_pares_correlacionados(
    df_corr_abs: pd.DataFrame,
    tau: float,
) -> pd.DataFrame:
    registros: list[dict[str, Any]] = []
    colunas = list(df_corr_abs.columns)

    for i, coluna_i in enumerate(colunas):
        for j in range(i + 1, len(colunas)):
            coluna_j = colunas[j]
            valor = df_corr_abs.iat[i, j]

            if pd.isna(valor):
                continue

            if float(valor) >= float(tau):
                registros.append(
                    {
                        "feature_1": coluna_i,
                        "feature_2": coluna_j,
                        "corr_abs": float(valor),
                        "tau": float(tau),
                    }
                )

    return pd.DataFrame(registros)


def montar_features_modelagem_rodada(
    df_base_alinhada: pd.DataFrame,
    id_columns: list[str],
    colunas_onehot: list[str],
    normalizacao_tipo: str,
    metodo_correlacao: str,
    tau: float,
    seed_selecao: int,
) -> tuple[pd.DataFrame, list[str], list[str], pd.DataFrame, pd.DataFrame]:
    colunas_excluir = obter_colunas_excluidas_do_bloco_numerico(
        id_columns=id_columns,
        colunas_onehot=colunas_onehot,
    )

    df_numerico = extrair_bloco_numerico_modelagem(
        df_base_alinhada=df_base_alinhada,
        colunas_excluir=colunas_excluir,
    )

    df_corr_abs = calcular_matriz_correlacao_absoluta(
        df_numerico=df_numerico,
        metodo=metodo_correlacao,
    )

    componentes = identificar_componentes_correlacionados(
        df_corr_abs=df_corr_abs,
        tau=tau,
    )

    features_mantidas, df_componentes = selecionar_representantes_por_seed(
        componentes=componentes,
        seed_selecao=seed_selecao,
    )

    features_removidas = sorted(
        [col for col in df_numerico.columns if col not in set(features_mantidas)]
    )

    df_pares = extrair_pares_correlacionados(
        df_corr_abs=df_corr_abs,
        tau=tau,
    )

    df_numerico_podado = df_numerico[features_mantidas].copy()

    df_numerico_normalizado, _ = normalizar_dados(
        df_numerico_podado,
        tipo=normalizacao_tipo,
    )

    df_onehot = aplicar_onehot(
        df_base_alinhada,
        colunas=colunas_onehot,
    )

    df_features = concatenar_blocos_modelagem(
        df_numerico=df_numerico_normalizado,
        df_onehot=df_onehot,
    )

    return (
        df_features,
        features_mantidas,
        features_removidas,
        df_componentes,
        df_pares,
    )


def executar_umap_hdbscan(
    df_features: pd.DataFrame,
    config_execucao: dict[str, Any],
    seed_umap: int,
    cfg_score: dict[str, Any],
) -> tuple[pd.DataFrame, np.ndarray, dict[str, Any], dict[str, Any], dict[str, Any]]:
    umap_params = dict(config_execucao["umap_params"])
    hdbscan_params = dict(config_execucao["hdbscan_params"])

    umap_params["random_state"] = int(seed_umap)

    config_reducer = {
        "name": "umap",
        "params": umap_params,
    }

    config_clusterer = {
        "name": "hdbscan",
        "params": hdbscan_params,
    }

    matriz_reduzida, _, metadados_reducer = aplicar_reducao_dimensionalidade(
        df_features=df_features,
        config_reducer=config_reducer,
    )

    labels, _, metadados_clusterizacao = aplicar_clusterizacao(
        matriz_reduzida=matriz_reduzida,
        config_clusterer=config_clusterer,
    )

    metricas_avaliacao = avaliar_resultado_clusterizacao(
        matriz_reduzida=matriz_reduzida,
        labels=labels,
        tipo_clusterizador="hdbscan",
        cfg_score=cfg_score,
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
    labels_ref: np.ndarray,
    labels_preditos: np.ndarray,
) -> dict[str, float]:
    return {
        "ami": float(adjusted_mutual_info_score(labels_ref, labels_preditos)),
        "nmi": float(normalized_mutual_info_score(labels_ref, labels_preditos)),
    }


def montar_resultado_amostras_rodada(
    df_base_alinhada: pd.DataFrame,
    df_embedding: pd.DataFrame,
    labels_preditos: np.ndarray,
) -> pd.DataFrame:
    df_ids = df_base_alinhada[
        ["TT", "ordem_no_tt", "MATGEN", "row_id", "label_ref"]
    ].reset_index(drop=True)

    df_resultado = pd.concat(
        [
            df_ids,
            df_embedding.reset_index(drop=True),
            pd.DataFrame({"cluster": labels_preditos}).reset_index(drop=True),
        ],
        axis=1,
    )

    return df_resultado


def bootstrap_ic95_media(
    valores: list[float],
    n_bootstrap: int,
    seed: int,
) -> tuple[float, float, float]:
    if not valores:
        return np.nan, np.nan, np.nan

    vetor = np.asarray(valores, dtype=float)

    if len(vetor) == 1:
        media = float(vetor[0])
        return media, media, media

    gerador = np.random.default_rng(seed)
    medias_boot = []

    for _ in range(int(n_bootstrap)):
        amostra = gerador.choice(vetor, size=len(vetor), replace=True)
        medias_boot.append(float(np.mean(amostra)))

    media = float(np.mean(vetor))
    ic_inf = float(np.quantile(medias_boot, 0.025))
    ic_sup = float(np.quantile(medias_boot, 0.975))

    return media, ic_inf, ic_sup


def calcular_estabilidade_pareada(
    lista_labels: list[np.ndarray],
) -> dict[str, float]:
    if len(lista_labels) < 2:
        return {
            "ami_pair_mean": np.nan,
            "ami_pair_std": np.nan,
            "nmi_pair_mean": np.nan,
            "nmi_pair_std": np.nan,
            "n_pairs": 0,
        }

    valores_ami: list[float] = []
    valores_nmi: list[float] = []

    for labels_a, labels_b in combinations(lista_labels, 2):
        valores_ami.append(
            float(adjusted_mutual_info_score(labels_a, labels_b))
        )
        valores_nmi.append(
            float(normalized_mutual_info_score(labels_a, labels_b))
        )

    return {
        "ami_pair_mean": float(np.mean(valores_ami)),
        "ami_pair_std": float(np.std(valores_ami, ddof=0)),
        "nmi_pair_mean": float(np.mean(valores_nmi)),
        "nmi_pair_std": float(np.std(valores_nmi, ddof=0)),
        "n_pairs": int(len(valores_ami)),
    }


def gerar_resumo_por_tau(
    df_runs: pd.DataFrame,
    labels_por_tau: dict[float, list[np.ndarray]],
    bootstrap_n: int,
    seed_base: int,
) -> pd.DataFrame:
    registros_resumo: list[dict[str, Any]] = []

    for tau in sorted(df_runs["tau"].unique()):
        df_tau = df_runs[df_runs["tau"] == tau].copy()

        ami_media, ami_ic_inf, ami_ic_sup = bootstrap_ic95_media(
            valores=df_tau["ami"].tolist(),
            n_bootstrap=bootstrap_n,
            seed=seed_base + int(tau * 1000),
        )

        nmi_media, nmi_ic_inf, nmi_ic_sup = bootstrap_ic95_media(
            valores=df_tau["nmi"].tolist(),
            n_bootstrap=bootstrap_n,
            seed=seed_base + 10000 + int(tau * 1000),
        )

        estabilidade = calcular_estabilidade_pareada(
            lista_labels=labels_por_tau.get(float(tau), []),
        )

        registros_resumo.append(
            {
                "tau": float(tau),
                "metodo_correlacao": df_tau["metodo_correlacao"].iloc[0],
                "n_rodadas": int(len(df_tau)),
                "ami_media": ami_media,
                "ami_std": float(df_tau["ami"].std(ddof=0)) if len(df_tau) > 1 else 0.0,
                "ami_ic_inf": ami_ic_inf,
                "ami_ic_sup": ami_ic_sup,
                "nmi_media": nmi_media,
                "nmi_std": float(df_tau["nmi"].std(ddof=0)) if len(df_tau) > 1 else 0.0,
                "nmi_ic_inf": nmi_ic_inf,
                "nmi_ic_sup": nmi_ic_sup,
                "score_final_media": float(df_tau["score_final"].mean()),
                "noise_pct_media": float(df_tau["noise_pct"].mean()),
                "n_clusters_media": float(df_tau["n_clusters"].mean()),
                "n_features_mantidas_media": float(df_tau["n_features_mantidas"].mean()),
                "n_features_removidas_media": float(df_tau["n_features_removidas"].mean()),
                "ami_min": float(df_tau["ami"].min()),
                "ami_max": float(df_tau["ami"].max()),
                "nmi_min": float(df_tau["nmi"].min()),
                "nmi_max": float(df_tau["nmi"].max()),
                **estabilidade,
            }
        )

    return pd.DataFrame(registros_resumo)


def gerar_grafico_sensibilidade(
    df_resumo_tau: pd.DataFrame,
    pasta_reports: Path,
    gerar_pdf: bool,
) -> dict[str, Path]:
    fig, ax = plt.subplots(figsize=(10, 6))

    ax.errorbar(
        df_resumo_tau["tau"],
        df_resumo_tau["ami_media"],
        yerr=[
            df_resumo_tau["ami_media"] - df_resumo_tau["ami_ic_inf"],
            df_resumo_tau["ami_ic_sup"] - df_resumo_tau["ami_media"],
        ],
        marker="o",
        capsize=4,
        label="AMI",
    )

    ax.errorbar(
        df_resumo_tau["tau"],
        df_resumo_tau["nmi_media"],
        yerr=[
            df_resumo_tau["nmi_media"] - df_resumo_tau["nmi_ic_inf"],
            df_resumo_tau["nmi_ic_sup"] - df_resumo_tau["nmi_media"],
        ],
        marker="s",
        capsize=4,
        label="NMI",
    )

    ax.set_xlabel("Limiar tau")
    ax.set_ylabel("Concordância com a partição de referência")
    ax.set_title("Sensibilidade à multicolinearidade - replicação do artigo")
    ax.grid(True, alpha=0.3)
    ax.legend()

    caminho_png = pasta_reports / "grafico_ami_nmi_sensibilidade.png"
    fig.tight_layout()
    fig.savefig(caminho_png, dpi=300, bbox_inches="tight")

    caminhos = {
        "grafico_ami_nmi_png": caminho_png,
    }

    if gerar_pdf:
        caminho_pdf = pasta_reports / "grafico_ami_nmi_sensibilidade.pdf"
        fig.savefig(caminho_pdf, bbox_inches="tight")
        caminhos["grafico_ami_nmi_pdf"] = caminho_pdf

    plt.close(fig)
    return caminhos


def salvar_artefatos_rodada(
    pasta_execucao: Path,
    df_resultado_amostras: pd.DataFrame,
    df_embedding: pd.DataFrame,
    features_mantidas: list[str],
    features_removidas: list[str],
    df_componentes: pd.DataFrame,
    df_pares: pd.DataFrame,
    resumo_rodada: dict[str, Any],
) -> dict[str, Path]:
    pasta_execucao = garantir_pasta(pasta_execucao)

    caminhos: dict[str, Path] = {}

    caminho_resultado = pasta_execucao / "resultado_amostras.csv"
    df_resultado_amostras.to_csv(caminho_resultado, index=False, encoding="utf-8-sig")
    caminhos["resultado_amostras_csv"] = caminho_resultado

    caminho_embedding = pasta_execucao / "embedding.csv"
    df_embedding.to_csv(caminho_embedding, index=False, encoding="utf-8-sig")
    caminhos["embedding_csv"] = caminho_embedding

    caminho_mantidas = pasta_execucao / "features_mantidas.csv"
    pd.DataFrame({"feature": features_mantidas, "status": "mantida"}).to_csv(
        caminho_mantidas,
        index=False,
        encoding="utf-8-sig",
    )
    caminhos["features_mantidas_csv"] = caminho_mantidas

    caminho_removidas = pasta_execucao / "features_removidas.csv"
    pd.DataFrame({"feature": features_removidas, "status": "removida"}).to_csv(
        caminho_removidas,
        index=False,
        encoding="utf-8-sig",
    )
    caminhos["features_removidas_csv"] = caminho_removidas

    caminho_componentes = pasta_execucao / "componentes_correlacionados.csv"
    df_componentes.to_csv(caminho_componentes, index=False, encoding="utf-8-sig")
    caminhos["componentes_correlacionados_csv"] = caminho_componentes

    caminho_pares = pasta_execucao / "pares_correlacionados.csv"
    df_pares.to_csv(caminho_pares, index=False, encoding="utf-8-sig")
    caminhos["pares_correlacionados_csv"] = caminho_pares

    caminho_resumo = pasta_execucao / "resumo_rodada.json"
    with caminho_resumo.open("w", encoding="utf-8") as arquivo:
        json.dump(resumo_rodada, arquivo, indent=2, ensure_ascii=False, default=str)
    caminhos["resumo_rodada_json"] = caminho_resumo

    return caminhos


def registrar_artefatos_mlflow(caminhos: dict[str, Path]) -> None:
    for caminho in caminhos.values():
        if caminho.exists():
            mlflow.log_artifact(str(caminho))


def executar_replicacao_artigo(
    caminho_config: str | Path,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    config = carregar_yaml(caminho_config)
    validar_configuracao(config)

    cfg_score = carregar_cfg_score_padrao()
    cfg_execucao = carregar_config_melhor(config)["config_execucao"]

    df_base_tratada = carregar_base_tratada(config)
    df_labels_ref = carregar_labels_ref(config)

    df_base_alinhada = montar_base_alinhada_artigo(
        df_base_tratada=df_base_tratada,
        df_labels_ref=df_labels_ref,
    )

    labels_ref = df_base_alinhada["label_ref"].to_numpy()

    preparar_experimento_mlflow(
        tracking_uri=config["projeto"]["tracking_uri"],
        nome_experimento=config["projeto"]["nome_experimento_mlflow"],
    )

    pasta_artefatos = garantir_pasta(config["saidas"]["pasta_artefatos"])
    pasta_reports = garantir_pasta(config["saidas"]["pasta_reports"])

    metodo_correlacao = config["suites"]["corr_pruning_sensitivity"]["metodo"]
    limiares_tau = [float(v) for v in config["suites"]["corr_pruning_sensitivity"]["limiares_tau"]]
    repeticoes_por_tau = int(config["suites"]["corr_pruning_sensitivity"]["repeticoes_por_tau"])
    bootstrap_n = int(config["suites"]["corr_pruning_sensitivity"]["bootstrap_n"])
    gerar_pdf = bool(config["suites"]["corr_pruning_sensitivity"]["gerar_pdf"])

    seeds_estabilidade = [int(v) for v in config["execucao"]["seeds_estabilidade"]]
    seed_remocao_base = int(config["suites"]["corr_pruning_sensitivity"]["seed_remocao_base"])
    id_columns = list(config["base"]["id_columns"])

    registros_runs: list[dict[str, Any]] = []
    registros_features: list[dict[str, Any]] = []
    labels_por_tau: dict[float, list[np.ndarray]] = {tau: [] for tau in limiares_tau}

    print("=" * 80)
    print("INÍCIO DA REPLICAÇÃO DO ARTIGO - MULTICOLINEARIDADE")
    print("=" * 80)
    print(f"Configuração: {caminho_config}")
    print(f"Método de correlação: {metodo_correlacao}")
    print(f"Limiar(es) tau: {limiares_tau}")
    print(f"Repetições por tau: {repeticoes_por_tau}")
    print(f"Seeds UMAP disponíveis: {seeds_estabilidade}")
    print()

    indice_global = 0

    for tau in limiares_tau:
        for repeticao in range(repeticoes_por_tau):
            indice_global += 1

            seed_umap = seeds_estabilidade[repeticao % len(seeds_estabilidade)]
            seed_selecao = seed_remocao_base + repeticao

            nome_execucao = (
                f"exec_{indice_global:03d}"
                f"__tau_{str(tau).replace('.', 'p')}"
                f"__rep_{repeticao + 1:02d}"
                f"__seed_umap_{seed_umap}"
                f"__seed_sel_{seed_selecao}"
            )

            print("-" * 80)
            print(f"Execução: {nome_execucao}")
            print(f"tau={tau} | repetição={repeticao + 1} | seed_umap={seed_umap} | seed_selecao={seed_selecao}")

            (
                df_features,
                features_mantidas,
                features_removidas,
                df_componentes,
                df_pares,
            ) = montar_features_modelagem_rodada(
                df_base_alinhada=df_base_alinhada,
                id_columns=id_columns,
                colunas_onehot=cfg_execucao["onehot_colunas"],
                normalizacao_tipo=cfg_execucao["normalizacao_tipo"],
                metodo_correlacao=metodo_correlacao,
                tau=tau,
                seed_selecao=seed_selecao,
            )

            (
                df_embedding,
                labels_preditos,
                metricas_avaliacao,
                metadados_reducer,
                metadados_clusterizacao,
            ) = executar_umap_hdbscan(
                df_features=df_features,
                config_execucao=cfg_execucao,
                seed_umap=seed_umap,
                cfg_score=cfg_score,
            )

            metricas_concordancia = calcular_ami_nmi(
                labels_ref=labels_ref,
                labels_preditos=labels_preditos,
            )

            labels_por_tau[tau].append(np.asarray(labels_preditos))

            df_resultado_amostras = montar_resultado_amostras_rodada(
                df_base_alinhada=df_base_alinhada,
                df_embedding=df_embedding,
                labels_preditos=labels_preditos,
            )

            resumo_rodada = {
                "nome_execucao": nome_execucao,
                "tau": tau,
                "metodo_correlacao": metodo_correlacao,
                "repeticao": repeticao + 1,
                "seed_umap": seed_umap,
                "seed_selecao": seed_selecao,
                "n_amostras": int(len(df_features)),
                "n_features_modelagem": int(df_features.shape[1]),
                "n_features_mantidas": int(len(features_mantidas)),
                "n_features_removidas": int(len(features_removidas)),
                "n_componentes_correlacionados": int(df_componentes["indice_componente"].nunique()) if not df_componentes.empty else 0,
                "n_pares_correlacionados": int(len(df_pares)),
                **metricas_avaliacao,
                **metricas_concordancia,
                **metadados_reducer,
                **metadados_clusterizacao,
            }

            pasta_execucao = pasta_artefatos / "rodadas" / nome_execucao

            with mlflow.start_run(run_name=nome_execucao):
                mlflow.set_tag("fase_experimental", "fase2d_artigo_replicado")
                mlflow.set_tag("tau", str(tau))
                mlflow.set_tag("metodo_correlacao", metodo_correlacao)

                parametros = {
                    "tau": tau,
                    "metodo_correlacao": metodo_correlacao,
                    "repeticao": repeticao + 1,
                    "seed_umap": seed_umap,
                    "seed_selecao": seed_selecao,
                    "normalizacao_tipo": cfg_execucao["normalizacao_tipo"],
                    "onehot_colunas": json.dumps(cfg_execucao["onehot_colunas"], ensure_ascii=False),
                    "imputacao_tipo_origem": cfg_execucao.get("imputacao_tipo"),
                    "n_features_mantidas": len(features_mantidas),
                    "n_features_removidas": len(features_removidas),
                    **{f"umap__{k}": v for k, v in cfg_execucao["umap_params"].items()},
                    **{f"hdbscan__{k}": v for k, v in cfg_execucao["hdbscan_params"].items()},
                }

                parametros_convertidos = {
                    chave: (json.dumps(valor, ensure_ascii=False) if isinstance(valor, (list, dict)) else valor)
                    for chave, valor in parametros.items()
                    if valor is not None
                }
                mlflow.log_params(parametros_convertidos)

                metricas_mlflow = {
                    chave: float(valor)
                    for chave, valor in resumo_rodada.items()
                    if isinstance(valor, (int, float)) and pd.notna(valor)
                }
                if metricas_mlflow:
                    mlflow.log_metrics(metricas_mlflow)

                caminhos_artefatos = salvar_artefatos_rodada(
                    pasta_execucao=pasta_execucao,
                    df_resultado_amostras=df_resultado_amostras,
                    df_embedding=df_embedding,
                    features_mantidas=features_mantidas,
                    features_removidas=features_removidas,
                    df_componentes=df_componentes,
                    df_pares=df_pares,
                    resumo_rodada=resumo_rodada,
                )
                registrar_artefatos_mlflow(caminhos_artefatos)

            registros_runs.append(resumo_rodada)

            registros_features.extend(
                [
                    {
                        "tau": tau,
                        "repeticao": repeticao + 1,
                        "seed_umap": seed_umap,
                        "seed_selecao": seed_selecao,
                        "feature": feature,
                        "status": "mantida",
                    }
                    for feature in features_mantidas
                ]
            )
            registros_features.extend(
                [
                    {
                        "tau": tau,
                        "repeticao": repeticao + 1,
                        "seed_umap": seed_umap,
                        "seed_selecao": seed_selecao,
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
            print(f"features_mantidas: {len(features_mantidas)}")
            print(f"features_removidas: {len(features_removidas)}")
            print()

    df_runs = pd.DataFrame(registros_runs)
    df_features = pd.DataFrame(registros_features)

    df_resumo_tau = gerar_resumo_por_tau(
        df_runs=df_runs,
        labels_por_tau=labels_por_tau,
        bootstrap_n=bootstrap_n,
        seed_base=int(config["projeto"]["semente_global"]),
    )

    caminho_runs = pasta_artefatos / "runs_detalhado.csv"
    caminho_features = pasta_artefatos / "features_por_rodada.csv"
    caminho_metadata = pasta_artefatos / "metadata.json"
    caminho_resumo = pasta_reports / "resumo_limiares.csv"

    df_runs.to_csv(caminho_runs, index=False, encoding="utf-8-sig")
    df_features.to_csv(caminho_features, index=False, encoding="utf-8-sig")
    df_resumo_tau.to_csv(caminho_resumo, index=False, encoding="utf-8-sig")

    metadata = {
        "config_usado": config,
        "config_execucao": cfg_execucao,
        "n_execucoes": int(len(df_runs)),
        "n_limiares_tau": int(len(limiares_tau)),
        "bootstrap_n": bootstrap_n,
    }

    with caminho_metadata.open("w", encoding="utf-8") as arquivo:
        json.dump(metadata, arquivo, indent=2, ensure_ascii=False, default=str)

    caminhos_graficos = gerar_grafico_sensibilidade(
        df_resumo_tau=df_resumo_tau,
        pasta_reports=pasta_reports,
        gerar_pdf=gerar_pdf,
    )

    print("=" * 80)
    print("REPLICAÇÃO DO ARTIGO FINALIZADA")
    print("=" * 80)
    print(f"runs_detalhado.csv: {caminho_runs}")
    print(f"features_por_rodada.csv: {caminho_features}")
    print(f"resumo_limiares.csv: {caminho_resumo}")
    print(f"metadata.json: {caminho_metadata}")
    for nome, caminho in caminhos_graficos.items():
        print(f"{nome}: {caminho}")
    print()

    return df_runs, df_resumo_tau


def main() -> None:
    parser = criar_parser_argumentos()
    args = parser.parse_args()
    executar_replicacao_artigo(args.config)


if __name__ == "__main__":
    main()