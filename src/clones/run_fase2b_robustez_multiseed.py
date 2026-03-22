from __future__ import annotations

"""
Script de execução da Fase 2B do Estudo de Caso 1.

Responsável por:
- carregar a configuração experimental da Fase 2B;
- expandir candidatas por múltiplas seeds;
- executar o pipeline para cada combinação;
- registrar parâmetros, métricas e artefatos no MLflow;
- calcular estabilidade AMI/NMI por candidata;
- salvar resumos consolidados da robustez.
"""

import argparse
import json
from pathlib import Path
from typing import Any

import mlflow
import numpy as np
import pandas as pd
from sklearn.metrics import adjusted_mutual_info_score, normalized_mutual_info_score

from src.clones.config import preparar_configuracoes_fase2b, resumir_configuracao_unitaria
from src.clones.pipeline_clones import executar_pipeline_clones
from src.clones.run_experimento_clones import (
    converter_valor_parametro_mlflow,
    garantir_pasta,
    preparar_experimento_mlflow,
    registrar_artefatos_mlflow,
    registrar_metricas_mlflow,
    salvar_artefatos_locais_execucao,
)


def criar_parser_argumentos() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Executa a Fase 2B do Estudo de Caso 1 (robustez multi-seed)."
    )

    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Caminho do arquivo YAML da fase experimental.",
    )

    return parser


def registrar_parametros_mlflow_fase2b(
    parametros_execucao: dict[str, Any],
    config_unitaria: dict[str, Any],
) -> None:
    """
    Registra parâmetros da execução no MLflow, incluindo metadados extras da Fase 2B.
    """
    parametros_completos = dict(parametros_execucao)

    parametros_completos["hash_configuracao"] = config_unitaria["execucao"].get("hash_configuracao")
    parametros_completos["fase_experimental"] = config_unitaria["execucao"].get("fase_experimental")
    parametros_completos["nome_base_candidata"] = config_unitaria["execucao"].get("nome_base_candidata")
    parametros_completos["seed_execucao"] = config_unitaria["execucao"].get("seed_execucao")

    parametros_convertidos = {
        chave: converter_valor_parametro_mlflow(valor)
        for chave, valor in parametros_completos.items()
        if valor is not None
    }

    if parametros_convertidos:
        mlflow.log_params(parametros_convertidos)


def ordenar_resultado_por_ids(
    df_resultado_amostras: pd.DataFrame,
    colunas_id: list[str],
) -> pd.DataFrame:
    """
    Ordena o resultado por colunas de identificação para garantir comparações
    consistentes entre execuções com seeds diferentes.
    """
    colunas_presentes = [col for col in colunas_id if col in df_resultado_amostras.columns]

    if not colunas_presentes:
        raise ValueError(
            "Nenhuma coluna de identificação da configuração foi encontrada em df_resultado_amostras."
        )

    return df_resultado_amostras.sort_values(colunas_presentes).reset_index(drop=True)


def calcular_estabilidade_multiseed(
    resultados_rotulos: list[dict[str, Any]],
) -> pd.DataFrame:
    """
    Calcula AMI e NMI de cada seed contra a seed de referência dentro de cada candidata.

    A seed de referência é escolhida como:
    - seed 30, se existir;
    - caso contrário, a menor seed disponível.
    """
    linhas_estabilidade: list[dict[str, Any]] = []

    df_resultados = pd.DataFrame(resultados_rotulos)

    for nome_base, grupo in df_resultados.groupby("nome_base_candidata"):
        grupo = grupo.sort_values("seed_execucao").reset_index(drop=True)

        if (grupo["seed_execucao"] == 30).any():
            linha_referencia = grupo[grupo["seed_execucao"] == 30].iloc[0]
        else:
            linha_referencia = grupo.iloc[0]

        labels_referencia = np.asarray(linha_referencia["labels"])
        seed_referencia = int(linha_referencia["seed_execucao"])

        for _, linha in grupo.iterrows():
            seed_atual = int(linha["seed_execucao"])
            labels_atual = np.asarray(linha["labels"])

            ami = adjusted_mutual_info_score(labels_referencia, labels_atual)
            nmi = normalized_mutual_info_score(labels_referencia, labels_atual)

            linhas_estabilidade.append({
                "nome_base_candidata": nome_base,
                "seed_referencia": seed_referencia,
                "seed_comparada": seed_atual,
                "ami_referencia": float(ami),
                "nmi_referencia": float(nmi),
            })

    return pd.DataFrame(linhas_estabilidade)


def consolidar_robustez(
    df_resumo_execucoes: pd.DataFrame,
    df_estabilidade: pd.DataFrame,
) -> pd.DataFrame:
    """
    Consolida as métricas da Fase 2B por candidata.

    Gera média e desvio padrão das métricas principais e adiciona AMI/NMI médios.
    """
    colunas_metricas = [
        "score_final",
        "silhouette",
        "dbcv",
        "davies_bouldin",
        "calinski_harabasz",
        "noise_pct",
        "n_clusters",
    ]

    colunas_disponiveis = [col for col in colunas_metricas if col in df_resumo_execucoes.columns]

    agregacoes: dict[str, list[str]] = {
        coluna: ["mean", "std"]
        for coluna in colunas_disponiveis
    }

    df_metricas = (
        df_resumo_execucoes
        .groupby("nome_base_candidata")
        .agg(agregacoes)
    )

    df_metricas.columns = [
        f"{coluna}_{estatistica}"
        for coluna, estatistica in df_metricas.columns
    ]
    df_metricas = df_metricas.reset_index()

    if not df_estabilidade.empty:
        df_estab = (
            df_estabilidade
            .groupby("nome_base_candidata")
            .agg({
                "ami_referencia": ["mean", "std"],
                "nmi_referencia": ["mean", "std"],
            })
        )
        df_estab.columns = [
            f"{coluna}_{estatistica}"
            for coluna, estatistica in df_estab.columns
        ]
        df_estab = df_estab.reset_index()

        df_metricas = df_metricas.merge(df_estab, on="nome_base_candidata", how="left")

    return df_metricas


def executar_fase2b(
    caminho_config: str | Path,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    configuracoes_unitarias, config_score = preparar_configuracoes_fase2b(caminho_config)

    if not configuracoes_unitarias:
        raise ValueError("Nenhuma configuração unitária foi gerada para a Fase 2B.")

    config_base = configuracoes_unitarias[0]
    config_projeto = config_base["projeto"]
    config_saidas = config_base["saidas"]
    config_execucao = config_base["execucao"]
    colunas_id = config_base["dados"].get("id_columns", [])

    preparar_experimento_mlflow(
        tracking_uri=config_projeto["tracking_uri"],
        nome_experimento=config_projeto["nome_experimento_mlflow"],
    )

    pasta_artefatos_base = garantir_pasta(config_saidas["pasta_artefatos"])

    limite_teste = config_execucao.get("limitar_execucoes_teste")
    if limite_teste is not None:
        configuracoes_unitarias = configuracoes_unitarias[: int(limite_teste)]

    resumos_execucoes: list[dict[str, Any]] = []
    resultados_rotulos: list[dict[str, Any]] = []

    print("=" * 80)
    print("INÍCIO DA EXECUÇÃO - FASE 2B: ROBUSTEZ MULTI-SEED")
    print("=" * 80)
    print(f"Arquivo de configuração: {caminho_config}")
    print(f"Total de execuções planejadas: {len(configuracoes_unitarias)}")
    print()

    for config_unitaria in configuracoes_unitarias:
        resumo_config = resumir_configuracao_unitaria(config_unitaria)
        indice_execucao = resumo_config["indice_execucao"]
        nome_execucao = resumo_config["nome_execucao"]
        nome_base_candidata = config_unitaria["execucao"]["nome_base_candidata"]
        seed_execucao = config_unitaria["execucao"]["seed_execucao"]

        print("-" * 80)
        print(f"Execução {indice_execucao:03d}")
        print(f"Nome: {nome_execucao}")
        print(f"Resumo: {json.dumps(resumo_config, ensure_ascii=False)}")

        pasta_execucao = pasta_artefatos_base / nome_execucao

        with mlflow.start_run(run_name=nome_execucao):
            mlflow.set_tag("fase_experimental", config_unitaria["execucao"]["fase_experimental"])
            mlflow.set_tag("nome_execucao", nome_execucao)
            mlflow.set_tag("nome_base_candidata", nome_base_candidata)
            mlflow.set_tag("seed_execucao", seed_execucao)
            mlflow.set_tag("hash_configuracao", config_unitaria["execucao"].get("hash_configuracao", ""))

            resultado_execucao = executar_pipeline_clones(
                config_unitaria=config_unitaria,
                config_score=config_score,
            )

            registrar_parametros_mlflow_fase2b(
                parametros_execucao=resultado_execucao["parametros_execucao"],
                config_unitaria=config_unitaria,
            )

            registrar_metricas_mlflow(
                metricas_avaliacao=resultado_execucao["metricas_avaliacao"],
                metadados_preprocessamento=resultado_execucao["metadados_preprocessamento"],
                metadados_reducer=resultado_execucao["metadados_reducer"],
                metadados_clusterizacao=resultado_execucao["metadados_clusterizacao"],
            )

            caminhos_artefatos = salvar_artefatos_locais_execucao(
                resultado_execucao=resultado_execucao,
                pasta_execucao=pasta_execucao,
            )

            registrar_artefatos_mlflow(caminhos_artefatos)

            resumo_execucao = dict(resultado_execucao["resumo_execucao"])
            resumo_execucao["nome_base_candidata"] = nome_base_candidata
            resumo_execucao["seed_execucao"] = seed_execucao
            resumo_execucao["hash_configuracao"] = config_unitaria["execucao"].get("hash_configuracao")
            resumos_execucoes.append(resumo_execucao)

            df_resultado_ordenado = ordenar_resultado_por_ids(
                df_resultado_amostras=resultado_execucao["df_resultado_amostras"],
                colunas_id=colunas_id,
            )

            resultados_rotulos.append({
                "nome_base_candidata": nome_base_candidata,
                "seed_execucao": seed_execucao,
                "labels": df_resultado_ordenado["cluster"].tolist(),
            })

            print(f"Score final: {resumo_execucao.get('score_final')}")
            print(f"Artefatos salvos em: {pasta_execucao}")
            print()

    df_resumo_execucoes = pd.DataFrame(resumos_execucoes)
    df_estabilidade = calcular_estabilidade_multiseed(resultados_rotulos)
    df_robustez = consolidar_robustez(df_resumo_execucoes, df_estabilidade)

    caminho_resumo_execucoes = pasta_artefatos_base / "resumo_execucoes_fase2b.csv"
    caminho_estabilidade = pasta_artefatos_base / "estabilidade_multiseed_fase2b.csv"
    caminho_robustez = pasta_artefatos_base / "resumo_robustez_fase2b.csv"

    df_resumo_execucoes.to_csv(caminho_resumo_execucoes, index=False, encoding="utf-8-sig")
    df_estabilidade.to_csv(caminho_estabilidade, index=False, encoding="utf-8-sig")
    df_robustez.to_csv(caminho_robustez, index=False, encoding="utf-8-sig")

    print("=" * 80)
    print("EXECUÇÃO FINALIZADA")
    print("=" * 80)
    print(f"Resumo das execuções salvo em: {caminho_resumo_execucoes}")
    print(f"Resumo de estabilidade salvo em: {caminho_estabilidade}")
    print(f"Resumo de robustez salvo em: {caminho_robustez}")
    print()

    return df_resumo_execucoes, df_estabilidade, df_robustez


def main() -> None:
    parser = criar_parser_argumentos()
    args = parser.parse_args()

    executar_fase2b(caminho_config=args.config)


if __name__ == "__main__":
    main()