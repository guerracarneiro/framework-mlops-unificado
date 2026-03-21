from __future__ import annotations

"""
Script de execução da Fase 2 do Estudo de Caso 1.

Responsável por:
- carregar a configuração experimental da Fase 2;
- expandir o grid de modelagem;
- executar o pipeline para cada combinação;
- registrar parâmetros, métricas e artefatos no MLflow;
- salvar o resumo consolidado das execuções.
"""

import argparse
import json
from pathlib import Path
from typing import Any

import mlflow
import pandas as pd

from src.clones.config import preparar_configuracoes_fase2, resumir_configuracao_unitaria
from src.clones.run_experimento_clones import (
    converter_valor_parametro_mlflow,
    garantir_pasta,
    preparar_experimento_mlflow,
    registrar_artefatos_mlflow,
    registrar_metricas_mlflow,
    salvar_artefatos_locais_execucao,
)
from src.clones.pipeline_clones import executar_pipeline_clones


def criar_parser_argumentos() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Executa a Fase 2 do Estudo de Caso 1 (tuning inicial UMAP + HDBSCAN)."
    )

    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Caminho do arquivo YAML da fase experimental.",
    )

    return parser


def registrar_parametros_mlflow(
    parametros_execucao: dict[str, Any],
    config_unitaria: dict[str, Any],
) -> None:
    """
    Registra parâmetros da execução no MLflow, incluindo metadados extras da Fase 2.
    """
    parametros_completos = dict(parametros_execucao)

    parametros_completos["hash_configuracao"] = config_unitaria["execucao"].get("hash_configuracao")
    parametros_completos["fase_experimental"] = config_unitaria["execucao"].get("fase_experimental")
    parametros_completos["seed_execucao"] = (
        config_unitaria["modelagem"]["reducer"]["params"].get("random_state")
    )

    parametros_convertidos = {
        chave: converter_valor_parametro_mlflow(valor)
        for chave, valor in parametros_completos.items()
        if valor is not None
    }

    if parametros_convertidos:
        mlflow.log_params(parametros_convertidos)


def executar_fase2(
    caminho_config: str | Path,
) -> pd.DataFrame:
    configuracoes_unitarias, config_score = preparar_configuracoes_fase2(caminho_config)

    if not configuracoes_unitarias:
        raise ValueError("Nenhuma configuração unitária foi gerada para a Fase 2.")

    config_base = configuracoes_unitarias[0]
    config_projeto = config_base["projeto"]
    config_saidas = config_base["saidas"]
    config_execucao = config_base["execucao"]

    preparar_experimento_mlflow(
        tracking_uri=config_projeto["tracking_uri"],
        nome_experimento=config_projeto["nome_experimento_mlflow"],
    )

    pasta_artefatos_base = garantir_pasta(config_saidas["pasta_artefatos"])

    limite_teste = config_execucao.get("limitar_execucoes_teste")
    if limite_teste is not None:
        configuracoes_unitarias = configuracoes_unitarias[: int(limite_teste)]

    resumos_execucoes: list[dict[str, Any]] = []

    print("=" * 80)
    print("INÍCIO DA EXECUÇÃO - FASE 2: TUNING INICIAL UMAP + HDBSCAN")
    print("=" * 80)
    print(f"Arquivo de configuração: {caminho_config}")
    print(f"Total de execuções planejadas: {len(configuracoes_unitarias)}")
    print()

    for config_unitaria in configuracoes_unitarias:
        resumo_config = resumir_configuracao_unitaria(config_unitaria)
        indice_execucao = resumo_config["indice_execucao"]
        nome_execucao = resumo_config["nome_execucao"]

        print("-" * 80)
        print(f"Execução {indice_execucao:03d}")
        print(f"Nome: {nome_execucao}")
        print(f"Resumo: {json.dumps(resumo_config, ensure_ascii=False)}")

        pasta_execucao = pasta_artefatos_base / nome_execucao

        with mlflow.start_run(run_name=nome_execucao):
            mlflow.set_tag("fase_experimental", config_unitaria["execucao"]["fase_experimental"])
            mlflow.set_tag("nome_execucao", nome_execucao)
            mlflow.set_tag("hash_configuracao", config_unitaria["execucao"].get("hash_configuracao", ""))

            resultado_execucao = executar_pipeline_clones(
                config_unitaria=config_unitaria,
                config_score=config_score,
            )

            registrar_parametros_mlflow(
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

            resumo_execucao = resultado_execucao["resumo_execucao"]
            resumo_execucao["hash_configuracao"] = config_unitaria["execucao"].get("hash_configuracao")
            resumos_execucoes.append(resumo_execucao)

            print(f"Score final: {resumo_execucao.get('score_final')}")
            print(f"Artefatos salvos em: {pasta_execucao}")
            print()

    df_resumo_execucoes = pd.DataFrame(resumos_execucoes)

    caminho_resumo_csv = pasta_artefatos_base / "resumo_fase2_tuning_robustez.csv"
    df_resumo_execucoes.to_csv(caminho_resumo_csv, index=False, encoding="utf-8-sig")

    print("=" * 80)
    print("EXECUÇÃO FINALIZADA")
    print("=" * 80)
    print(f"Resumo consolidado salvo em: {caminho_resumo_csv}")
    print()

    return df_resumo_execucoes


def main() -> None:
    parser = criar_parser_argumentos()
    args = parser.parse_args()

    executar_fase2(caminho_config=args.config)


if __name__ == "__main__":
    main()