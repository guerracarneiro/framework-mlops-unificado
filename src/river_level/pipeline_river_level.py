from __future__ import annotations

from pathlib import Path

import mlflow

from src.river_level.avaliacao import (
    executar_avaliacao_baseline,
    montar_caminhos_saida_avaliacao,
)
from src.river_level.treino import (
    executar_treinamento_baseline,
    montar_caminhos_saida_treinamento,
)
from src.utils.io_utils import ler_yaml
from src.utils.mlflow_utils import (
    configurar_mlflow,
    registrar_artefato,
    registrar_metricas,
    registrar_parametros_config,
    registrar_tags_execucao_river_level,
    registrar_varios_artefatos,
)


def validar_blocos_configuracao(config: dict) -> None:
    """
    Valida se a configuração possui os blocos mínimos esperados
    para a execução supervisionada do Caso 2.
    """
    blocos_obrigatorios = [
        "projeto",
        "execucao",
        "modelo",
        "treinamento",
        "saidas",
        "preparo_treino",
    ]

    blocos_ausentes = [bloco for bloco in blocos_obrigatorios if bloco not in config]

    if blocos_ausentes:
        raise ValueError(
            "A configuração não possui todos os blocos obrigatórios. "
            f"Blocos ausentes: {blocos_ausentes}"
        )


def extrair_resumo_execucao(config: dict, caminho_config: str | Path) -> dict:
    """
    Extrai um resumo padronizado da execução para uso no runner
    e nas próximas etapas da orquestração.
    """
    return {
        "caminho_config": str(caminho_config),
        "nome_experimento_mlflow": config["projeto"]["nome_experimento_mlflow"],
        "tracking_uri": config["projeto"]["tracking_uri"],
        "semente_global": int(config["projeto"]["semente_global"]),
        "familia_execucao": config["execucao"]["familia_execucao"],
        "nome_execucao": config["execucao"]["nome_execucao"],
        "baseline_referencia": config["execucao"].get("baseline_referencia"),
        "descricao_execucao": config["execucao"].get("descricao"),
        "tipo_modelo": config["modelo"]["tipo"],
        "caminho_dataset_preparado": config["preparo_treino"]["caminho_dataset_saida"],
        "caminho_scaler_y": config["preparo_treino"]["caminho_scaler_y"],
        "pasta_artefatos": config["saidas"]["pasta_artefatos"],
        "pasta_modelos": config["saidas"]["pasta_modelos"],
        "pasta_relatorios": config["saidas"]["pasta_relatorios"],
    }


def montar_caminhos_artefatos_execucao(config: dict) -> list[str]:
    """
    Monta a lista de artefatos principais gerados pelo treino e pela avaliação
    para registro no MLflow.
    """
    caminhos_treinamento = montar_caminhos_saida_treinamento(config)
    caminhos_avaliacao = montar_caminhos_saida_avaliacao(config)

    return [
        caminhos_treinamento["caminho_modelo_checkpoint"],
        caminhos_treinamento["caminho_log_treinamento"],
        caminhos_treinamento["caminho_resumo_treinamento"],
        caminhos_avaliacao["caminho_resumo_avaliacao"],
        caminhos_avaliacao["caminho_predicoes_teste"],
    ]


def executar_pipeline_treino_avaliacao(
    caminho_config: str | Path,
    executar_treino: bool = True,
    executar_avaliacao: bool = True,
    registrar_mlflow: bool = True,
) -> dict:
    """
    Núcleo de orquestração do Caso 2 para a trilha genérica.

    Nesta etapa, a função:
    - carrega e valida a configuração;
    - executa treino e avaliação;
    - registra a execução no MLflow quando solicitado.
    """
    caminho_config = Path(caminho_config)
    config = ler_yaml(caminho_config)
    config["_caminho_config"] = str(caminho_config)

    validar_blocos_configuracao(config)

    resumo_execucao = extrair_resumo_execucao(
        config=config,
        caminho_config=caminho_config,
    )

    resumo_treinamento = None
    resumo_avaliacao = None
    run_id = None

    def executar_fluxo() -> tuple[dict | None, dict | None]:
        resumo_treino_local = None
        resumo_avaliacao_local = None

        if executar_treino:
            resumo_treino_local = executar_treinamento_baseline(config)

        if executar_avaliacao:
            resumo_avaliacao_local = executar_avaliacao_baseline(config)

        return resumo_treino_local, resumo_avaliacao_local

    if registrar_mlflow:
        configurar_mlflow(
            nome_experimento=config["projeto"]["nome_experimento_mlflow"],
            tracking_uri=config["projeto"]["tracking_uri"],
        )

        with mlflow.start_run(run_name=resumo_execucao["nome_execucao"]):
            registrar_tags_execucao_river_level(config)
            registrar_parametros_config(config)
            registrar_artefato(caminho_config, pasta_destino="config")

            resumo_treinamento, resumo_avaliacao = executar_fluxo()

            if resumo_treinamento is not None:
                registrar_metricas(resumo_treinamento)

            if resumo_avaliacao is not None:
                registrar_metricas(resumo_avaliacao)

            caminhos_artefatos = montar_caminhos_artefatos_execucao(config)
            registrar_varios_artefatos(caminhos_artefatos, pasta_destino="artefatos")

            run_id = mlflow.active_run().info.run_id
    else:
        resumo_treinamento, resumo_avaliacao = executar_fluxo()

    return {
        "resumo_execucao": resumo_execucao,
        "resumo_treinamento": resumo_treinamento,
        "resumo_avaliacao": resumo_avaliacao,
        "run_id": run_id,
    }