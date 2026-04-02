from __future__ import annotations

from pathlib import Path

from src.river_level.avaliacao import executar_avaliacao_baseline
from src.river_level.treino import executar_treinamento_baseline
from src.utils.io_utils import ler_yaml


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


def executar_pipeline_treino_avaliacao(
    caminho_config: str | Path,
    executar_treino: bool = True,
    executar_avaliacao: bool = True,
) -> dict:
    """
    Núcleo mínimo de orquestração do Caso 2.

    Nesta etapa 2C, a função:
    - carrega a configuração;
    - valida sua estrutura;
    - extrai um resumo padronizado da execução;
    - conecta treino real;
    - conecta avaliação real.

    O MLflow ainda não é utilizado nesta trilha nesta etapa.
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

    if executar_treino:
        resumo_treinamento = executar_treinamento_baseline(config)

    if executar_avaliacao:
        resumo_avaliacao = executar_avaliacao_baseline(config)

    return {
        "resumo_execucao": resumo_execucao,
        "resumo_treinamento": resumo_treinamento,
        "resumo_avaliacao": resumo_avaliacao,
    }