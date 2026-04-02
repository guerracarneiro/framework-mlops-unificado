from __future__ import annotations

from pathlib import Path
from typing import Any

import mlflow


def configurar_mlflow(nome_experimento: str, tracking_uri: str) -> None:
    """
    Configura o tracking URI e o experimento ativo do MLflow.
    """
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(nome_experimento)


def achatar_dict(dicionario: dict, prefixo: str = "") -> dict:
    """
    Achata um dicionário aninhado usando prefixos com ponto.

    Exemplo:
    {"modelo": {"dropout": 0.2}} -> {"modelo.dropout": 0.2}
    """
    itens = {}

    for chave, valor in dicionario.items():
        chave_nova = f"{prefixo}.{chave}" if prefixo else str(chave)

        if isinstance(valor, dict):
            itens.update(achatar_dict(valor, chave_nova))
        else:
            itens[chave_nova] = valor

    return itens


def filtrar_valores_logaveis(dicionario: dict[str, Any]) -> dict[str, Any]:
    """
    Mantém apenas valores simples adequados para log como parâmetros.
    """
    valores_aceitos = (str, int, float, bool)

    return {
        chave: valor
        for chave, valor in dicionario.items()
        if isinstance(valor, valores_aceitos) or valor is None
    }


def registrar_parametros_config(config: dict) -> None:
    """
    Registra os parâmetros do YAML de forma achatada no MLflow.
    """
    config_achatada = achatar_dict(config)
    config_logavel = filtrar_valores_logaveis(config_achatada)

    if config_logavel:
        mlflow.log_params(config_logavel)


def registrar_metricas(dicionario_metricas: dict) -> None:
    """
    Registra métricas numéricas no MLflow.
    """
    metricas_logaveis = {}

    for chave, valor in dicionario_metricas.items():
        if isinstance(valor, (int, float)):
            metricas_logaveis[chave] = float(valor)

    if metricas_logaveis:
        mlflow.log_metrics(metricas_logaveis)


def registrar_tags_basicas(config: dict) -> None:
    """
    Registra tags úteis para identificação do experimento.
    """
    mlflow.set_tags(
        {
            "caso_estudo": "river_level",
            "tipo_modelo": str(config.get("modelo", {}).get("tipo", "desconhecido")),
            "pipeline": "baseline_lstm",
        }
    )


def registrar_artefato(caminho_artefato: str | Path, pasta_destino: str | None = None) -> None:
    """
    Registra um arquivo como artefato do run.
    """
    caminho_artefato = Path(caminho_artefato)

    if caminho_artefato.exists():
        mlflow.log_artifact(str(caminho_artefato), artifact_path=pasta_destino)


def registrar_varios_artefatos(caminhos: list[str | Path], pasta_destino: str | None = None) -> None:
    """
    Registra vários arquivos como artefatos do run.
    """
    for caminho in caminhos:
        registrar_artefato(caminho, pasta_destino=pasta_destino)

def registrar_tags_execucao_river_level(config: dict) -> None:
    """
    Registra tags específicas para a trilha genérica do estudo de caso 2.

    Esta função não substitui a estratégia da baseline oficial.
    Ela atende às execuções genéricas de tuning e experimentação.
    """
    execucao_cfg = config.get("execucao", {})
    modelo_cfg = config.get("modelo", {})

    tags = {
        "caso_estudo": "river_level",
        "tipo_modelo": str(modelo_cfg.get("tipo", "desconhecido")),
        "familia_execucao": str(execucao_cfg.get("familia_execucao", "nao_informada")),
        "pipeline": str(execucao_cfg.get("familia_execucao", "nao_informada")),
    }

    baseline_referencia = execucao_cfg.get("baseline_referencia")
    if baseline_referencia is not None:
        tags["baseline_referencia"] = str(baseline_referencia)

    descricao_execucao = execucao_cfg.get("descricao")
    if descricao_execucao is not None:
        tags["descricao_execucao"] = str(descricao_execucao)

    mlflow.set_tags(tags)        