from __future__ import annotations

"""
Script de execução da Fase 1 do Estudo de Caso 1.

Responsável por:
- carregar a configuração experimental;
- expandir as configurações unitárias da fase;
- executar o pipeline para cada combinação;
- registrar parâmetros, métricas e artefatos no MLflow;
- salvar o resumo consolidado das execuções.

Uso sugerido:
    python -m src.clones.run_experimento_clones --config experiments/clones/fase1_preprocessamento.yaml
"""

import argparse
import json
from pathlib import Path
from typing import Any

import mlflow
import pandas as pd

from src.clones.config import preparar_configuracoes_fase1, resumir_configuracao_unitaria
from src.clones.pipeline_clones import executar_pipeline_clones


def criar_parser_argumentos() -> argparse.ArgumentParser:
    """
    Cria o parser de argumentos da linha de comando.

    Retorno
    -------
    argparse.ArgumentParser
        Parser configurado para execução do experimento.
    """
    parser = argparse.ArgumentParser(
        description="Executa a Fase 1 do Estudo de Caso 1 (preprocessamento)."
    )

    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Caminho do arquivo YAML da fase experimental.",
    )

    return parser


def garantir_pasta(caminho_pasta: str | Path) -> Path:
    """
    Garante a existência de uma pasta e retorna seu caminho como Path.

    Parâmetros
    ----------
    caminho_pasta : str | Path
        Caminho da pasta.

    Retorno
    -------
    Path
        Caminho da pasta criada ou já existente.
    """
    caminho_pasta = Path(caminho_pasta)
    caminho_pasta.mkdir(parents=True, exist_ok=True)
    return caminho_pasta


def preparar_experimento_mlflow(
    tracking_uri: str,
    nome_experimento: str,
) -> None:
    """
    Configura o MLflow para a execução atual.

    Parâmetros
    ----------
    tracking_uri : str
        URI do tracking do MLflow.
    nome_experimento : str
        Nome do experimento no MLflow.
    """
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(nome_experimento)


def converter_valor_parametro_mlflow(valor: Any) -> Any:
    """
    Converte um valor para formato compatível com registro em parâmetros do MLflow.

    Parâmetros
    ----------
    valor : Any
        Valor original.

    Retorno
    -------
    Any
        Valor convertido para formato simples.
    """
    if isinstance(valor, (list, dict, tuple)):
        return json.dumps(valor, ensure_ascii=False)

    return valor


def registrar_parametros_mlflow(
    parametros_execucao: dict[str, Any],
) -> None:
    """
    Registra parâmetros da execução no MLflow.

    Parâmetros
    ----------
    parametros_execucao : dict[str, Any]
        Dicionário de parâmetros da execução.
    """
    parametros_convertidos = {
        chave: converter_valor_parametro_mlflow(valor)
        for chave, valor in parametros_execucao.items()
        if valor is not None
    }

    if parametros_convertidos:
        mlflow.log_params(parametros_convertidos)


def registrar_metricas_mlflow(
    metricas_avaliacao: dict[str, Any],
    metadados_preprocessamento: dict[str, Any],
    metadados_reducer: dict[str, Any],
    metadados_clusterizacao: dict[str, Any],
) -> None:
    """
    Registra métricas numéricas no MLflow.

    Parâmetros
    ----------
    metricas_avaliacao : dict[str, Any]
        Métricas de avaliação do experimento.
    metadados_preprocessamento : dict[str, Any]
        Metadados do preprocessamento.
    metadados_reducer : dict[str, Any]
        Metadados da redução de dimensionalidade.
    metadados_clusterizacao : dict[str, Any]
        Metadados da clusterização.
    """
    dicionarios = [
        metricas_avaliacao,
        metadados_preprocessamento,
        metadados_reducer,
        metadados_clusterizacao,
    ]

    metricas_numericas: dict[str, float] = {}

    for dicionario in dicionarios:
        for chave, valor in dicionario.items():
            if isinstance(valor, bool):
                continue

            if isinstance(valor, (int, float)) and pd.notna(valor):
                metricas_numericas[chave] = float(valor)

    if metricas_numericas:
        mlflow.log_metrics(metricas_numericas)


def salvar_artefatos_locais_execucao(
    resultado_execucao: dict[str, Any],
    pasta_execucao: Path,
) -> dict[str, Path]:
    """
    Salva artefatos locais da execução em disco.

    Artefatos salvos:
    - resultado por amostra em CSV;
    - embedding em CSV;
    - resumo da execução em JSON.

    Parâmetros
    ----------
    resultado_execucao : dict[str, Any]
        Resultado consolidado da execução.
    pasta_execucao : Path
        Pasta local da execução.

    Retorno
    -------
    dict[str, Path]
        Caminhos dos artefatos salvos.
    """
    pasta_execucao = garantir_pasta(pasta_execucao)

    caminhos_artefatos: dict[str, Path] = {}

    df_resultado_amostras = resultado_execucao["df_resultado_amostras"]
    caminho_resultado_amostras = pasta_execucao / "resultado_amostras.csv"
    df_resultado_amostras.to_csv(caminho_resultado_amostras, index=False, encoding="utf-8-sig")
    caminhos_artefatos["resultado_amostras_csv"] = caminho_resultado_amostras

    df_embedding = resultado_execucao["df_embedding"]
    caminho_embedding = pasta_execucao / "embedding.csv"
    df_embedding.to_csv(caminho_embedding, index=False, encoding="utf-8-sig")
    caminhos_artefatos["embedding_csv"] = caminho_embedding

    resumo_execucao = resultado_execucao["resumo_execucao"]
    caminho_resumo_json = pasta_execucao / "resumo_execucao.json"
    with caminho_resumo_json.open("w", encoding="utf-8") as arquivo_json:
        json.dump(resumo_execucao, arquivo_json, ensure_ascii=False, indent=2, default=str)
    caminhos_artefatos["resumo_execucao_json"] = caminho_resumo_json

    return caminhos_artefatos


def registrar_artefatos_mlflow(
    caminhos_artefatos: dict[str, Path],
) -> None:
    """
    Registra artefatos locais no MLflow.

    Parâmetros
    ----------
    caminhos_artefatos : dict[str, Path]
        Dicionário com caminhos dos artefatos salvos.
    """
    for _, caminho_artefato in caminhos_artefatos.items():
        if caminho_artefato.exists():
            mlflow.log_artifact(str(caminho_artefato))


def executar_fase1(
    caminho_config: str | Path,
) -> pd.DataFrame:
    """
    Executa todas as configurações unitárias da Fase 1.

    Parâmetros
    ----------
    caminho_config : str | Path
        Caminho do arquivo YAML da fase experimental.

    Retorno
    -------
    pd.DataFrame
        DataFrame consolidado com o resumo de todas as execuções.
    """
    configuracoes_unitarias, config_score = preparar_configuracoes_fase1(caminho_config)
    #configuracoes_unitarias = configuracoes_unitarias[:1]

    if not configuracoes_unitarias:
        raise ValueError("Nenhuma configuração unitária foi gerada para a Fase 1.")

    config_base = configuracoes_unitarias[0]
    config_projeto = config_base["projeto"]
    config_saidas = config_base["saidas"]

    preparar_experimento_mlflow(
        tracking_uri=config_projeto["tracking_uri"],
        nome_experimento=config_projeto["nome_experimento_mlflow"],
    )

    pasta_artefatos_base = garantir_pasta(config_saidas["pasta_artefatos"])

    resumos_execucoes: list[dict[str, Any]] = []

    print("=" * 80)
    print("INÍCIO DA EXECUÇÃO - FASE 1: PREPROCESSAMENTO")
    print("=" * 80)
    print(f"Arquivo de configuração: {caminho_config}")
    print(f"Total de execuções planejadas: {len(configuracoes_unitarias)}")
    print()

    for config_unitaria in configuracoes_unitarias:
        resumo_config = resumir_configuracao_unitaria(config_unitaria)
        indice_execucao = resumo_config["indice_execucao"]
        nome_execucao = resumo_config["nome_execucao"]

        print("-" * 80)
        print(f"Execução {indice_execucao:02d}")
        print(f"Nome: {nome_execucao}")
        print(f"Resumo: {resumo_config}")

        pasta_execucao = pasta_artefatos_base / nome_execucao

        with mlflow.start_run(run_name=nome_execucao):
            mlflow.set_tag("fase_experimental", config_unitaria["execucao"]["fase_experimental"])
            mlflow.set_tag("nome_execucao", nome_execucao)

            resultado_execucao = executar_pipeline_clones(
                config_unitaria=config_unitaria,
                config_score=config_score,
            )

            registrar_parametros_mlflow(
                resultado_execucao["parametros_execucao"]
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
            resumos_execucoes.append(resumo_execucao)

            score_final = resumo_execucao.get("score_final")
            print(f"Score final: {score_final}")
            print(f"Artefatos salvos em: {pasta_execucao}")
            print()

    df_resumo_execucoes = pd.DataFrame(resumos_execucoes)

    caminho_resumo_csv = pasta_artefatos_base / "resumo_fase1_preprocessamento.csv"
    df_resumo_execucoes.to_csv(caminho_resumo_csv, index=False, encoding="utf-8-sig")

    print("=" * 80)
    print("EXECUÇÃO FINALIZADA")
    print("=" * 80)
    print(f"Resumo consolidado salvo em: {caminho_resumo_csv}")
    print()

    return df_resumo_execucoes


def main() -> None:
    """
    Executa a rotina principal do script.
    """
    parser = criar_parser_argumentos()
    args = parser.parse_args()

    executar_fase1(caminho_config=args.config)


if __name__ == "__main__":
    main()