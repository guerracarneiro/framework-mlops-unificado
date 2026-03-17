from __future__ import annotations

"""
Módulo de configuração do Estudo de Caso 1.

- carregar arquivos YAML do experimento e do score final;
- validar a estrutura mínima esperada;
- expandir combinações da Fase 1 de preprocessamento;
- gerar configurações unitárias prontas para execução;
"""

from copy import deepcopy
from itertools import product
from pathlib import Path
from typing import Any

import yaml


def carregar_yaml(caminho_arquivo: str | Path) -> dict[str, Any]:
    """
    Carrega um arquivo YAML e retorna seu conteúdo como dicionário.

    Parâmetros
    ----------
    caminho_arquivo : str | Path
        Caminho do arquivo YAML.

    Retorno
    -------
    dict[str, Any]
        Conteúdo carregado do YAML.

    Exceções
    --------
    FileNotFoundError
        Quando o arquivo não existe.
    ValueError
        Quando o conteúdo não é um dicionário válido.
    """
    caminho_arquivo = Path(caminho_arquivo)

    if not caminho_arquivo.exists():
        raise FileNotFoundError(f"Arquivo YAML não encontrado: {caminho_arquivo}")

    with caminho_arquivo.open("r", encoding="utf-8") as arquivo:
        conteudo = yaml.safe_load(arquivo)

    if not isinstance(conteudo, dict):
        raise ValueError(f"O arquivo YAML deve conter um dicionário no nível raiz: {caminho_arquivo}")

    return conteudo


def validar_config_experimento(config: dict[str, Any]) -> None:
    """
    Valida se a configuração principal possui os blocos mínimos esperados.

    """
    chaves_obrigatorias = [
        "projeto",
        "dados",
        "preprocessamento",
        "modelagem",
        "criterios",
        "saidas",
    ]

    for chave in chaves_obrigatorias:
        if chave not in config:
            raise KeyError(f"Bloco obrigatório ausente na configuração: '{chave}'")

    # Validações mínimas de subchaves importantes
    if "nome_experimento_mlflow" not in config["projeto"]:
        raise KeyError("Chave obrigatória ausente: projeto.nome_experimento_mlflow")

    if "tracking_uri" not in config["projeto"]:
        raise KeyError("Chave obrigatória ausente: projeto.tracking_uri")

    if "caminho_excel" not in config["dados"]:
        raise KeyError("Chave obrigatória ausente: dados.caminho_excel")

    if "aba" not in config["dados"]:
        raise KeyError("Chave obrigatória ausente: dados.aba")

    if "grid_preprocessamento" not in config["preprocessamento"]:
        raise KeyError("Chave obrigatória ausente: preprocessamento.grid_preprocessamento")

    if "imputacao" not in config["preprocessamento"]["grid_preprocessamento"]:
        raise KeyError("Chave obrigatória ausente: preprocessamento.grid_preprocessamento.imputacao")

    if "onehot" not in config["preprocessamento"]["grid_preprocessamento"]:
        raise KeyError("Chave obrigatória ausente: preprocessamento.grid_preprocessamento.onehot")

    if "caminho_score_final" not in config["criterios"]:
        raise KeyError("Chave obrigatória ausente: criterios.caminho_score_final")


def validar_config_score(config_score: dict[str, Any]) -> None:
    """
    Valida a presença mínima dos campos do score final.

    """
    if not config_score:
        raise ValueError("A configuração de score final está vazia.")

    # A validação é intencionalmente leve nesta etapa.
    # Caso depois seja necessário, posso endurecer esta checagem.
    if not isinstance(config_score, dict):
        raise ValueError("A configuração de score final deve ser um dicionário.")


def carregar_config_experimento(caminho_config: str | Path) -> dict[str, Any]:
    """
    Carrega e valida a configuração principal do experimento.
    """
    config = carregar_yaml(caminho_config)
    validar_config_experimento(config)
    return config


def carregar_config_score(caminho_config_score: str | Path) -> dict[str, Any]:
    """
    Carrega e valida a configuração do score final.
    """
    config_score = carregar_yaml(caminho_config_score)
    validar_config_score(config_score)
    return config_score


def normalizar_nome_colunas_onehot(colunas: list[str]) -> str:
    """
    Gera um nome curto e padronizado para representar a estratégia de one-hot.

    Exemplos
    --------
    [] -> "sem_onehot"
    ["REGIAO"] -> "onehot_regiao"
    ["ESPECIE", "REGIAO"] -> "onehot_especie_regiao"
    """
    if not colunas:
        return "sem_onehot"

    nomes = [col.lower() for col in colunas]
    return "onehot_" + "_".join(nomes)


def gerar_nome_execucao(
    indice_execucao: int,
    config_unitaria: dict[str, Any],
) -> str:
    """
    Gera um nome curto e informativo para a execução.

    Esse nome será útil no MLflow e também nos arquivos de saída.
    """
    imputacao = config_unitaria["preprocessamento"]["imputacao"]["tipo"]
    colunas_onehot = config_unitaria["preprocessamento"]["onehot"]["colunas"]

    nome_onehot = normalizar_nome_colunas_onehot(colunas_onehot)

    return f"exec_{indice_execucao:02d}__imp_{imputacao}__{nome_onehot}"


def expandir_grid_preprocessamento(config_base: dict[str, Any]) -> list[dict[str, Any]]:
    """
    Expande a Fase 1 do estudo em configurações unitárias.

    A lógica desta fase considera a combinação entre:
    - estratégias de imputação
    - estratégias de one-hot encoding

    Cada combinação gera uma configuração pronta para execução.
    """
    config_base = deepcopy(config_base)

    grid_preprocessamento = config_base["preprocessamento"]["grid_preprocessamento"]
    lista_imputacao = grid_preprocessamento["imputacao"]
    lista_onehot = grid_preprocessamento["onehot"]

    configuracoes_unitarias: list[dict[str, Any]] = []

    combinacoes = list(product(lista_imputacao, lista_onehot))

    for indice, (cfg_imputacao, cfg_onehot) in enumerate(combinacoes, start=1):
        config_unitaria = deepcopy(config_base)

        # Removo o grid da configuração unitária, pois a partir daqui
        # cada execução deve conter apenas uma escolha concreta.
        config_unitaria["preprocessamento"].pop("grid_preprocessamento", None)

        # Registro a estratégia específica da execução.
        config_unitaria["preprocessamento"]["imputacao"] = deepcopy(cfg_imputacao)
        config_unitaria["preprocessamento"]["onehot"] = deepcopy(cfg_onehot)

        # Metadados da execução
        config_unitaria["execucao"] = {
            "indice_execucao": indice,
            "nome_execucao": "",
            "fase_experimental": "fase1_preprocessamento",
        }

        nome_execucao = gerar_nome_execucao(indice, config_unitaria)
        config_unitaria["execucao"]["nome_execucao"] = nome_execucao

        configuracoes_unitarias.append(config_unitaria)

    return configuracoes_unitarias


def preparar_configuracoes_fase1(caminho_config_experimento: str | Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """
    Carrega a configuração principal da Fase 1, carrega o score final e
    retorns a lista de configurações unitárias já expandidas.

    Retorno
    -------
    tuple[list[dict[str, Any]], dict[str, Any]]
        - lista de configurações unitárias da Fase 1
        - configuração do score final
    """
    config_experimento = carregar_config_experimento(caminho_config_experimento)

    caminho_score = config_experimento["criterios"]["caminho_score_final"]
    config_score = carregar_config_score(caminho_score)

    configuracoes_unitarias = expandir_grid_preprocessamento(config_experimento)

    return configuracoes_unitarias, config_score


def resumir_configuracao_unitaria(config_unitaria: dict[str, Any]) -> dict[str, Any]:
    """
    Gera um pequeno resumo da configuração unitária.

    Útil para logs, debug e conferência das execuções.
    """
    return {
        "indice_execucao": config_unitaria["execucao"]["indice_execucao"],
        "nome_execucao": config_unitaria["execucao"]["nome_execucao"],
        "imputacao": config_unitaria["preprocessamento"]["imputacao"]["tipo"],
        "onehot_colunas": config_unitaria["preprocessamento"]["onehot"]["colunas"],
        "reducer": config_unitaria["modelagem"]["reducer"]["name"],
        "clusterer": config_unitaria["modelagem"]["clusterer"]["name"],
    }