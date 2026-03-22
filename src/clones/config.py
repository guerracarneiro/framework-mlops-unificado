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
import hashlib
import json



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
    resumo = {
        "indice_execucao": config_unitaria["execucao"]["indice_execucao"],
        "nome_execucao": config_unitaria["execucao"]["nome_execucao"],
        "fase_experimental": config_unitaria["execucao"].get("fase_experimental"),
        "nome_base_candidata": config_unitaria["execucao"].get("nome_base_candidata"),
        "seed_execucao": config_unitaria["execucao"].get("seed_execucao"),
        "reducer": config_unitaria["modelagem"]["reducer"]["name"],
        "clusterer": config_unitaria["modelagem"]["clusterer"]["name"],
    }

    if "hash_configuracao" in config_unitaria["execucao"]:
        resumo["hash_configuracao"] = config_unitaria["execucao"]["hash_configuracao"]

    resumo.update({
        f"reducer__{chave}": valor
        for chave, valor in config_unitaria["modelagem"]["reducer"].get("params", {}).items()
    })

    resumo.update({
        f"clusterer__{chave}": valor
        for chave, valor in config_unitaria["modelagem"]["clusterer"].get("params", {}).items()
    })

    return resumo
def validar_config_experimento_fase2(config: dict[str, Any]) -> None:
    """
    Valida a estrutura mínima esperada para a Fase 2.

    A Fase 2 fixa o preprocessamento e expande apenas a modelagem.
    """
    chaves_obrigatorias = [
        "projeto",
        "dados",
        "preprocessamento",
        "modelagem",
        "criterios",
        "saidas",
        "execucao",
    ]

    for chave in chaves_obrigatorias:
        if chave not in config:
            raise KeyError(f"Bloco obrigatório ausente na configuração: '{chave}'")

    if "grid_modelagem" not in config["modelagem"]:
        raise KeyError("Chave obrigatória ausente: modelagem.grid_modelagem")

    if "reducer" not in config["modelagem"]["grid_modelagem"]:
        raise KeyError("Chave obrigatória ausente: modelagem.grid_modelagem.reducer")

    if "clusterer" not in config["modelagem"]["grid_modelagem"]:
        raise KeyError("Chave obrigatória ausente: modelagem.grid_modelagem.clusterer")

    if "imputacao" not in config["preprocessamento"]:
        raise KeyError("Chave obrigatória ausente: preprocessamento.imputacao")

    if "onehot" not in config["preprocessamento"]:
        raise KeyError("Chave obrigatória ausente: preprocessamento.onehot")


def gerar_hash_configuracao(config_unitaria: dict[str, Any]) -> str:
    """
    Gera um hash curto e determinístico da configuração unitária.
    """
    texto_config = json.dumps(config_unitaria, sort_keys=True, ensure_ascii=False, default=str)
    return hashlib.md5(texto_config.encode("utf-8")).hexdigest()[:12]


def gerar_nome_execucao_fase2(
    indice_execucao: int,
    config_unitaria: dict[str, Any],
) -> str:
    """
    Gera nome padronizado para runs da Fase 2.
    """
    imputacao = config_unitaria["preprocessamento"]["imputacao"]["tipo"]
    colunas_onehot = config_unitaria["preprocessamento"]["onehot"]["colunas"]
    nome_onehot = normalizar_nome_colunas_onehot(colunas_onehot)

    reducer_params = config_unitaria["modelagem"]["reducer"]["params"]
    clusterer_params = config_unitaria["modelagem"]["clusterer"]["params"]

    nome_execucao = (
        f"exec_{indice_execucao:03d}"
        f"__imp_{imputacao}"
        f"__{nome_onehot}"
        f"__umap_nn{reducer_params.get('n_neighbors')}"
        f"_nc{reducer_params.get('n_components')}"
        f"_md{str(reducer_params.get('min_dist')).replace('.', 'p')}"
        f"__hdb_mcs{clusterer_params.get('min_cluster_size')}"
        f"_ms{clusterer_params.get('min_samples')}"
        f"_eps{str(clusterer_params.get('cluster_selection_epsilon')).replace('.', 'p')}"
        f"_seed{reducer_params.get('random_state')}"
    )

    return nome_execucao


def expandir_grid_modelagem(config_base: dict[str, Any]) -> list[dict[str, Any]]:
    """
    Expande o grid de modelagem da Fase 2.

    Nesta fase:
    - preprocessamento fica fixo;
    - apenas os hiperparâmetros de UMAP + HDBSCAN variam.
    """
    config_base = deepcopy(config_base)

    grid_modelagem = config_base["modelagem"]["grid_modelagem"]

    reducer_cfg = grid_modelagem["reducer"]
    clusterer_cfg = grid_modelagem["clusterer"]

    reducer_name = reducer_cfg["name"]
    clusterer_name = clusterer_cfg["name"]

    reducer_params = reducer_cfg["params"]
    clusterer_params = clusterer_cfg["params"]

    lista_n_neighbors = reducer_params.get("n_neighbors", [15])
    lista_n_components = reducer_params.get("n_components", [10])
    lista_min_dist = reducer_params.get("min_dist", [0.05])
    lista_metric_reducer = reducer_params.get("metric", ["euclidean"])
    lista_random_state = reducer_params.get("random_state", [30])

    lista_min_cluster_size = clusterer_params.get("min_cluster_size", [5])
    lista_min_samples = clusterer_params.get("min_samples", [3])
    lista_cluster_selection_epsilon = clusterer_params.get("cluster_selection_epsilon", [1.0])
    lista_metric_clusterer = clusterer_params.get("metric", ["euclidean"])

    combinacoes = product(
        lista_n_neighbors,
        lista_n_components,
        lista_min_dist,
        lista_metric_reducer,
        lista_random_state,
        lista_min_cluster_size,
        lista_min_samples,
        lista_cluster_selection_epsilon,
        lista_metric_clusterer,
    )

    configuracoes_unitarias: list[dict[str, Any]] = []

    for indice, (
        n_neighbors,
        n_components,
        min_dist,
        metric_reducer,
        random_state,
        min_cluster_size,
        min_samples,
        cluster_selection_epsilon,
        metric_clusterer,
    ) in enumerate(combinacoes, start=1):

        config_unitaria = deepcopy(config_base)

        config_unitaria["modelagem"].pop("grid_modelagem", None)

        config_unitaria["modelagem"]["reducer"] = {
            "name": reducer_name,
            "params": {
                "n_neighbors": n_neighbors,
                "n_components": n_components,
                "min_dist": min_dist,
                "metric": metric_reducer,
                "random_state": random_state,
            },
        }

        config_unitaria["modelagem"]["clusterer"] = {
            "name": clusterer_name,
            "params": {
                "min_cluster_size": min_cluster_size,
                "min_samples": min_samples,
                "cluster_selection_epsilon": cluster_selection_epsilon,
                "metric": metric_clusterer,
            },
        }

        config_unitaria["execucao"]["indice_execucao"] = indice
        config_unitaria["execucao"]["fase_experimental"] = config_unitaria["execucao"].get(
            "fase_experimental",
            "fase2_tuning_robustez"
        )

        nome_execucao = gerar_nome_execucao_fase2(indice, config_unitaria)
        hash_configuracao = gerar_hash_configuracao(config_unitaria)

        config_unitaria["execucao"]["nome_execucao"] = nome_execucao
        config_unitaria["execucao"]["hash_configuracao"] = hash_configuracao

        configuracoes_unitarias.append(config_unitaria)

    return configuracoes_unitarias


def preparar_configuracoes_fase2(
    caminho_config_experimento: str | Path,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """
    Carrega a configuração principal da Fase 2 e expande o grid de modelagem.
    """
    config_experimento = carregar_yaml(caminho_config_experimento)
    validar_config_experimento_fase2(config_experimento)

    caminho_score = config_experimento["criterios"]["caminho_score_final"]
    config_score = carregar_config_score(caminho_score)

    configuracoes_unitarias = expandir_grid_modelagem(config_experimento)

    return configuracoes_unitarias, config_score

def validar_config_experimento_fase2b(config: dict[str, Any]) -> None:
    """
    Valida a estrutura mínima esperada para a Fase 2B.

    A Fase 2B fixa o preprocessamento e uma lista pequena de candidatas,
    expandindo cada candidata por múltiplas seeds.
    """
    chaves_obrigatorias = [
        "projeto",
        "dados",
        "preprocessamento",
        "robustez",
        "criterios",
        "saidas",
        "execucao",
    ]

    for chave in chaves_obrigatorias:
        if chave not in config:
            raise KeyError(f"Bloco obrigatório ausente na configuração: '{chave}'")

    if "seeds" not in config["robustez"]:
        raise KeyError("Chave obrigatória ausente: robustez.seeds")

    if "configuracoes_candidatas" not in config["robustez"]:
        raise KeyError("Chave obrigatória ausente: robustez.configuracoes_candidatas")

    if not config["robustez"]["configuracoes_candidatas"]:
        raise ValueError("A lista robustez.configuracoes_candidatas está vazia.")

    for indice, candidata in enumerate(config["robustez"]["configuracoes_candidatas"], start=1):
        if "nome_base" not in candidata:
            raise KeyError(f"Candidata {indice} sem chave 'nome_base'.")
        if "reducer" not in candidata:
            raise KeyError(f"Candidata {indice} sem chave 'reducer'.")
        if "clusterer" not in candidata:
            raise KeyError(f"Candidata {indice} sem chave 'clusterer'.")


def gerar_hash_configuracao_fase2b(config_unitaria: dict[str, Any]) -> str:
    """
    Gera um hash curto e determinístico da configuração unitária da Fase 2B.
    """
    texto_config = json.dumps(config_unitaria, sort_keys=True, ensure_ascii=False, default=str)
    return hashlib.md5(texto_config.encode("utf-8")).hexdigest()[:12]


def gerar_nome_execucao_fase2b(
    indice_execucao: int,
    config_unitaria: dict[str, Any],
) -> str:
    """
    Gera um nome padronizado para execuções da Fase 2B.
    """
    nome_base = config_unitaria["execucao"]["nome_base_candidata"]

    reducer_params = config_unitaria["modelagem"]["reducer"]["params"]
    clusterer_params = config_unitaria["modelagem"]["clusterer"]["params"]

    seed = reducer_params.get("random_state")

    nome_execucao = (
        f"exec_{indice_execucao:03d}"
        f"__{nome_base}"
        f"__umap_nn{reducer_params.get('n_neighbors')}"
        f"_nc{reducer_params.get('n_components')}"
        f"_md{str(reducer_params.get('min_dist')).replace('.', 'p')}"
        f"__hdb_mcs{clusterer_params.get('min_cluster_size')}"
        f"_ms{clusterer_params.get('min_samples')}"
        f"_eps{str(clusterer_params.get('cluster_selection_epsilon')).replace('.', 'p')}"
        f"_seed{seed}"
    )

    return nome_execucao


def expandir_configuracoes_fase2b(config_base: dict[str, Any]) -> list[dict[str, Any]]:
    """
    Expande a Fase 2B em configurações unitárias.

    Cada candidata é repetida para cada seed informada.
    """
    config_base = deepcopy(config_base)

    seeds = config_base["robustez"]["seeds"]
    candidatas = config_base["robustez"]["configuracoes_candidatas"]

    configuracoes_unitarias: list[dict[str, Any]] = []
    indice_execucao = 1

    for candidata in candidatas:
        nome_base = candidata["nome_base"]
        reducer = deepcopy(candidata["reducer"])
        clusterer = deepcopy(candidata["clusterer"])

        for seed in seeds:
            config_unitaria = deepcopy(config_base)

            config_unitaria.pop("robustez", None)

            config_unitaria["modelagem"] = {
                "reducer": deepcopy(reducer),
                "clusterer": deepcopy(clusterer),
            }

            config_unitaria["modelagem"]["reducer"].setdefault("params", {})
            config_unitaria["modelagem"]["reducer"]["params"]["random_state"] = seed

            config_unitaria["execucao"]["indice_execucao"] = indice_execucao
            config_unitaria["execucao"]["fase_experimental"] = config_unitaria["execucao"].get(
                "fase_experimental",
                "fase2b_robustez_multiseed",
            )
            config_unitaria["execucao"]["nome_base_candidata"] = nome_base
            config_unitaria["execucao"]["seed_execucao"] = seed

            nome_execucao = gerar_nome_execucao_fase2b(indice_execucao, config_unitaria)
            hash_configuracao = gerar_hash_configuracao_fase2b(config_unitaria)

            config_unitaria["execucao"]["nome_execucao"] = nome_execucao
            config_unitaria["execucao"]["hash_configuracao"] = hash_configuracao

            configuracoes_unitarias.append(config_unitaria)
            indice_execucao += 1

    return configuracoes_unitarias


def preparar_configuracoes_fase2b(
    caminho_config_experimento: str | Path,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """
    Carrega a configuração principal da Fase 2B e expande candidatas x seeds.
    """
    config_experimento = carregar_yaml(caminho_config_experimento)
    validar_config_experimento_fase2b(config_experimento)

    caminho_score = config_experimento["criterios"]["caminho_score_final"]
    config_score = carregar_config_score(caminho_score)

    configuracoes_unitarias = expandir_configuracoes_fase2b(config_experimento)

    return configuracoes_unitarias, config_score