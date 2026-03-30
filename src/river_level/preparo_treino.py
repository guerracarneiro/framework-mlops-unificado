from __future__ import annotations

from pathlib import Path
import json

import joblib
import numpy as np
from sklearn.preprocessing import MinMaxScaler, StandardScaler


def carregar_dataset_sequencial(caminho_dataset: str | Path) -> dict:
    """
    Carrega o dataset sequencial gerado na Fase 3A.
    """
    caminho_dataset = Path(caminho_dataset)

    if not caminho_dataset.exists():
        raise FileNotFoundError(f"Dataset sequencial não encontrado: {caminho_dataset}")

    with np.load(caminho_dataset, allow_pickle=True) as dataset:
        dados = {
            "X": dataset["X"],
            "y": dataset["y"],
            "datas_inicio_janela": dataset["datas_inicio_janela"],
            "datas_fim_janela": dataset["datas_fim_janela"],
            "datas_alvo": dataset["datas_alvo"],
            "colunas_entrada": dataset["colunas_entrada"],
            "coluna_alvo": dataset["coluna_alvo"][0],
            "indice_alvo_nas_entradas": int(dataset["indice_alvo_nas_entradas"][0]),
            "passos_entrada": int(dataset["passos_entrada"][0]),
            "horizonte_previsao": int(dataset["horizonte_previsao"][0]),
        }

    return dados


def validar_dataset_sequencial(dados: dict) -> None:
    """
    Valida a consistência básica do dataset sequencial.
    """
    X = dados["X"]
    y = dados["y"]
    datas_alvo = dados["datas_alvo"]
    colunas_entrada = dados["colunas_entrada"]

    if X.ndim != 3:
        raise ValueError(f"X deve ter 3 dimensões. Shape recebido: {X.shape}")

    if y.ndim != 1:
        raise ValueError(f"y deve ter 1 dimensão. Shape recebido: {y.shape}")

    if X.shape[0] != y.shape[0]:
        raise ValueError("X e y possuem quantidades diferentes de amostras.")

    if X.shape[0] != len(datas_alvo):
        raise ValueError("Quantidade de datas alvo incompatível com o número de amostras.")

    if X.shape[2] != len(colunas_entrada):
        raise ValueError("Número de colunas de entrada incompatível com a última dimensão de X.")

    if np.isnan(X).any():
        raise ValueError("Foram encontrados valores NaN em X.")

    if np.isnan(y).any():
        raise ValueError("Foram encontrados valores NaN em y.")

    if np.isinf(X).any():
        raise ValueError("Foram encontrados valores infinitos em X.")

    if np.isinf(y).any():
        raise ValueError("Foram encontrados valores infinitos em y.")


def validar_proporcoes(
    proporcao_treino: float,
    proporcao_validacao: float,
    proporcao_teste: float,
) -> None:
    """
    Valida as proporções de divisão temporal.
    """
    proporcoes = [proporcao_treino, proporcao_validacao, proporcao_teste]

    if any(valor <= 0 for valor in proporcoes):
        raise ValueError("Todas as proporções devem ser maiores que zero.")

    soma = proporcao_treino + proporcao_validacao + proporcao_teste

    if not np.isclose(soma, 1.0, atol=1e-8):
        raise ValueError(
            f"As proporções devem somar 1.0. Soma atual: {soma}"
        )


def dividir_dataset_temporalmente(
    dados: dict,
    proporcao_treino: float,
    proporcao_validacao: float,
    proporcao_teste: float,
) -> dict:
    """
    Divide o dataset sequencial em treino, validação e teste, preservando a ordem temporal.
    """
    validar_proporcoes(
        proporcao_treino=proporcao_treino,
        proporcao_validacao=proporcao_validacao,
        proporcao_teste=proporcao_teste,
    )

    X = dados["X"]
    y = dados["y"]
    datas_alvo = dados["datas_alvo"]

    quantidade_amostras = X.shape[0]

    quantidade_treino = int(quantidade_amostras * proporcao_treino)
    quantidade_validacao = int(quantidade_amostras * proporcao_validacao)
    quantidade_teste = quantidade_amostras - quantidade_treino - quantidade_validacao

    if quantidade_treino <= 0 or quantidade_validacao <= 0 or quantidade_teste <= 0:
        raise ValueError(
            "A divisão temporal gerou uma partição vazia. Verifique as proporções."
        )

    indice_fim_treino = quantidade_treino
    indice_fim_validacao = quantidade_treino + quantidade_validacao

    particoes = {
        "X_treino": X[:indice_fim_treino],
        "y_treino": y[:indice_fim_treino],
        "datas_alvo_treino": datas_alvo[:indice_fim_treino],

        "X_validacao": X[indice_fim_treino:indice_fim_validacao],
        "y_validacao": y[indice_fim_treino:indice_fim_validacao],
        "datas_alvo_validacao": datas_alvo[indice_fim_treino:indice_fim_validacao],

        "X_teste": X[indice_fim_validacao:],
        "y_teste": y[indice_fim_validacao:],
        "datas_alvo_teste": datas_alvo[indice_fim_validacao:],
    }

    return particoes


def criar_scaler(tipo_scaler: str):
    """
    Cria o scaler configurado para a etapa de preparo do treino.
    """
    tipo_scaler = tipo_scaler.strip().lower()

    if tipo_scaler == "minmax":
        return MinMaxScaler()

    if tipo_scaler == "standard":
        return StandardScaler()

    raise ValueError(
        f"Tipo de scaler não suportado: {tipo_scaler}. "
        f"Tipos aceitos: 'minmax', 'standard'."
    )


def achatar_X_para_2d(X: np.ndarray) -> np.ndarray:
    """
    Converte X de 3D para 2D para ajuste e transformação com scaler.
    """
    quantidade_amostras, passos_entrada, quantidade_features = X.shape
    return X.reshape(quantidade_amostras * passos_entrada, quantidade_features)


def reconstruir_X_para_3d(
    X_2d: np.ndarray,
    quantidade_amostras: int,
    passos_entrada: int,
    quantidade_features: int,
) -> np.ndarray:
    """
    Reconstrói X para o formato 3D esperado pela LSTM.
    """
    return X_2d.reshape(quantidade_amostras, passos_entrada, quantidade_features)


def ajustar_scalers(
    X_treino: np.ndarray,
    y_treino: np.ndarray,
    tipo_scaler_X: str,
    tipo_scaler_y: str,
):
    """
    Ajusta os scalers somente com a partição de treino.
    """
    scaler_X = criar_scaler(tipo_scaler_X)
    scaler_y = criar_scaler(tipo_scaler_y)

    X_treino_2d = achatar_X_para_2d(X_treino)
    y_treino_2d = y_treino.reshape(-1, 1)

    scaler_X.fit(X_treino_2d)
    scaler_y.fit(y_treino_2d)

    return scaler_X, scaler_y


def transformar_X_com_scaler(X: np.ndarray, scaler_X) -> np.ndarray:
    """
    Aplica o scaler ao tensor X preservando o formato 3D.
    """
    quantidade_amostras, passos_entrada, quantidade_features = X.shape

    X_2d = achatar_X_para_2d(X)
    X_transformado_2d = scaler_X.transform(X_2d)

    return reconstruir_X_para_3d(
        X_2d=X_transformado_2d,
        quantidade_amostras=quantidade_amostras,
        passos_entrada=passos_entrada,
        quantidade_features=quantidade_features,
    )


def transformar_y_com_scaler(y: np.ndarray, scaler_y) -> np.ndarray:
    """
    Aplica o scaler ao vetor alvo.
    """
    return scaler_y.transform(y.reshape(-1, 1)).reshape(-1)


def transformar_particoes(particoes: dict, scaler_X, scaler_y) -> dict:
    """
    Aplica os scalers ajustados no treino às três partições.
    """
    return {
        "X_treino": transformar_X_com_scaler(particoes["X_treino"], scaler_X),
        "y_treino": transformar_y_com_scaler(particoes["y_treino"], scaler_y),
        "datas_alvo_treino": particoes["datas_alvo_treino"],

        "X_validacao": transformar_X_com_scaler(particoes["X_validacao"], scaler_X),
        "y_validacao": transformar_y_com_scaler(particoes["y_validacao"], scaler_y),
        "datas_alvo_validacao": particoes["datas_alvo_validacao"],

        "X_teste": transformar_X_com_scaler(particoes["X_teste"], scaler_X),
        "y_teste": transformar_y_com_scaler(particoes["y_teste"], scaler_y),
        "datas_alvo_teste": particoes["datas_alvo_teste"],
    }


def garantir_pasta_arquivo(caminho_arquivo: str | Path) -> None:
    """
    Garante que a pasta do arquivo exista antes da gravação.
    """
    Path(caminho_arquivo).parent.mkdir(parents=True, exist_ok=True)


def salvar_dataset_preparado(
    caminho_dataset_saida: str | Path,
    dados_transformados: dict,
    dados_originais: dict,
) -> None:
    """
    Salva o dataset final preparado para a etapa de treino.
    """
    caminho_dataset_saida = Path(caminho_dataset_saida)
    garantir_pasta_arquivo(caminho_dataset_saida)

    np.savez_compressed(
        caminho_dataset_saida,
        X_treino=dados_transformados["X_treino"],
        y_treino=dados_transformados["y_treino"],
        X_validacao=dados_transformados["X_validacao"],
        y_validacao=dados_transformados["y_validacao"],
        X_teste=dados_transformados["X_teste"],
        y_teste=dados_transformados["y_teste"],
        datas_alvo_treino=dados_transformados["datas_alvo_treino"],
        datas_alvo_validacao=dados_transformados["datas_alvo_validacao"],
        datas_alvo_teste=dados_transformados["datas_alvo_teste"],
        colunas_entrada=dados_originais["colunas_entrada"],
        coluna_alvo=np.array([dados_originais["coluna_alvo"]]),
        indice_alvo_nas_entradas=np.array([dados_originais["indice_alvo_nas_entradas"]]),
        passos_entrada=np.array([dados_originais["passos_entrada"]]),
        horizonte_previsao=np.array([dados_originais["horizonte_previsao"]]),
    )


def salvar_scalers(
    scaler_X,
    scaler_y,
    caminho_scaler_X: str | Path,
    caminho_scaler_y: str | Path,
) -> None:
    """
    Salva os scalers ajustados no treino.
    """
    garantir_pasta_arquivo(caminho_scaler_X)
    garantir_pasta_arquivo(caminho_scaler_y)

    joblib.dump(scaler_X, caminho_scaler_X)
    joblib.dump(scaler_y, caminho_scaler_y)


def converter_valor_json(valor):
    """
    Converte tipos NumPy para tipos serializáveis em JSON.
    """
    if isinstance(valor, (np.integer,)):
        return int(valor)

    if isinstance(valor, (np.floating,)):
        return float(valor)

    if isinstance(valor, (np.bool_,)):
        return bool(valor)

    if isinstance(valor, (np.datetime64,)):
        return str(valor)

    return valor


def montar_resumo_preparo_treino(
    dados_originais: dict,
    particoes: dict,
    dados_transformados: dict,
    config_preparo_treino: dict,
) -> dict:
    """
    Monta o resumo da execução da Fase 3B.
    """
    resumo = {
        "coluna_alvo": dados_originais["coluna_alvo"],
        "indice_alvo_nas_entradas": dados_originais["indice_alvo_nas_entradas"],
        "passos_entrada": dados_originais["passos_entrada"],
        "horizonte_previsao": dados_originais["horizonte_previsao"],
        "tipo_scaler_X": config_preparo_treino["tipo_scaler_X"],
        "tipo_scaler_y": config_preparo_treino["tipo_scaler_y"],
        "proporcao_treino": config_preparo_treino["proporcao_treino"],
        "proporcao_validacao": config_preparo_treino["proporcao_validacao"],
        "proporcao_teste": config_preparo_treino["proporcao_teste"],
        "shape_X_treino": list(dados_transformados["X_treino"].shape),
        "shape_y_treino": list(dados_transformados["y_treino"].shape),
        "shape_X_validacao": list(dados_transformados["X_validacao"].shape),
        "shape_y_validacao": list(dados_transformados["y_validacao"].shape),
        "shape_X_teste": list(dados_transformados["X_teste"].shape),
        "shape_y_teste": list(dados_transformados["y_teste"].shape),
        "primeira_data_treino": str(particoes["datas_alvo_treino"][0]),
        "ultima_data_treino": str(particoes["datas_alvo_treino"][-1]),
        "primeira_data_validacao": str(particoes["datas_alvo_validacao"][0]),
        "ultima_data_validacao": str(particoes["datas_alvo_validacao"][-1]),
        "primeira_data_teste": str(particoes["datas_alvo_teste"][0]),
        "ultima_data_teste": str(particoes["datas_alvo_teste"][-1]),
        "caminho_dataset_sequencial_entrada": config_preparo_treino["caminho_dataset_sequencial_entrada"],
        "caminho_dataset_saida": config_preparo_treino["caminho_dataset_saida"],
        "caminho_scaler_X": config_preparo_treino["caminho_scaler_X"],
        "caminho_scaler_y": config_preparo_treino["caminho_scaler_y"],
    }

    return {chave: converter_valor_json(valor) for chave, valor in resumo.items()}


def salvar_resumo_execucao(resumo: dict, caminho_resumo_execucao: str | Path) -> None:
    """
    Salva o resumo da execução em JSON.
    """
    garantir_pasta_arquivo(caminho_resumo_execucao)

    with open(caminho_resumo_execucao, "w", encoding="utf-8") as arquivo:
        json.dump(resumo, arquivo, ensure_ascii=False, indent=2)


def executar_preparo_treino(config_preparo_treino: dict) -> dict:
    """
    Executa a Fase 3B:
    divide o dataset sequencial,
    ajusta os scalers no treino,
    transforma as partições
    e salva os artefatos finais.
    """
    dados_originais = carregar_dataset_sequencial(
        config_preparo_treino["caminho_dataset_sequencial_entrada"]
    )

    validar_dataset_sequencial(dados_originais)

    particoes = dividir_dataset_temporalmente(
        dados=dados_originais,
        proporcao_treino=config_preparo_treino["proporcao_treino"],
        proporcao_validacao=config_preparo_treino["proporcao_validacao"],
        proporcao_teste=config_preparo_treino["proporcao_teste"],
    )

    scaler_X, scaler_y = ajustar_scalers(
        X_treino=particoes["X_treino"],
        y_treino=particoes["y_treino"],
        tipo_scaler_X=config_preparo_treino["tipo_scaler_X"],
        tipo_scaler_y=config_preparo_treino["tipo_scaler_y"],
    )

    dados_transformados = transformar_particoes(
        particoes=particoes,
        scaler_X=scaler_X,
        scaler_y=scaler_y,
    )

    salvar_dataset_preparado(
        caminho_dataset_saida=config_preparo_treino["caminho_dataset_saida"],
        dados_transformados=dados_transformados,
        dados_originais=dados_originais,
    )

    salvar_scalers(
        scaler_X=scaler_X,
        scaler_y=scaler_y,
        caminho_scaler_X=config_preparo_treino["caminho_scaler_X"],
        caminho_scaler_y=config_preparo_treino["caminho_scaler_y"],
    )

    resumo = montar_resumo_preparo_treino(
        dados_originais=dados_originais,
        particoes=particoes,
        dados_transformados=dados_transformados,
        config_preparo_treino=config_preparo_treino,
    )

    salvar_resumo_execucao(
        resumo=resumo,
        caminho_resumo_execucao=config_preparo_treino["caminho_resumo_execucao"],
    )

    return resumo