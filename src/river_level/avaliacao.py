from __future__ import annotations

from pathlib import Path
import json

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from tensorflow.keras.models import load_model

from src.river_level.treino import (
    carregar_dataset_preparado,
    carregar_scaler_y,
    garantir_pasta_arquivo,
    obter_identificador_execucao,
)


def carregar_modelo_treinado(caminho_modelo: str | Path):
    """
    Carrega o modelo treinado para inferência.

    O carregamento é feito com compile=False porque nesta etapa
    só precisamos gerar predições, sem recompilar a loss customizada.
    """
    caminho_modelo = Path(caminho_modelo)

    if not caminho_modelo.exists():
        raise FileNotFoundError(f"Modelo treinado não encontrado: {caminho_modelo}")

    return load_model(caminho_modelo, compile=False)


def inverter_escala_alvo(valores_escalados: np.ndarray, scaler_y) -> np.ndarray:
    """
    Converte valores do alvo da escala normalizada para a escala original.
    """
    return scaler_y.inverse_transform(valores_escalados.reshape(-1, 1)).reshape(-1)


def calcular_metricas_regressao(y_real: np.ndarray, y_previsto: np.ndarray) -> dict:
    """
    Calcula métricas de regressão na escala original do nível do rio.
    """
    mae = float(mean_absolute_error(y_real, y_previsto))
    rmse = float(np.sqrt(mean_squared_error(y_real, y_previsto)))
    r2 = float(r2_score(y_real, y_previsto))
    vies_medio = float(np.mean(y_previsto - y_real))
    nmae_percentual = float((mae / np.mean(y_real)) * 100.0)

    return {
        "mae": mae,
        "rmse": rmse,
        "r2": r2,
        "vies_medio": vies_medio,
        "nmae_percentual": nmae_percentual,
    }


def montar_dataframe_predicoes(
    datas_alvo_teste: np.ndarray,
    y_real: np.ndarray,
    y_previsto: np.ndarray,
) -> pd.DataFrame:
    """
    Monta o dataframe de predições do conjunto de teste.
    """
    df_predicoes = pd.DataFrame(
        {
            "Data": pd.to_datetime(datas_alvo_teste),
            "Nivel_Real": y_real,
            "Nivel_Previsto": y_previsto,
        }
    )

    df_predicoes["Erro"] = df_predicoes["Nivel_Previsto"] - df_predicoes["Nivel_Real"]
    df_predicoes["Erro_Absoluto"] = df_predicoes["Erro"].abs()

    return df_predicoes.sort_values("Data").reset_index(drop=True)


def montar_caminhos_saida_avaliacao(config: dict) -> dict:
    """
    Define os caminhos principais de saída da avaliação.

    Mantém compatibilidade com a baseline oficial e gera nomes
    específicos para execuções genéricas.
    """
    pasta_modelos = Path(config["saidas"]["pasta_modelos"])
    pasta_artefatos = Path(config["saidas"]["pasta_artefatos"]) / "avaliacao"
    pasta_relatorios = Path(config["saidas"]["pasta_relatorios"])

    identificador_execucao = obter_identificador_execucao(config)

    if identificador_execucao == "baseline_lstm":
        nome_modelo = "modelo_lstm_baseline.keras"
        nome_resumo = "resumo_avaliacao_teste_baseline.json"
        nome_predicoes = "predicoes_teste_baseline.csv"
    else:
        nome_modelo = f"modelo_{identificador_execucao}.keras"
        nome_resumo = f"resumo_avaliacao_teste_{identificador_execucao}.json"
        nome_predicoes = f"predicoes_teste_{identificador_execucao}.csv"

    return {
        "caminho_modelo": str(pasta_modelos / nome_modelo),
        "caminho_resumo_avaliacao": str(pasta_artefatos / nome_resumo),
        "caminho_predicoes_teste": str(pasta_relatorios / nome_predicoes),
    }

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


def salvar_resumo_avaliacao(resumo: dict, caminho_resumo: str | Path) -> None:
    """
    Salva o resumo da avaliação em JSON.
    """
    garantir_pasta_arquivo(caminho_resumo)

    with open(caminho_resumo, "w", encoding="utf-8") as arquivo:
        json.dump(resumo, arquivo, ensure_ascii=False, indent=2)


def salvar_predicoes_teste(df_predicoes: pd.DataFrame, caminho_saida: str | Path) -> None:
    """
    Salva as predições do conjunto de teste em CSV.
    """
    garantir_pasta_arquivo(caminho_saida)
    df_predicoes.to_csv(caminho_saida, index=False, encoding="utf-8-sig")


def executar_avaliacao_baseline(config: dict) -> dict:
    """
    Executa a avaliação do modelo baseline no conjunto de teste.
    """
    dados = carregar_dataset_preparado(config["preparo_treino"]["caminho_dataset_saida"])
    scaler_y = carregar_scaler_y(config["preparo_treino"]["caminho_scaler_y"])
    caminhos_saida = montar_caminhos_saida_avaliacao(config)
    modelo = carregar_modelo_treinado(caminhos_saida["caminho_modelo"])

    y_previsto_escalado = modelo.predict(dados["X_teste"], verbose=0).reshape(-1)
    y_real_escalado = dados["y_teste"].reshape(-1)

    y_previsto = inverter_escala_alvo(y_previsto_escalado, scaler_y)
    y_real = inverter_escala_alvo(y_real_escalado, scaler_y)

    metricas = calcular_metricas_regressao(y_real=y_real, y_previsto=y_previsto)

    df_predicoes = montar_dataframe_predicoes(
        datas_alvo_teste=dados["datas_alvo_teste"],
        y_real=y_real,
        y_previsto=y_previsto,
    )

    salvar_predicoes_teste(
        df_predicoes=df_predicoes,
        caminho_saida=caminhos_saida["caminho_predicoes_teste"],
    )

    resumo = {
        "tipo_modelo": config["modelo"]["tipo"],
        "tipo_loss": config["modelo"].get("tipo_loss", "mse"),
        "limiar_perda_ponderada_original": config["modelo"].get("limiar_perda_ponderada"),
        "peso_perda_ponderada": config["modelo"].get("peso_perda_ponderada"),
        "shape_X_teste": list(dados["X_teste"].shape),
        "shape_y_teste": list(dados["y_teste"].shape),
        "primeira_data_teste": str(dados["datas_alvo_teste"][0]),
        "ultima_data_teste": str(dados["datas_alvo_teste"][-1]),
        "mae_teste": metricas["mae"],
        "rmse_teste": metricas["rmse"],
        "r2_teste": metricas["r2"],
        "vies_medio_teste": metricas["vies_medio"],
        "nmae_percentual_teste": metricas["nmae_percentual"],
        "caminho_modelo": caminhos_saida["caminho_modelo"],
        "caminho_predicoes_teste": caminhos_saida["caminho_predicoes_teste"],
        "caminho_resumo_avaliacao": caminhos_saida["caminho_resumo_avaliacao"],
    }

    resumo = {chave: converter_valor_json(valor) for chave, valor in resumo.items()}

    salvar_resumo_avaliacao(
        resumo=resumo,
        caminho_resumo=caminhos_saida["caminho_resumo_avaliacao"],
    )

    return resumo