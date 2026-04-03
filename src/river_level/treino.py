from __future__ import annotations

from pathlib import Path
import json
import joblib

import numpy as np
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, CSVLogger
from optuna_integration import TFKerasPruningCallback

import tensorflow as tf

from src.river_level.modelo_lstm import construir_modelo_lstm

def preparar_config_modelo_para_treino(config: dict, scaler_y) -> dict:
    """
    Prepara a configuração efetiva do modelo para o treino.

    Quando a loss ponderada é usada, o limiar informado na escala original
    do nível do rio é convertido para a escala normalizada do alvo.
    """
    config_modelo_treino = dict(config["modelo"])
    tipo_loss = str(config_modelo_treino.get("tipo_loss", "mse")).strip().lower()

    if tipo_loss == "eqm_ponderado":
        limiar_original = float(config_modelo_treino["limiar_perda_ponderada"])
        limiar_escalado = converter_limiar_original_para_escalado(
            limiar_original=limiar_original,
            scaler_y=scaler_y,
        )

        config_modelo_treino["limiar_perda_ponderada_original"] = limiar_original
        config_modelo_treino["limiar_perda_ponderada"] = limiar_escalado

    return config_modelo_treino

def carregar_scaler_y(caminho_scaler_y: str | Path):
    """
    Carrega o scaler do alvo gerado na Fase 3B.
    """
    caminho_scaler_y = Path(caminho_scaler_y)

    if not caminho_scaler_y.exists():
        raise FileNotFoundError(f"Scaler y não encontrado: {caminho_scaler_y}")

    return joblib.load(caminho_scaler_y)

def converter_limiar_original_para_escalado(limiar_original: float, scaler_y) -> float:
    """
    Converte o limiar da escala original do nível do rio para a escala usada no treino.
    """
    valor_escalado = scaler_y.transform(np.array([[limiar_original]], dtype=float))[0, 0]
    return float(valor_escalado)

def configurar_reprodutibilidade(config_projeto: dict) -> None:
    """
    Aplica a semente global configurada para aumentar a reprodutibilidade do treino.
    """
    semente_global = int(config_projeto["semente_global"])
    tf.keras.utils.set_random_seed(semente_global)

def garantir_pasta_arquivo(caminho_arquivo: str | Path) -> None:
    """
    Garante que a pasta do arquivo exista antes da gravação.
    """
    Path(caminho_arquivo).parent.mkdir(parents=True, exist_ok=True)


def carregar_dataset_preparado(caminho_dataset: str | Path) -> dict:
    """
    Carrega o dataset preparado na Fase 3B.
    """
    caminho_dataset = Path(caminho_dataset)

    if not caminho_dataset.exists():
        raise FileNotFoundError(f"Dataset preparado não encontrado: {caminho_dataset}")

    with np.load(caminho_dataset, allow_pickle=True) as dataset:
        dados = {
            "X_treino": dataset["X_treino"],
            "y_treino": dataset["y_treino"],
            "X_validacao": dataset["X_validacao"],
            "y_validacao": dataset["y_validacao"],
            "X_teste": dataset["X_teste"],
            "y_teste": dataset["y_teste"],
            "datas_alvo_treino": dataset["datas_alvo_treino"],
            "datas_alvo_validacao": dataset["datas_alvo_validacao"],
            "datas_alvo_teste": dataset["datas_alvo_teste"],
            "colunas_entrada": dataset["colunas_entrada"],
            "coluna_alvo": dataset["coluna_alvo"][0],
            "indice_alvo_nas_entradas": int(dataset["indice_alvo_nas_entradas"][0]),
            "passos_entrada": int(dataset["passos_entrada"][0]),
            "horizonte_previsao": int(dataset["horizonte_previsao"][0]),
        }

    return dados


def validar_dataset_preparado(dados: dict) -> None:
    """
    Valida a consistência básica do dataset preparado para treino.
    """
    for nome in [
        "X_treino",
        "y_treino",
        "X_validacao",
        "y_validacao",
        "X_teste",
        "y_teste",
    ]:
        if np.isnan(dados[nome]).any():
            raise ValueError(f"Foram encontrados NaN em {nome}.")

        if np.isinf(dados[nome]).any():
            raise ValueError(f"Foram encontrados infinitos em {nome}.")

    if dados["X_treino"].ndim != 3:
        raise ValueError(f"X_treino deve ser 3D. Shape: {dados['X_treino'].shape}")

    if dados["X_validacao"].ndim != 3:
        raise ValueError(f"X_validacao deve ser 3D. Shape: {dados['X_validacao'].shape}")

    if dados["X_teste"].ndim != 3:
        raise ValueError(f"X_teste deve ser 3D. Shape: {dados['X_teste'].shape}")

    if dados["y_treino"].ndim != 1:
        raise ValueError(f"y_treino deve ser 1D. Shape: {dados['y_treino'].shape}")

    if dados["y_validacao"].ndim != 1:
        raise ValueError(f"y_validacao deve ser 1D. Shape: {dados['y_validacao'].shape}")

    if dados["y_teste"].ndim != 1:
        raise ValueError(f"y_teste deve ser 1D. Shape: {dados['y_teste'].shape}")


def criar_callbacks_treinamento(config_treinamento: dict, caminhos_saida: dict) -> list:
    """
    Cria os callbacks básicos de treino da baseline.
    """
    paciencia = int(config_treinamento["patience_early_stopping"])

    caminho_modelo_checkpoint = caminhos_saida["caminho_modelo_checkpoint"]
    caminho_log_treinamento = caminhos_saida["caminho_log_treinamento"]

    garantir_pasta_arquivo(caminho_modelo_checkpoint)
    garantir_pasta_arquivo(caminho_log_treinamento)

    callbacks = [
        EarlyStopping(
            monitor="val_loss",
            patience=paciencia,
            restore_best_weights=True,
            verbose=1,
        ),
        ModelCheckpoint(
            filepath=caminho_modelo_checkpoint,
            monitor="val_loss",
            save_best_only=True,
            save_weights_only=False,
            verbose=1,
        ),
        CSVLogger(
            filename=caminho_log_treinamento,
            separator=",",
            append=False,
        ),
    ]

    trial_optuna = config.get("optuna_trial")
    metrica_monitorada_optuna = config.get("optuna_metrica_monitorada", "val_loss")

    if trial_optuna is not None:
        callbacks.append(
            TFKerasPruningCallback(
                trial_optuna,
                metrica_monitorada_optuna,
            )
        )

    return callbacks

def obter_identificador_execucao(config: dict) -> str:
    """
    Obtém um identificador textual para nomear artefatos da execução.

    Regras:
    - se existir bloco 'execucao' com 'nome_execucao', usa esse valor;
    - caso contrário, mantém 'baseline_lstm' para preservar compatibilidade
      com a baseline oficial já consolidada.
    """
    execucao_cfg = config.get("execucao", {})
    identificador = execucao_cfg.get("nome_execucao", "baseline_lstm")
    identificador = str(identificador).strip().replace(" ", "_")
    return identificador

def montar_caminhos_saida_treinamento(config: dict) -> dict:
    """
    Define os caminhos principais de saída do treino.

    Mantém compatibilidade com a baseline oficial e gera nomes
    específicos para execuções genéricas.
    """
    pasta_modelos = Path(config["saidas"]["pasta_modelos"])
    pasta_artefatos = Path(config["saidas"]["pasta_artefatos"]) / "treinamento"

    identificador_execucao = obter_identificador_execucao(config)

    if identificador_execucao == "baseline_lstm":
        nome_modelo = "modelo_lstm_baseline.keras"
        nome_historico = "historico_treinamento_baseline.csv"
        nome_resumo = "resumo_treinamento_baseline.json"
    else:
        nome_modelo = f"modelo_{identificador_execucao}.keras"
        nome_historico = f"historico_treinamento_{identificador_execucao}.csv"
        nome_resumo = f"resumo_treinamento_{identificador_execucao}.json"

    return {
        "caminho_modelo_checkpoint": str(pasta_modelos / nome_modelo),
        "caminho_log_treinamento": str(pasta_artefatos / nome_historico),
        "caminho_resumo_treinamento": str(pasta_artefatos / nome_resumo),
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


def montar_resumo_treinamento(
    dados: dict,
    config: dict,
    config_modelo_treino: dict,
    historico,
    caminhos_saida: dict,
) -> dict:
    """
    Monta o resumo da execução do treino baseline.
    """
    historico_dict = historico.history

    melhor_epoch = int(np.argmin(historico_dict["val_loss"])) + 1
    melhor_val_loss = float(np.min(historico_dict["val_loss"]))
    melhor_val_mae = float(historico_dict["val_mae"][melhor_epoch - 1])

    resumo = {
        "tipo_modelo": config["modelo"]["tipo"],
        "unidades_lstm_1": config["modelo"]["unidades_lstm_1"],
        "unidades_lstm_2": config["modelo"]["unidades_lstm_2"],
        "dropout": config["modelo"]["dropout"],
        "learning_rate": config["modelo"]["learning_rate"],

        "tipo_loss": config_modelo_treino.get("tipo_loss", "mse"),
        "limiar_perda_ponderada": config_modelo_treino.get("limiar_perda_ponderada"),
        "limiar_perda_ponderada_original": config_modelo_treino.get("limiar_perda_ponderada_original"),
        "peso_perda_ponderada": config_modelo_treino.get("peso_perda_ponderada"),

        "epochs_configurado": config["treinamento"]["epochs"],
        "batch_size": config["treinamento"]["batch_size"],
        "patience_early_stopping": config["treinamento"]["patience_early_stopping"],
        "shape_X_treino": list(dados["X_treino"].shape),
        "shape_y_treino": list(dados["y_treino"].shape),
        "shape_X_validacao": list(dados["X_validacao"].shape),
        "shape_y_validacao": list(dados["y_validacao"].shape),
        "shape_X_teste": list(dados["X_teste"].shape),
        "shape_y_teste": list(dados["y_teste"].shape),
        "primeira_data_treino": str(dados["datas_alvo_treino"][0]),
        "ultima_data_treino": str(dados["datas_alvo_treino"][-1]),
        "primeira_data_validacao": str(dados["datas_alvo_validacao"][0]),
        "ultima_data_validacao": str(dados["datas_alvo_validacao"][-1]),
        "primeira_data_teste": str(dados["datas_alvo_teste"][0]),
        "ultima_data_teste": str(dados["datas_alvo_teste"][-1]),
        "epochs_executado": len(historico_dict["loss"]),
        "melhor_epoch": melhor_epoch,
        "loss_final_treino": float(historico_dict["loss"][-1]),
        "mae_final_treino": float(historico_dict["mae"][-1]),
        "loss_final_validacao": float(historico_dict["val_loss"][-1]),
        "mae_final_validacao": float(historico_dict["val_mae"][-1]),
        "melhor_val_loss": melhor_val_loss,
        "melhor_val_mae": melhor_val_mae,
        "caminho_modelo_checkpoint": caminhos_saida["caminho_modelo_checkpoint"],
        "caminho_log_treinamento": caminhos_saida["caminho_log_treinamento"],
    }

    return {chave: converter_valor_json(valor) for chave, valor in resumo.items()}


def salvar_resumo_treinamento(resumo: dict, caminho_resumo: str | Path) -> None:
    """
    Salva o resumo do treino em JSON.
    """
    garantir_pasta_arquivo(caminho_resumo)

    with open(caminho_resumo, "w", encoding="utf-8") as arquivo:
        json.dump(resumo, arquivo, ensure_ascii=False, indent=2)


def executar_treinamento_baseline(config: dict) -> dict:
    """
    Executa o treino baseline da LSTM para o estudo de caso 2.
    """
    configurar_reprodutibilidade(config["projeto"])
    config_preparo_treino = config["preparo_treino"]
    caminho_dataset_preparado = config_preparo_treino["caminho_dataset_saida"]

    dados = carregar_dataset_preparado(caminho_dataset_preparado)
    validar_dataset_preparado(dados)

    passos_entrada = dados["X_treino"].shape[1]
    quantidade_features = dados["X_treino"].shape[2]

    scaler_y = carregar_scaler_y(config["preparo_treino"]["caminho_scaler_y"])

    config_modelo_treino = preparar_config_modelo_para_treino(
        config=config,
        scaler_y=scaler_y,
    )

    modelo = construir_modelo_lstm(
        passos_entrada=passos_entrada,
        quantidade_features=quantidade_features,
        config_modelo=config_modelo_treino,
    )

    caminhos_saida = montar_caminhos_saida_treinamento(config)

    callbacks = criar_callbacks_treinamento(
        config_treinamento=config["treinamento"],
        caminhos_saida=caminhos_saida,
    )

    historico = modelo.fit(
        dados["X_treino"],
        dados["y_treino"],
        validation_data=(dados["X_validacao"], dados["y_validacao"]),
        epochs=int(config["treinamento"]["epochs"]),
        batch_size=int(config["treinamento"]["batch_size"]),
        verbose=1,
        callbacks=callbacks,
        shuffle=False,
    )

    resumo = montar_resumo_treinamento(
        dados=dados,
        config=config,
        config_modelo_treino=config_modelo_treino,
        historico=historico,
        caminhos_saida=caminhos_saida,
    )

    salvar_resumo_treinamento(
        resumo=resumo,
        caminho_resumo=caminhos_saida["caminho_resumo_treinamento"],
    )

    return resumo