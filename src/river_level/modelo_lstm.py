from __future__ import annotations

import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam


def criar_funcao_perda(config_modelo: dict):
    """
    Cria a função de perda configurada para o baseline.

    Estratégias suportadas:
    - mse: erro quadrático médio padrão
    - eqm_ponderado: MSE com maior peso para níveis baixos do rio
    """
    tipo_loss = str(config_modelo.get("tipo_loss", "mse")).strip().lower()

    if tipo_loss == "mse":
        return "mse"

    if tipo_loss == "eqm_ponderado":
        limiar = float(config_modelo["limiar_perda_ponderada"])
        peso = float(config_modelo["peso_perda_ponderada"])

        def perda_eqm_ponderado(y_true, y_pred):
            erro_quadratico = tf.square(y_true - y_pred)
            pesos = tf.where(y_true <= limiar, peso, 1.0)
            return tf.reduce_mean(erro_quadratico * pesos)

        perda_eqm_ponderado.__name__ = "eqm_ponderado"
        return perda_eqm_ponderado

    raise ValueError(
        f"Tipo de loss não suportado: {tipo_loss}. "
        f"Tipos aceitos: 'mse', 'eqm_ponderado'."
    )


def construir_modelo_lstm(
    passos_entrada: int,
    quantidade_features: int,
    config_modelo: dict,
) -> Model:
    """
    Constrói e compila o modelo LSTM baseline do estudo de caso 2.
    """
    unidades_lstm_1 = int(config_modelo["unidades_lstm_1"])
    unidades_lstm_2 = int(config_modelo["unidades_lstm_2"])
    dropout = float(config_modelo["dropout"])
    learning_rate = float(config_modelo["learning_rate"])
    funcao_perda = criar_funcao_perda(config_modelo)

    modelo = Sequential(
        [
            Input(shape=(passos_entrada, quantidade_features)),
            LSTM(unidades_lstm_1, return_sequences=True, name="lstm_1"),
            Dropout(dropout, name="dropout_1"),
            LSTM(unidades_lstm_2, return_sequences=False, name="lstm_2"),
            Dropout(dropout, name="dropout_2"),
            Dense(1, activation="linear", name="saida"),
        ],
        name="modelo_lstm_rio_doce",
    )

    modelo.compile(
        optimizer=Adam(learning_rate=learning_rate),
        loss=funcao_perda,
        metrics=["mae"],
    )

    return modelo