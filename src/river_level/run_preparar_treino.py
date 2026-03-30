from __future__ import annotations

import argparse
from pathlib import Path

from src.river_level.preparo_treino import executar_preparo_treino
from src.utils.io_utils import ler_yaml


def executar_fase_preparo_treino(caminho_config: str | Path) -> None:
    """
    Executa a Fase 3B do Caso 2:
    divisão temporal + normalização para treino da LSTM.
    """
    caminho_config = Path(caminho_config)
    config = ler_yaml(caminho_config)
    config["_caminho_config"] = str(caminho_config)

    preparo_treino_cfg = config["preparo_treino"]

    resumo_execucao = executar_preparo_treino(preparo_treino_cfg)

    print("Preparo de treino concluído com sucesso.")
    print(
        f"Dataset sequencial de entrada: "
        f"{preparo_treino_cfg['caminho_dataset_sequencial_entrada']}"
    )
    print(
        f"Dataset final de saída:       "
        f"{preparo_treino_cfg['caminho_dataset_saida']}"
    )
    print(
        f"Scaler X:                     "
        f"{preparo_treino_cfg['caminho_scaler_X']}"
    )
    print(
        f"Scaler y:                     "
        f"{preparo_treino_cfg['caminho_scaler_y']}"
    )
    print(
        f"Resumo JSON:                  "
        f"{preparo_treino_cfg['caminho_resumo_execucao']}"
    )
    print(f"Shape X treino:               {resumo_execucao['shape_X_treino']}")
    print(f"Shape y treino:               {resumo_execucao['shape_y_treino']}")
    print(
        f"Shape X validação:            "
        f"{resumo_execucao['shape_X_validacao']}"
    )
    print(
        f"Shape y validação:            "
        f"{resumo_execucao['shape_y_validacao']}"
    )
    print(f"Shape X teste:                {resumo_execucao['shape_X_teste']}")
    print(f"Shape y teste:                {resumo_execucao['shape_y_teste']}")
    print(
        f"Período treino:               "
        f"{resumo_execucao['primeira_data_treino']} "
        f"até {resumo_execucao['ultima_data_treino']}"
    )
    print(
        f"Período validação:            "
        f"{resumo_execucao['primeira_data_validacao']} "
        f"até {resumo_execucao['ultima_data_validacao']}"
    )
    print(
        f"Período teste:                "
        f"{resumo_execucao['primeira_data_teste']} "
        f"até {resumo_execucao['ultima_data_teste']}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Executa o preparo de treino do estudo de caso Rio Doce."
    )
    parser.add_argument(
        "--config",
        required=True,
        help="Caminho do arquivo YAML de configuração.",
    )

    args = parser.parse_args()
    executar_fase_preparo_treino(args.config)


if __name__ == "__main__":
    main()