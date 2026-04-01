from __future__ import annotations

import argparse
from pathlib import Path

from src.utils.io_utils import ler_yaml


def executar_tuning_modelo(caminho_config: str | Path) -> None:
    """
    Runner inicial para futura etapa de tuning de hiperparâmetros.
    Nesta primeira etapa, o objetivo é apenas validar a nova organização.
    """
    caminho_config = Path(caminho_config)
    config = ler_yaml(caminho_config)

    print("Runner de tuning criado com sucesso.")
    print(f"Configuração carregada: {caminho_config}")
    print(f"Experimento MLflow: {config['projeto']['nome_experimento_mlflow']}")
    print(f"Nome da execução: {config.get('execucao', {}).get('nome_execucao', 'não informado')}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Executa a estrutura inicial de tuning do estudo de caso Rio Doce."
    )
    parser.add_argument(
        "--config",
        required=True,
        help="Caminho do arquivo YAML de configuração.",
    )

    args = parser.parse_args()
    executar_tuning_modelo(args.config)


if __name__ == "__main__":
    main()