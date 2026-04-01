from __future__ import annotations

import argparse
from pathlib import Path

from src.utils.io_utils import ler_yaml


def executar_treino_avaliacao_modelo(caminho_config: str | Path) -> None:
    """
    Runner genérico para futura execução de treino e avaliação
    de modelos supervisionados do Caso 2.
    Nesta primeira etapa, o objetivo é apenas preparar a estrutura.
    """
    caminho_config = Path(caminho_config)
    config = ler_yaml(caminho_config)

    print("Runner genérico criado com sucesso.")
    print(f"Configuração carregada: {caminho_config}")
    print(f"Família de execução: {config.get('execucao', {}).get('familia_execucao', 'não informada')}")
    print(f"Nome da execução: {config.get('execucao', {}).get('nome_execucao', 'não informado')}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Executa treino e avaliação genéricos do estudo de caso Rio Doce."
    )
    parser.add_argument(
        "--config",
        required=True,
        help="Caminho do arquivo YAML de configuração.",
    )

    args = parser.parse_args()
    executar_treino_avaliacao_modelo(args.config)


if __name__ == "__main__":
    main()