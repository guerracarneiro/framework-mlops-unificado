from __future__ import annotations

import argparse
import copy
from pathlib import Path

from src.river_level.pipeline_river_level import executar_pipeline_treino_avaliacao
from src.utils.io_utils import ler_yaml


def atualizar_dicionario_recursivo(base: dict, sobrescrita: dict) -> dict:
    """
    Atualiza recursivamente um dicionário base com os valores de sobrescrita.
    """
    for chave, valor in sobrescrita.items():
        if isinstance(valor, dict) and isinstance(base.get(chave), dict):
            atualizar_dicionario_recursivo(base[chave], valor)
        else:
            base[chave] = valor

    return base


def montar_config_execucao(
    config_base: dict,
    definicao_execucao: dict,
) -> dict:
    """
    Gera a configuração final de uma execução de tuning a partir da configuração base.
    """
    config_execucao = copy.deepcopy(config_base)

    nome_execucao = definicao_execucao["nome_execucao"]
    descricao = definicao_execucao.get("descricao")

    config_execucao["execucao"]["nome_execucao"] = nome_execucao
    config_execucao["execucao"]["familia_execucao"] = "tuning_modelo"
    config_execucao["execucao"]["baseline_referencia"] = "baseline_lstm"
    config_execucao["execucao"]["descricao"] = descricao

    if "modelo" in definicao_execucao:
        atualizar_dicionario_recursivo(
            config_execucao["modelo"],
            definicao_execucao["modelo"],
        )

    if "treinamento" in definicao_execucao:
        atualizar_dicionario_recursivo(
            config_execucao["treinamento"],
            definicao_execucao["treinamento"],
        )

    config_execucao["saidas"]["pasta_artefatos"] = (
        f"artifacts/river_level/tuning_modelo/{nome_execucao}"
    )
    config_execucao["saidas"]["pasta_modelos"] = (
        f"models/river_level/tuning_modelo/{nome_execucao}"
    )
    config_execucao["saidas"]["pasta_relatorios"] = (
        f"reports/river_level/tuning_modelo/{nome_execucao}"
    )

    # Mantém o experimento do MLflow da trilha de tuning.
    config_execucao["projeto"]["nome_experimento_mlflow"] = "river_level_tuning_modelo"

    return config_execucao


def executar_tuning_modelo(caminho_config: str | Path) -> None:
    """
    Executa uma sequência de experimentos manuais de tuning do Caso 2.
    """
    caminho_config = Path(caminho_config)
    config_tuning = ler_yaml(caminho_config)

    caminho_config_base = Path(config_tuning["config_base"])
    config_base = ler_yaml(caminho_config_base)

    lista_execucoes = config_tuning.get("execucoes", [])

    if not lista_execucoes:
        raise ValueError("Nenhuma execução foi definida no arquivo de tuning.")

    print("Iniciando tuning manual do Caso 2.")
    print(f"Arquivo de tuning:              {caminho_config}")
    print(f"Configuração base:             {caminho_config_base}")
    print(f"Quantidade de execuções:       {len(lista_execucoes)}")
    print("")

    for indice, definicao_execucao in enumerate(lista_execucoes, start=1):
        nome_execucao = definicao_execucao["nome_execucao"]
        print("=" * 90)
        print(f"Execução {indice}/{len(lista_execucoes)}: {nome_execucao}")

        config_execucao = montar_config_execucao(
            config_base=config_base,
            definicao_execucao=definicao_execucao,
        )

        resultado = executar_pipeline_treino_avaliacao(
            caminho_config=caminho_config_base,
            executar_treino=True,
            executar_avaliacao=True,
            registrar_mlflow=True,
            config_override=config_execucao,
        )

        resumo_treinamento = resultado["resumo_treinamento"]
        resumo_avaliacao = resultado["resumo_avaliacao"]
        run_id = resultado["run_id"]

        print(f"Run MLflow:                    {run_id}")
        print(f"Melhor val_loss:               {resumo_treinamento['melhor_val_loss']}")
        print(f"Melhor val_mae:                {resumo_treinamento['melhor_val_mae']}")
        print(f"MAE teste:                     {resumo_avaliacao['mae_teste']}")
        print(f"RMSE teste:                    {resumo_avaliacao['rmse_teste']}")
        print(f"R² teste:                      {resumo_avaliacao['r2_teste']}")
        print("")

    print("Tuning manual concluído com sucesso.")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Executa tuning manual de hiperparâmetros do estudo de caso Rio Doce."
    )
    parser.add_argument(
        "--config",
        required=True,
        help="Caminho do arquivo YAML de tuning manual.",
    )

    args = parser.parse_args()
    executar_tuning_modelo(args.config)


if __name__ == "__main__":
    main()