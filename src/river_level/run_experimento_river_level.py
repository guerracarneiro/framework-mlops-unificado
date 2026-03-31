from __future__ import annotations

import argparse
from pathlib import Path

import mlflow

from src.river_level.avaliacao import executar_avaliacao_baseline
from src.river_level.treino import executar_treinamento_baseline
from src.utils.io_utils import ler_yaml
from src.utils.mlflow_utils import (
    configurar_mlflow,
    registrar_artefato,
    registrar_metricas,
    registrar_parametros_config,
    registrar_tags_basicas,
    registrar_varios_artefatos,
)


def executar_experimento_river_level(caminho_config: str | Path) -> None:
    """
    Executa o experimento baseline do estudo de caso 2 com registro no MLflow.
    """
    caminho_config = Path(caminho_config)
    config = ler_yaml(caminho_config)
    config["_caminho_config"] = str(caminho_config)

    configurar_mlflow(
        nome_experimento=config["projeto"]["nome_experimento_mlflow"],
        tracking_uri=config["projeto"]["tracking_uri"],
    )

    with mlflow.start_run(run_name="baseline_lstm"):
        registrar_tags_basicas(config)
        registrar_parametros_config(config)
        registrar_artefato(caminho_config, pasta_destino="config")

        resumo_treinamento = executar_treinamento_baseline(config)
        resumo_avaliacao = executar_avaliacao_baseline(config)

        registrar_metricas(resumo_treinamento)
        registrar_metricas(resumo_avaliacao)

        caminhos_artefatos = [
            "models/river_level/modelo_lstm_baseline.keras",
            "artifacts/river_level/baseline_lstm/treinamento/historico_treinamento_baseline.csv",
            "artifacts/river_level/baseline_lstm/treinamento/resumo_treinamento_baseline.json",
            "artifacts/river_level/baseline_lstm/avaliacao/resumo_avaliacao_teste_baseline.json",
            "reports/river_level/predicoes_teste_baseline.csv",
        ]

        registrar_varios_artefatos(caminhos_artefatos, pasta_destino="artefatos")

        print("Treinamento baseline concluído com sucesso.")
        print(f"Tipo de modelo:                {resumo_treinamento['tipo_modelo']}")
        print(f"Shape X treino:                {resumo_treinamento['shape_X_treino']}")
        print(f"Shape y treino:                {resumo_treinamento['shape_y_treino']}")
        print(f"Shape X validação:             {resumo_treinamento['shape_X_validacao']}")
        print(f"Shape y validação:             {resumo_treinamento['shape_y_validacao']}")
        print(f"Shape X teste:                 {resumo_treinamento['shape_X_teste']}")
        print(f"Shape y teste:                 {resumo_treinamento['shape_y_teste']}")
        print(f"Épocas executadas:             {resumo_treinamento['epochs_executado']}")
        print(f"Melhor época:                  {resumo_treinamento['melhor_epoch']}")
        print(f"Melhor val_loss:               {resumo_treinamento['melhor_val_loss']}")
        print(f"Melhor val_mae:                {resumo_treinamento['melhor_val_mae']}")
        print(f"Modelo salvo em:               {resumo_treinamento['caminho_modelo_checkpoint']}")
        print(f"Log de treinamento salvo em:   {resumo_treinamento['caminho_log_treinamento']}")

        print("Avaliação no teste concluída com sucesso.")
        print(f"MAE teste:                     {resumo_avaliacao['mae_teste']}")
        print(f"RMSE teste:                    {resumo_avaliacao['rmse_teste']}")
        print(f"R² teste:                      {resumo_avaliacao['r2_teste']}")
        print(f"Viés médio teste:              {resumo_avaliacao['vies_medio_teste']}")
        print(f"NMAE% teste:                   {resumo_avaliacao['nmae_percentual_teste']}")
        print(f"Predições salvas em:           {resumo_avaliacao['caminho_predicoes_teste']}")
        print(f"Run MLflow:                    {mlflow.active_run().info.run_id}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Executa o treinamento baseline do estudo de caso Rio Doce."
    )
    parser.add_argument(
        "--config",
        required=True,
        help="Caminho do arquivo YAML de configuração.",
    )

    args = parser.parse_args()
    executar_experimento_river_level(args.config)


if __name__ == "__main__":
    main()