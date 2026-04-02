from __future__ import annotations

import argparse
from pathlib import Path

from src.river_level.pipeline_river_level import executar_pipeline_treino_avaliacao


def executar_treino_avaliacao_modelo(caminho_config: str | Path) -> None:
    """
    Runner genérico para treino e avaliação
    de modelos supervisionados do Caso 2.
    """
    resultado_pipeline = executar_pipeline_treino_avaliacao(
        caminho_config=caminho_config,
        executar_treino=True,
        executar_avaliacao=True,
    )

    resumo_execucao = resultado_pipeline["resumo_execucao"]
    resumo_treinamento = resultado_pipeline["resumo_treinamento"]
    resumo_avaliacao = resultado_pipeline["resumo_avaliacao"]

    print("Pipeline genérico executado com sucesso.")
    print(f"Configuração carregada:        {resumo_execucao['caminho_config']}")
    print(f"Família de execução:           {resumo_execucao['familia_execucao']}")
    print(f"Nome da execução:              {resumo_execucao['nome_execucao']}")
    print(f"Experimento MLflow:            {resumo_execucao['nome_experimento_mlflow']}")
    print(f"Tracking URI:                  {resumo_execucao['tracking_uri']}")
    print(f"Tipo de modelo:                {resumo_execucao['tipo_modelo']}")
    print(f"Dataset preparado:             {resumo_execucao['caminho_dataset_preparado']}")
    print(f"Scaler y:                      {resumo_execucao['caminho_scaler_y']}")
    print(f"Pasta de artefatos:            {resumo_execucao['pasta_artefatos']}")
    print(f"Pasta de modelos:              {resumo_execucao['pasta_modelos']}")
    print(f"Pasta de relatórios:           {resumo_execucao['pasta_relatorios']}")

    if resumo_treinamento is not None:
        print("")
        print("Resumo do treino:")
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

    if resumo_avaliacao is not None:
        print("")
        print("Resumo da avaliação:")
        print(f"MAE teste:                     {resumo_avaliacao['mae_teste']}")
        print(f"RMSE teste:                    {resumo_avaliacao['rmse_teste']}")
        print(f"R² teste:                      {resumo_avaliacao['r2_teste']}")
        print(f"Viés médio teste:              {resumo_avaliacao['vies_medio_teste']}")
        print(f"NMAE% teste:                   {resumo_avaliacao['nmae_percentual_teste']}")
        print(f"Predições salvas em:           {resumo_avaliacao['caminho_predicoes_teste']}")
        if "caminho_resumo_avaliacao" in resumo_avaliacao:
            print(f"Resumo da avaliação salvo em:  {resumo_avaliacao['caminho_resumo_avaliacao']}")
        else:
            print("Resumo da avaliação salvo em:  chave não retornada pelo módulo de avaliação")


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