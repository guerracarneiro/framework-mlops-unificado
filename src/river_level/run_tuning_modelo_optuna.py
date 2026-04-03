from __future__ import annotations

import argparse
import copy
from optuna_integration import TFKerasPruningCallback
from pathlib import Path

import optuna

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


def sugerir_parametro(trial: optuna.Trial, nome: str, definicao: dict):
    """
    Traduz a definição do espaço de busca em uma sugestão do Optuna.
    """
    tipo = definicao["tipo"]

    if tipo == "int":
        return trial.suggest_int(
            name=nome,
            low=int(definicao["low"]),
            high=int(definicao["high"]),
            step=int(definicao.get("step", 1)),
        )

    if tipo == "float":
        return trial.suggest_float(
            name=nome,
            low=float(definicao["low"]),
            high=float(definicao["high"]),
            step=definicao.get("step"),
            log=bool(definicao.get("log", False)),
        )

    if tipo == "categorical":
        return trial.suggest_categorical(
            name=nome,
            choices=list(definicao["choices"]),
        )

    raise ValueError(f"Tipo de parâmetro não suportado: {tipo}")


def montar_config_execucao_optuna(
    config_base: dict,
    config_optuna: dict,
    trial: optuna.Trial,
) -> dict:
    """
    Monta a configuração final de uma trial do Optuna.
    """
    config_execucao = copy.deepcopy(config_base)

    parametros_sugeridos = {}
    espaco_busca = config_optuna["espaco_busca"]

    for nome_parametro, definicao in espaco_busca.items():
        parametros_sugeridos[nome_parametro] = sugerir_parametro(
            trial=trial,
            nome=nome_parametro,
            definicao=definicao,
        )

    nome_execucao = f"optuna_trial_{trial.number:03d}"

    config_execucao["execucao"]["nome_execucao"] = nome_execucao
    config_execucao["execucao"]["familia_execucao"] = config_optuna["execucao_base"]["familia_execucao"]
    config_execucao["execucao"]["baseline_referencia"] = config_optuna["execucao_base"]["baseline_referencia"]
    config_execucao["execucao"]["descricao"] = config_optuna["execucao_base"]["descricao"]

    # Ponto de partida vindo do melhor candidato consolidado manual:
    config_execucao["modelo"]["unidades_lstm_1"] = int(parametros_sugeridos["unidades_lstm_1"])
    config_execucao["modelo"]["unidades_lstm_2"] = int(parametros_sugeridos["unidades_lstm_2"])
    config_execucao["modelo"]["dropout"] = float(parametros_sugeridos["dropout"])
    config_execucao["modelo"]["learning_rate"] = float(parametros_sugeridos["learning_rate"])
    config_execucao["modelo"]["peso_perda_ponderada"] = float(parametros_sugeridos["peso_perda_ponderada"])

    config_execucao["treinamento"]["batch_size"] = int(parametros_sugeridos["batch_size"])

    parametros_fixos = config_optuna.get("parametros_fixos", {})

    if "modelo" in parametros_fixos:
        atualizar_dicionario_recursivo(
            config_execucao["modelo"],
            parametros_fixos["modelo"],
        )

    if "treinamento" in parametros_fixos:
        atualizar_dicionario_recursivo(
            config_execucao["treinamento"],
            parametros_fixos["treinamento"],
        )

    config_execucao["saidas"]["pasta_artefatos"] = f"artifacts/river_level/tuning_modelo/{nome_execucao}"
    config_execucao["saidas"]["pasta_modelos"] = f"models/river_level/tuning_modelo/{nome_execucao}"
    config_execucao["saidas"]["pasta_relatorios"] = f"reports/river_level/tuning_modelo/{nome_execucao}"

    config_execucao["projeto"]["nome_experimento_mlflow"] = "river_level_tuning_modelo"

    return config_execucao


def executar_tuning_optuna(caminho_config: str | Path) -> None:
    """
    Executa uma busca inicial com Optuna para o Caso 2.
    """
    caminho_config = Path(caminho_config)
    config_optuna = ler_yaml(caminho_config)

    caminho_config_base = Path(config_optuna["config_base"])
    config_base = ler_yaml(caminho_config_base)

    nome_estudo = config_optuna["estudo"]["nome_estudo"]
    direction = config_optuna["estudo"]["direction"]
    metrica_objetivo = config_optuna["estudo"]["metrica_objetivo"]
    n_trials = int(config_optuna["estudo"]["n_trials"])
    timeout_segundos = config_optuna["estudo"].get("timeout_segundos")

    print("Iniciando tuning com Optuna do Caso 2.")
    print(f"Arquivo de configuração:       {caminho_config}")
    print(f"Configuração base:             {caminho_config_base}")
    print(f"Nome do estudo:               {nome_estudo}")
    print(f"Métrica objetivo:             {metrica_objetivo}")
    print(f"Número de trials:             {n_trials}")
    print("")

    def objective(trial: optuna.Trial) -> float:
        config_execucao = montar_config_execucao_optuna(
            config_base=config_base,
            config_optuna=config_optuna,
            trial=trial,
        )

        config_execucao["optuna_trial"] = trial
        config_execucao["optuna_metrica_monitorada"] = "val_loss"

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

        trial.set_user_attr("run_id_mlflow", run_id)
        trial.set_user_attr("nome_execucao", config_execucao["execucao"]["nome_execucao"])
        trial.set_user_attr("mae_teste", resumo_avaliacao["mae_teste"])
        trial.set_user_attr("rmse_teste", resumo_avaliacao["rmse_teste"])
        trial.set_user_attr("r2_teste", resumo_avaliacao["r2_teste"])

        valor_objetivo = float(resumo_treinamento[metrica_objetivo])

        print("=" * 90)
        print(f"Trial {trial.number}")
        print(f"Run MLflow:                    {run_id}")
        print(f"Execução:                      {config_execucao['execucao']['nome_execucao']}")
        print(f"{metrica_objetivo}:            {valor_objetivo}")
        print(f"MAE teste:                     {resumo_avaliacao['mae_teste']}")
        print(f"RMSE teste:                    {resumo_avaliacao['rmse_teste']}")
        print(f"R² teste:                      {resumo_avaliacao['r2_teste']}")
        print("")

        return valor_objetivo

    storage = config_optuna["estudo"].get("storage")
    load_if_exists = bool(config_optuna["estudo"].get("load_if_exists", True))

    pruning_cfg = config_optuna.get("pruning", {})
    pruning_ativado = bool(pruning_cfg.get("ativado", False))

    if pruning_ativado:
        pruner = optuna.pruners.MedianPruner(
            n_startup_trials=int(pruning_cfg.get("n_startup_trials", 3)),
            n_warmup_steps=int(pruning_cfg.get("n_warmup_steps", 5)),
            interval_steps=int(pruning_cfg.get("interval_steps", 1)),
        )
    else:
        pruner = optuna.pruners.NopPruner()

    study = optuna.create_study(
        study_name=nome_estudo,
        direction=direction,
        storage=storage,
        load_if_exists=load_if_exists,
        pruner=pruner,
    )

    study.optimize(
        objective,
        n_trials=n_trials,
        timeout=timeout_segundos,
    )

    print("Busca Optuna concluída com sucesso.")
    print(f"Melhor trial:                  {study.best_trial.number}")
    print(f"Melhor valor objetivo:         {study.best_value}")
    print(f"Melhores parâmetros:           {study.best_trial.params}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Executa tuning controlado com Optuna para o estudo de caso Rio Doce."
    )
    parser.add_argument(
        "--config",
        required=True,
        help="Caminho do arquivo YAML de configuração do Optuna.",
    )

    args = parser.parse_args()
    executar_tuning_optuna(args.config)


if __name__ == "__main__":
    main()