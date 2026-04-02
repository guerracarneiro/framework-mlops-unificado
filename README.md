# Framework MLOps Unificado

Projeto de dissertação de mestrado focado na aplicação de práticas de MLOps
em pipelines de aprendizado de máquina para problemas industriais
no setor de celulose e papel.

## Estudos de Caso

1. Segmentação de clones de eucalipto (UMAP + HDBSCAN)
2. Previsão do nível do Rio Doce
3. Reconstrução de dados meteorológicos

## Estrutura do Projeto

data/         → dados brutos e processados  
notebooks/    → exploração  
src/          → código fonte  
experiments/  → configurações de experimentos  
reports/      → resultados e gráficos  
models/       → modelos treinados  

## Ferramentas

Python  
MLflow  
Git / GitHub  
DVC (adoção gradual)

==========================================================================================================
## ESTUDO DE CASO 1
==========================================================================================================


## Iniciar Ambiente

1. Terminal > New Terminal (abrir novo terminal)
2. Saida de ambientes VENV (comando: deactivate)
3. conda activate mlops_clones (entrar no Conda)
4. Entrar na pasta do projeto (cd FrameWork_MLOps_Unificado)
5. git status (verificar se git ativo)
6. mlflow server --backend-store-uri sqlite:///mlflow.db --host 127.0.0.1 --port 5000 --workers 1 (ativar MLFLow)
7. ### Gerar Zip ###
        $arquivoZip = "fase1_codigo_config.zip"
        $pastaTemp = "_tmp_zip_fase1"

        if (Test-Path $pastaTemp) { Remove-Item $pastaTemp -Recurse -Force }
        if (Test-Path $arquivoZip) { Remove-Item $arquivoZip -Force }

        New-Item -ItemType Directory -Path $pastaTemp | Out-Null

        $itensFixos = @(
            "README.md",
            "requirements.txt",
            "pyproject.toml",
            "params.yaml",
            "params.yml",
            "dvc.yaml",
            "dvc.lock",
            ".dvcignore",
            "data\raw\clones\DadosCaracterizacao.xlsx.dvc"
        )

        foreach ($item in $itensFixos) {
            if (Test-Path $item) {
                $destino = Join-Path $pastaTemp $item
                $pastaDestino = Split-Path $destino -Parent
                if (-not (Test-Path $pastaDestino)) {
                    New-Item -ItemType Directory -Path $pastaDestino -Force | Out-Null
                }
                Copy-Item $item $destino -Force
            }
        }

        $pastasCodigo = @(
            "src\clones",
            "src\utils",
            "experiments\clones"
        )

        foreach ($pasta in $pastasCodigo) {
            if (Test-Path $pasta) {
                Get-ChildItem $pasta -Recurse -File | Where-Object {
                    $_.Extension -in @(".py", ".yaml", ".yml", ".dvc", ".md", ".txt")
                } | Where-Object {
                    $_.FullName -notmatch "\\__pycache__\\"
                } | ForEach-Object {
                    $relativo = Resolve-Path -Relative $_.FullName
                    $relativo = $relativo -replace "^\.\\" , ""
                    $destino = Join-Path $pastaTemp $relativo
                    $pastaDestino = Split-Path $destino -Parent
                    if (-not (Test-Path $pastaDestino)) {
                        New-Item -ItemType Directory -Path $pastaDestino -Force | Out-Null
                    }
                    Copy-Item $_.FullName $destino -Force
                }
            }
        }

        Compress-Archive -Path "$pastaTemp\*" -DestinationPath $arquivoZip -Force
        Remove-Item $pastaTemp -Recurse -Force

        Write-Host ""
        Write-Host "ZIP gerado com sucesso:" (Resolve-Path $arquivoZip)


## Rodar o Experimento Fase 1

python -m src.clones.run_experimento_clones --config experiments/clones/fase1_preprocessamento.yaml

## Rodar Experimento Fase 2

python -m src.clones.run_fase2_tuning_robustez --config experiments/clones/fase2_tuning_robustez.yaml

## Seleciona Candidatos para testes de robustez multi-seed

python -c "import pandas as pd; df=pd.read_csv('artifacts/clones/fase2_tuning_robustez/resumo_fase2_tuning_robustez.csv'); df5=df[df['n_clusters']==5].sort_values(['noise_pct','score_final'], ascending=[True,False]).head(2); df6=df[df['n_clusters']==6].sort_values(['noise_pct','score_final'], ascending=[True,False]).head(2); out=pd.concat([df5,df6], ignore_index=True); out.to_csv('artifacts/clones/fase2_tuning_robustez/candidatas_fase2b.csv', index=False, encoding='utf-8-sig'); print(out[['nome_execucao','score_final','noise_pct','n_clusters','reducer__n_neighbors','reducer__min_dist','clusterer__min_cluster_size','clusterer__min_samples','clusterer__cluster_selection_epsilon']].to_string(index=False))"

## Rodar Experimento Fase 2B

python -m src.clones.run_fase2b_robustez_multiseed --config experiments/clones/fase2b_robustez_multiseed.yaml

## Rodar Experimento Final

python -m src.clones.run_fase2_tuning_robustez --config experiments/clones/fase2c_config_final_artigo.yaml

## Rodar Caracterização

python -m src.clones.run_fase2c_caracterizacao_clusters

## Rorar Finalização Fase 2C

python -m src.clones.run_fase2_tuning_robustez --config experiments/clones/fase2c_config_final_artigo.yaml

## Rodar Fase 2D

python -m src.clones.run_fase2d_sensibilidade_multicolinearidade --config experiments/clones/fase2d_sensibilidade_multicolinearidade.yaml

## Rodada Gerador de Referências (artigo)

python -m src.clones.gerar_referencia_fase2d_artigo --config-fase2d experiments/clones/fase2d_artigo_replicado.yaml

## Rodar Multicolinearidade (artigo)

python -m src.clones.run_fase2d_multicolinearidade_artigo --config experiments/clones/fase2d_artigo_replicado.yaml


==========================================================================================================
## ESTUDO DE CASO 2
==========================================================================================================


## Ativar Estudo de Caso 02 (mlops_river_level)

conda activate mlops_river_level

## 1- Executar Engenharia de Features

python -m src.river_level.run_engenharia_features --config experiments/river_level/features/baseline_features.yaml

## 2- Comando para executar a fase 3A (geração do dataset sequencial)

python -m src.river_level.run_preparar_sequencias --config experiments/river_level/baseline_lstm.yaml

## 3 - Execução da fase 3B (A Fase 3B do Estudo de Caso 2 corresponde ao preparo final dos dados para treinamento da LSTM, a partir do dataset sequencial gerado na Fase 3A.)

python -m src.river_level.run_preparar_treino --config experiments/river_level/baseline_lstm.yaml

## 4 - Execução da fase 4 (A Fase 4 do Estudo de Caso 2 corresponde ao treinamento da baseline LSTM e à avaliação do modelo no conjunto de teste)

python -m src.river_level.run_experimento_river_level --config experiments/river_level/baseline_lstm.yaml

## 5 - Refatoração Fase 4

1. python -m src.river_level.run_treinar_avaliar_modelo --config experiments/river_level/tuning_modelo/lstm_tuning_v1_base.yaml
2. python -m src.river_level.run_treinar_avaliar_modelo --config experiments/river_level/tuning_modelo/lstm_tuning_v1_smoke.yaml
3. python -m src.river_level.run_tuning_modelo --config experiments/river_level/tuning_modelo/lstm_tuning_v1_manual.yaml
4. python -m src.river_level.run_tuning_modelo --config experiments/river_level/tuning_modelo/lstm_tuning_v1_consolidacao.yaml
