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

## Iniciar Ambiente

1. Terminal > New Terminal (abrir novo terminal)
2. Saida de ambientes VENV (comando: deactivate)
3. conda activate mlops_clones (entrar no Conda)
4. Entrar na pasta do projeto (FrameWork_MLOps_Unificado)
5. git status (verificar se git ativo)

## Rodar o Experimento

python -m src.clones.run_experimento_clones --config experiments/clones/fase1_preprocessamento.yaml




