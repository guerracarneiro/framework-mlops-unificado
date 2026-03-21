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
4. Entrar na pasta do projeto (cd FrameWork_MLOps_Unificado)
5. git status (verificar se git ativo)
6. ### Gerar Zip ###
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



