# Decisão final da Fase 2 — Estudo de Caso 1

## Objetivo
Consolidar a escolha da configuração final do pipeline de segmentação de clones no Estudo de Caso 1, considerando qualidade interna, estabilidade entre execuções, ruído e utilidade operacional.

## Síntese do processo
A Fase 2 foi conduzida em duas etapas:

- **Fase 2A — tuning inicial**: exploração controlada do espaço de hiperparâmetros do pipeline UMAP + HDBSCAN.
- **Fase 2B — robustez multi-seed**: avaliação das configurações finalistas sob múltiplas sementes do embedding, com análise de estabilidade por AMI/NMI.

## Configuração oficial selecionada
A configuração oficial do Estudo de Caso 1 é:

- **UMAP**
  - `n_neighbors = 15`
  - `n_components = 10`
  - `min_dist = 0.05`
  - `metric = euclidean`

- **HDBSCAN**
  - `min_cluster_size = 5`
  - `min_samples = 3`
  - `cluster_selection_epsilon = 1.0`
  - `metric = euclidean`

## Justificativa da escolha
A escolha foi feita com base nos seguintes critérios:

1. a configuração está inserida em uma **região estável** do espaço de hiperparâmetros;
2. apresentou **alta concordância entre execuções** sob variação controlada de sementes;
3. manteve **ruído nulo ou desprezível**;
4. favorece uma segmentação com **cinco grupos**, mais adequada à interpretação tecnológica e à operacionalização das ações;
5. a utilidade prática do agrupamento foi tratada como critério central, e não apenas o valor máximo de métricas internas.

## Configuração exploratória vizinha
Foi mantida como evidência complementar a configuração:

- `n_neighbors = 10`
- `n_components = 10`
- `min_dist = 0.0`
- `min_cluster_size = 5`
- `min_samples = 3`
- `cluster_selection_epsilon = 1.0`

Essa configuração apresentou métricas internas fortes e alta estabilidade, porém mostrou tendência a maior subdivisão da estrutura latente, aproximando-se de uma solução com cerca de seis grupos, o que reduz a aderência operacional para o caso em estudo.

## Encaminhamento para a Fase 2C
A partir desta decisão, a etapa seguinte consiste em:

- executar a configuração oficial final;
- gerar artefatos finais da partição;
- caracterizar estatisticamente os cinco clusters;
- consolidar a narrativa interpretativa do Estudo de Caso 1 para a monografia.