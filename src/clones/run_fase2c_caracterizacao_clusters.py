from __future__ import annotations

"""
Script da Fase 2C para caracterização final dos clusters.

Objetivo:
- ler os artefatos da execução oficial da configuração final;
- usar diretamente a base analítica tratada persistida no pipeline;
- relacionar as labels finais dos clusters com os dados tratados;
- gerar tabelas descritivas dos clusters;
- gerar figuras para uso na monografia;
- gerar análise de multicolinearidade sobre a base tratada oficial;
- salvar saídas organizadas em reports/clones/fase2c_caracterizacao.

Entradas esperadas na pasta da execução:
- base_analitica_tratada.csv
- resultado_amostras.csv
- embedding.csv

Saídas principais:
- base_rotulada_clusters.csv
- cluster_tamanho.csv
- perfil_clusters_media.csv
- perfil_clusters_mediana.csv
- variaveis_discriminantes.csv
- distribuicao_especie_cluster.csv (se existir coluna ESPECIE)
- distribuicao_regiao_cluster.csv (se existir coluna REGIAO)
- grafico_clusters_2d.png
- grafico_tamanho_clusters.png
- heatmap_perfil_clusters.png
- grafico_barplot_normalizado_artigo.png
- grafico_barplot_normalizado_artigo.pdf
- scores_por_agrupamento_artigo.png
- scores_por_agrupamento_artigo.pdf
- score_medio_por_cluster_artigo.csv
- matriz_correlacao.csv
- pares_multicolineares.csv
- resumo_multicolinearidade_limiares.csv
- heatmap_multicolinearidade.png
"""

import argparse
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.feature_selection import f_classif
from sklearn.preprocessing import MinMaxScaler


# ======================================================================================
# Constantes do artigo
# ======================================================================================

VARIAVEIS_PERFIL_ARTIGO = [
    "F_DB",
    "Q_CEL",
    "Q_HEM",
    "Q_LIG_TOT",
    "Q_EXT",
    "P_KAPPA",
    "P_REJ",
    "P_RD",
    "L_ARES",
    "B_CLO2TOT",
    "B_OO_HEXA",
    "B_OO_VISCO",
]

# Formato: variavel -> (peso, maior_melhor)
PESOS_SCORE_ARTIGO = {
    "P_RD": (0.35, True),
    "B_CLO2TOT": (0.25, False),
    "F_DB": (0.20, True),
    "L_ARES": (0.15, True),
    "B_OO_VISCO": (0.05, True),
}


# ======================================================================================
# Utilidades
# ======================================================================================

def garantir_pasta(caminho: Path) -> Path:
    caminho.mkdir(parents=True, exist_ok=True)
    return caminho


def resolver_pasta_execucao(
    pasta_base: Path,
    pasta_execucao: Path | None = None,
) -> Path:
    """
    Resolve a pasta da execução.

    Regras:
    - se pasta_execucao for informada, usa diretamente;
    - caso contrário, procura subpastas iniciadas por 'exec_' dentro de pasta_base
      e escolhe a mais recente.
    """
    if pasta_execucao is not None:
        if not pasta_execucao.exists():
            raise FileNotFoundError(f"Pasta de execução não encontrada: {pasta_execucao}")
        return pasta_execucao

    if not pasta_base.exists():
        raise FileNotFoundError(f"Pasta base não encontrada: {pasta_base}")

    candidatas = [
        p for p in pasta_base.iterdir()
        if p.is_dir() and p.name.startswith("exec_")
    ]

    if not candidatas:
        raise FileNotFoundError(
            f"Nenhuma subpasta de execução encontrada em: {pasta_base}"
        )

    candidatas = sorted(candidatas, key=lambda p: p.stat().st_mtime, reverse=True)
    return candidatas[0]


def detectar_coluna_cluster(df: pd.DataFrame) -> str:
    candidatos = ["cluster", "label", "labels", "grupo", "cluster_id"]

    for coluna in candidatos:
        if coluna in df.columns:
            return coluna

    raise KeyError(
        "Nenhuma coluna de cluster encontrada. "
        "Esperado algo como: cluster, label, labels, grupo ou cluster_id."
    )


def detectar_colunas_categoricas_existentes(df: pd.DataFrame) -> dict[str, str]:
    mapa = {}

    if "ESPECIE" in df.columns:
        mapa["especie"] = "ESPECIE"

    if "REGIAO" in df.columns:
        mapa["regiao"] = "REGIAO"

    return mapa


def detectar_colunas_numericas(
    df: pd.DataFrame,
    coluna_cluster: str,
    colunas_excluir: Iterable[str] | None = None,
) -> list[str]:
    if colunas_excluir is None:
        colunas_excluir = []

    colunas_excluir = set(colunas_excluir)
    colunas_excluir.add(coluna_cluster)

    colunas_numericas = df.select_dtypes(include=[np.number]).columns.tolist()
    colunas_numericas = [c for c in colunas_numericas if c not in colunas_excluir]

    return colunas_numericas


def filtrar_variaveis_existentes(
    df: pd.DataFrame,
    variaveis: list[str],
) -> list[str]:
    return [col for col in variaveis if col in df.columns]


def padronizar_medias_por_coluna(df_media: pd.DataFrame) -> pd.DataFrame:
    """
    Padroniza as colunas por z-score usando média e desvio entre clusters.
    """
    df_pad = df_media.copy()

    for coluna in df_pad.columns:
        media = df_pad[coluna].mean()
        desvio = df_pad[coluna].std(ddof=0)

        if desvio == 0 or np.isnan(desvio):
            df_pad[coluna] = 0.0
        else:
            df_pad[coluna] = (df_pad[coluna] - media) / desvio

    return df_pad


# ======================================================================================
# Relação entre base tratada e labels do clustering
# ======================================================================================

def alinhar_base_tratada_com_clusters(
    df_base_tratada: pd.DataFrame,
    df_resultado_amostras: pd.DataFrame,
) -> tuple[pd.DataFrame, list[str]]:
    """
    Relaciona a base tratada com as labels do clustering.

    Estratégia:
    1. tenta alinhamento posicional;
    2. se existirem colunas ID e a ordem for a mesma, valida esse alinhamento;
    3. caso necessário, tenta merge por IDs + ordem de ocorrência.

    Retorna:
    - dataframe rotulado com coluna cluster
    - lista de colunas ID efetivamente usadas
    """
    coluna_cluster = detectar_coluna_cluster(df_resultado_amostras)

    df_base = df_base_tratada.reset_index(drop=True).copy()
    df_res = df_resultado_amostras.reset_index(drop=True).copy()

    if len(df_base) != len(df_res):
        raise ValueError(
            "base_analitica_tratada e resultado_amostras possuem quantidades de linhas diferentes. "
            f"base={len(df_base)} | resultado={len(df_res)}"
        )

    colunas_id_candidatas = ["TT", "MATGEN"]
    colunas_id_presentes = [
        c for c in colunas_id_candidatas
        if c in df_base.columns and c in df_res.columns
    ]

    # Tentativa 1: se houver IDs e estiverem na mesma ordem, usa alinhamento posicional validado.
    if colunas_id_presentes:
        ids_alinhados = True
        for col in colunas_id_presentes:
            serie_base = df_base[col].astype(str).fillna("")
            serie_res = df_res[col].astype(str).fillna("")
            if not serie_base.equals(serie_res):
                ids_alinhados = False
                break

        if ids_alinhados:
            df_rotulado = df_base.copy()
            df_rotulado[coluna_cluster] = df_res[coluna_cluster].values
            return df_rotulado, colunas_id_presentes

    # Tentativa 2: merge por IDs + ordem de ocorrência
    if colunas_id_presentes:
        df_base_occ = df_base.copy()
        df_res_occ = df_res[colunas_id_presentes + [coluna_cluster]].copy()

        df_base_occ["__ordem_ocorrencia"] = df_base_occ.groupby(colunas_id_presentes).cumcount()
        df_res_occ["__ordem_ocorrencia"] = df_res_occ.groupby(colunas_id_presentes).cumcount()

        chaves_merge = colunas_id_presentes + ["__ordem_ocorrencia"]

        df_rotulado = df_base_occ.merge(
            df_res_occ,
            on=chaves_merge,
            how="inner",
            validate="one_to_one",
        )

        if len(df_rotulado) == len(df_res_occ):
            df_rotulado = df_rotulado.drop(columns=["__ordem_ocorrencia"])
            return df_rotulado, colunas_id_presentes

    # Tentativa 3: fallback posicional puro
    df_rotulado = df_base.copy()
    df_rotulado[coluna_cluster] = df_res[coluna_cluster].values
    return df_rotulado, colunas_id_presentes


# ======================================================================================
# Tabelas de caracterização
# ======================================================================================

def calcular_variaveis_discriminantes(
    df: pd.DataFrame,
    coluna_cluster: str,
    colunas_numericas: list[str],
) -> pd.DataFrame:
    df_validacao = df[df[coluna_cluster] != -1].copy()

    if df_validacao.empty:
        return pd.DataFrame(columns=["variavel", "f_score", "p_valor"])

    X = df_validacao[colunas_numericas].copy()
    y = df_validacao[coluna_cluster].copy()

    colunas_validas = []
    for coluna in X.columns:
        serie = X[coluna]
        if serie.isna().any():
            continue
        if serie.nunique(dropna=True) <= 1:
            continue
        colunas_validas.append(coluna)

    if not colunas_validas:
        return pd.DataFrame(columns=["variavel", "f_score", "p_valor"])

    X = X[colunas_validas]

    f_scores, p_valores = f_classif(X, y)

    df_saida = pd.DataFrame({
        "variavel": colunas_validas,
        "f_score": f_scores,
        "p_valor": p_valores,
    }).sort_values("f_score", ascending=False)

    return df_saida.reset_index(drop=True)


def gerar_tabelas_caracterizacao(
    df_rotulado: pd.DataFrame,
    coluna_cluster: str,
    pasta_saida: Path,
    colunas_id: list[str],
) -> dict[str, Path]:
    caminhos_saida: dict[str, Path] = {}

    caminho_base_rotulada = pasta_saida / "base_rotulada_clusters.csv"
    df_rotulado.to_csv(caminho_base_rotulada, index=False, encoding="utf-8-sig")
    caminhos_saida["base_rotulada_clusters"] = caminho_base_rotulada

    df_sem_ruido = df_rotulado[df_rotulado[coluna_cluster] != -1].copy()

    df_tamanho = (
        df_sem_ruido.groupby(coluna_cluster)
        .size()
        .reset_index(name="quantidade")
        .sort_values(coluna_cluster)
    )
    caminho_tamanho = pasta_saida / "cluster_tamanho.csv"
    df_tamanho.to_csv(caminho_tamanho, index=False, encoding="utf-8-sig")
    caminhos_saida["cluster_tamanho"] = caminho_tamanho

    mapa_categoricas = detectar_colunas_categoricas_existentes(df_rotulado)

    if "especie" in mapa_categoricas:
        coluna_especie = mapa_categoricas["especie"]
        df_especie = pd.crosstab(
            df_sem_ruido[coluna_cluster],
            df_sem_ruido[coluna_especie],
            normalize="index",
        ) * 100.0
        caminho_especie = pasta_saida / "distribuicao_especie_cluster.csv"
        df_especie.to_csv(caminho_especie, encoding="utf-8-sig")
        caminhos_saida["distribuicao_especie_cluster"] = caminho_especie

    if "regiao" in mapa_categoricas:
        coluna_regiao = mapa_categoricas["regiao"]
        df_regiao = pd.crosstab(
            df_sem_ruido[coluna_cluster],
            df_sem_ruido[coluna_regiao],
            normalize="index",
        ) * 100.0
        caminho_regiao = pasta_saida / "distribuicao_regiao_cluster.csv"
        df_regiao.to_csv(caminho_regiao, encoding="utf-8-sig")
        caminhos_saida["distribuicao_regiao_cluster"] = caminho_regiao

    colunas_excluir = set(colunas_id) | {
        coluna_cluster, "ARV", "PROC",
    }
    colunas_numericas = detectar_colunas_numericas(
        df=df_rotulado,
        coluna_cluster=coluna_cluster,
        colunas_excluir=list(colunas_excluir),
    )

    if not colunas_numericas:
        raise ValueError("Nenhuma coluna numérica útil foi encontrada para caracterização.")

    df_media = (
        df_sem_ruido.groupby(coluna_cluster)[colunas_numericas]
        .mean()
        .sort_index()
    )
    caminho_media = pasta_saida / "perfil_clusters_media.csv"
    df_media.to_csv(caminho_media, encoding="utf-8-sig")
    caminhos_saida["perfil_clusters_media"] = caminho_media

    df_mediana = (
        df_sem_ruido.groupby(coluna_cluster)[colunas_numericas]
        .median()
        .sort_index()
    )
    caminho_mediana = pasta_saida / "perfil_clusters_mediana.csv"
    df_mediana.to_csv(caminho_mediana, encoding="utf-8-sig")
    caminhos_saida["perfil_clusters_mediana"] = caminho_mediana

    df_discriminantes = calcular_variaveis_discriminantes(
        df=df_rotulado,
        coluna_cluster=coluna_cluster,
        colunas_numericas=colunas_numericas,
    )
    caminho_discriminantes = pasta_saida / "variaveis_discriminantes.csv"
    df_discriminantes.to_csv(caminho_discriminantes, index=False, encoding="utf-8-sig")
    caminhos_saida["variaveis_discriminantes"] = caminho_discriminantes

    return caminhos_saida


# ======================================================================================
# Gráficos de agrupamento
# ======================================================================================

def gerar_grafico_clusters_2d(
    df_resultado: pd.DataFrame,
    df_embedding: pd.DataFrame,
    coluna_cluster: str,
    pasta_saida: Path,
) -> Path | None:
    if len(df_resultado) != len(df_embedding):
        print("Aviso: resultado_amostras e embedding têm tamanhos diferentes. Gráfico 2D não será gerado.")
        return None

    colunas_numericas_embedding = df_embedding.select_dtypes(include=[np.number]).columns.tolist()

    if len(colunas_numericas_embedding) < 2:
        print("Aviso: embedding possui menos de duas colunas numéricas. Gráfico 2D não será gerado.")
        return None

    eixo_x = colunas_numericas_embedding[0]
    eixo_y = colunas_numericas_embedding[1]

    df_plot = pd.DataFrame({
        "x": df_embedding[eixo_x].values,
        "y": df_embedding[eixo_y].values,
        "cluster": df_resultado[coluna_cluster].values,
    })

    plt.figure(figsize=(10, 8))

    clusters_validos = sorted(df_plot["cluster"].dropna().unique().tolist())

    for cluster in clusters_validos:
        mascara = df_plot["cluster"] == cluster
        rotulo = "Ruído" if cluster == -1 else f"Cluster {int(cluster)}"
        alpha = 0.45 if cluster == -1 else 0.85

        plt.scatter(
            df_plot.loc[mascara, "x"],
            df_plot.loc[mascara, "y"],
            s=28,
            alpha=alpha,
            label=rotulo,
        )

    plt.xlabel(eixo_x)
    plt.ylabel(eixo_y)
    plt.title("Embedding 2D da configuração oficial")
    plt.legend(loc="best", fontsize=8)
    plt.tight_layout()

    caminho_saida = pasta_saida / "grafico_clusters_2d.png"
    plt.savefig(caminho_saida, dpi=300, bbox_inches="tight")
    plt.close()

    return caminho_saida


def gerar_grafico_tamanho_clusters(
    df_rotulado: pd.DataFrame,
    coluna_cluster: str,
    pasta_saida: Path,
) -> Path:
    df_sem_ruido = df_rotulado[df_rotulado[coluna_cluster] != -1].copy()

    df_tamanho = (
        df_sem_ruido.groupby(coluna_cluster)
        .size()
        .reset_index(name="quantidade")
        .sort_values(coluna_cluster)
    )

    plt.figure(figsize=(8, 5))
    plt.bar(df_tamanho[coluna_cluster].astype(str), df_tamanho["quantidade"])
    plt.xlabel("Cluster")
    plt.ylabel("Quantidade de amostras")
    plt.title("Tamanho dos clusters")
    plt.tight_layout()

    caminho_saida = pasta_saida / "grafico_tamanho_clusters.png"
    plt.savefig(caminho_saida, dpi=300, bbox_inches="tight")
    plt.close()

    return caminho_saida


def gerar_heatmap_perfil_clusters(
    df_rotulado: pd.DataFrame,
    coluna_cluster: str,
    pasta_saida: Path,
    colunas_id: list[str],
) -> Path:
    df_sem_ruido = df_rotulado[df_rotulado[coluna_cluster] != -1].copy()

    colunas_excluir = set(colunas_id) | {
        coluna_cluster, "ARV", "PROC",
    }
    colunas_numericas = detectar_colunas_numericas(
        df=df_rotulado,
        coluna_cluster=coluna_cluster,
        colunas_excluir=list(colunas_excluir),
    )

    df_media = (
        df_sem_ruido.groupby(coluna_cluster)[colunas_numericas]
        .mean()
        .sort_index()
    )

    df_heatmap = padronizar_medias_por_coluna(df_media)

    plt.figure(figsize=(max(10, len(df_heatmap.columns) * 0.35), 6))
    plt.imshow(df_heatmap.values, aspect="auto")
    plt.colorbar(label="Média padronizada (z-score)")
    plt.yticks(
        ticks=np.arange(len(df_heatmap.index)),
        labels=[f"Cluster {int(idx)}" for idx in df_heatmap.index],
    )
    plt.xticks(
        ticks=np.arange(len(df_heatmap.columns)),
        labels=df_heatmap.columns,
        rotation=90,
    )
    plt.title("Perfil médio padronizado das variáveis por cluster")
    plt.tight_layout()

    caminho_saida = pasta_saida / "heatmap_perfil_clusters.png"
    plt.savefig(caminho_saida, dpi=300, bbox_inches="tight")
    plt.close()

    return caminho_saida


def gerar_grafico_distribuicao_categorica(
    df_rotulado: pd.DataFrame,
    coluna_cluster: str,
    coluna_categorica: str,
    nome_arquivo: str,
    titulo: str,
    pasta_saida: Path,
) -> Path:
    df_sem_ruido = df_rotulado[df_rotulado[coluna_cluster] != -1].copy()

    tabela = pd.crosstab(
        df_sem_ruido[coluna_cluster],
        df_sem_ruido[coluna_categorica],
        normalize="index",
    ) * 100.0

    tabela.plot(
        kind="bar",
        stacked=True,
        figsize=(10, 6),
    )
    plt.xlabel("Cluster")
    plt.ylabel("Percentual (%)")
    plt.title(titulo)
    plt.legend(title=coluna_categorica, bbox_to_anchor=(1.02, 1), loc="upper left")
    plt.tight_layout()

    caminho_saida = pasta_saida / nome_arquivo
    plt.savefig(caminho_saida, dpi=300, bbox_inches="tight")
    plt.close()

    return caminho_saida


# ======================================================================================
# Gráficos no estilo do artigo
# ======================================================================================

def calcular_score_ponderado_artigo(
    df: pd.DataFrame,
    pesos_score: dict[str, tuple[float, bool]],
) -> pd.Series:
    variaveis = filtrar_variaveis_existentes(df, list(pesos_score.keys()))

    if not variaveis:
        raise ValueError("Nenhuma variável do score do artigo foi encontrada no DataFrame.")

    df_base = df[variaveis].copy()

    if df_base.isna().any().any():
        raise ValueError(
            "Há valores ausentes nas variáveis do score. "
            "É necessário resolver isso antes de calcular o score."
        )

    scaler = MinMaxScaler()
    df_norm = pd.DataFrame(
        scaler.fit_transform(df_base),
        columns=variaveis,
        index=df.index,
    )

    score = pd.Series(0.0, index=df.index, dtype=float)

    for variavel in variaveis:
        peso, maior_melhor = pesos_score[variavel]
        valores = df_norm[variavel]

        if not maior_melhor:
            valores = 1.0 - valores

        score = score + (peso * valores)

    return score


def gerar_barplot_perfil_normalizado_artigo(
    df_rotulado: pd.DataFrame,
    coluna_cluster: str,
    pasta_saida: Path,
    variaveis: list[str] | None = None,
    fontsize_labels: int = 12,
) -> tuple[Path | None, Path | None]:
    if variaveis is None:
        variaveis = VARIAVEIS_PERFIL_ARTIGO

    variaveis = filtrar_variaveis_existentes(df_rotulado, variaveis)

    if not variaveis:
        print("Aviso: nenhuma variável do perfil do artigo foi encontrada. Gráfico não será gerado.")
        return None, None

    df_plot = df_rotulado[df_rotulado[coluna_cluster] != -1].copy()

    if df_plot.empty:
        print("Aviso: não há amostras válidas sem ruído para o gráfico de perfil.")
        return None, None

    if df_plot[variaveis].isna().any().any():
        print("Aviso: há valores ausentes nas variáveis do perfil. Gráfico não será gerado.")
        return None, None

    scaler = MinMaxScaler()
    df_plot[variaveis] = scaler.fit_transform(df_plot[variaveis])

    df_melt = df_plot[[coluna_cluster] + variaveis].melt(
        id_vars=coluna_cluster,
        var_name="variavel",
        value_name="valor",
    )

    df_stats = (
        df_melt.groupby([coluna_cluster, "variavel"])["valor"]
        .agg(["mean", "std", "count"])
        .reset_index()
    )
    df_stats["ci95"] = 1.96 * (df_stats["std"] / np.sqrt(df_stats["count"].clip(lower=1)))
    df_stats["ci95"] = df_stats["ci95"].fillna(0.0)

    clusters = sorted(df_stats[coluna_cluster].unique().tolist())
    n_clusters = len(clusters)
    n_variaveis = len(variaveis)

    x = np.arange(n_variaveis)
    largura = 0.8 / max(n_clusters, 1)

    fig, ax = plt.subplots(figsize=(15, 4))
    cmap = plt.cm.get_cmap("viridis", n_clusters)

    for i, cluster in enumerate(clusters):
        df_cluster = (
            df_stats[df_stats[coluna_cluster] == cluster]
            .set_index("variavel")
            .reindex(variaveis)
            .reset_index()
        )

        deslocamento = (i - (n_clusters - 1) / 2) * largura

        ax.bar(
            x + deslocamento,
            df_cluster["mean"].values,
            width=largura,
            yerr=df_cluster["ci95"].values,
            capsize=3,
            label=f"Cluster {int(cluster)}",
            color=cmap(i),
            edgecolor="black",
            linewidth=0.5,
        )

    ax.set_xlabel("Features", fontsize=fontsize_labels)
    ax.set_ylabel("Normalized Values", fontsize=fontsize_labels)
    ax.set_xticks(x)
    ax.set_xticklabels(variaveis, rotation=90, fontsize=fontsize_labels)
    ax.tick_params(axis="y", labelsize=fontsize_labels)
    ax.grid(True, axis="y", color="grey", linewidth=0.1)

    for spine in ax.spines.values():
        spine.set_linewidth(0.8)
        spine.set_edgecolor("black")

    ax.legend(
        title="Clusters",
        loc="upper right",
        ncol=max(1, n_clusters),
        framealpha=1,
        facecolor="white",
        fontsize=fontsize_labels,
        title_fontsize=fontsize_labels,
    )

    plt.tight_layout()

    caminho_png = pasta_saida / "grafico_barplot_normalizado_artigo.png"
    caminho_pdf = pasta_saida / "grafico_barplot_normalizado_artigo.pdf"

    plt.savefig(caminho_png, dpi=300, bbox_inches="tight")
    plt.savefig(caminho_pdf, format="pdf", bbox_inches="tight")
    plt.close()

    return caminho_png, caminho_pdf


def gerar_score_por_cluster_artigo(
    df_rotulado: pd.DataFrame,
    coluna_cluster: str,
    pasta_saida: Path,
    pesos_score: dict[str, tuple[float, bool]] | None = None,
    fontsize_labels: int = 12,
) -> tuple[Path | None, Path | None, Path | None]:
    if pesos_score is None:
        pesos_score = PESOS_SCORE_ARTIGO

    df_plot = df_rotulado[df_rotulado[coluna_cluster] != -1].copy()

    if df_plot.empty:
        print("Aviso: não há amostras válidas sem ruído para o gráfico de score.")
        return None, None, None

    variaveis_score = filtrar_variaveis_existentes(df_plot, list(pesos_score.keys()))
    if not variaveis_score:
        print("Aviso: nenhuma variável do score do artigo foi encontrada. Gráfico não será gerado.")
        return None, None, None

    if df_plot[variaveis_score].isna().any().any():
        print("Aviso: há valores ausentes nas variáveis do score. Gráfico não será gerado.")
        return None, None, None

    df_plot["Score"] = calcular_score_ponderado_artigo(df_plot, pesos_score)

    df_score_cluster = (
        df_plot.groupby(coluna_cluster)["Score"]
        .agg(["mean", "std", "count"])
        .reset_index()
        .rename(columns={
            "mean": "score_medio",
            "std": "score_desvio",
            "count": "quantidade",
        })
    )

    df_score_cluster["ci95"] = 1.96 * (
        df_score_cluster["score_desvio"] / np.sqrt(df_score_cluster["quantidade"].clip(lower=1))
    )
    df_score_cluster["ci95"] = df_score_cluster["ci95"].fillna(0.0)

    df_score_cluster = df_score_cluster.sort_values("score_medio", ascending=False).reset_index(drop=True)

    fig, ax = plt.subplots(figsize=(5, 3))
    x = np.arange(len(df_score_cluster))

    ax.bar(
        x,
        df_score_cluster["score_medio"].values,
        yerr=df_score_cluster["ci95"].values,
        capsize=4,
        edgecolor="black",
        linewidth=0.5,
    )

    ax.set_xlabel("Clusters", fontsize=fontsize_labels)
    ax.set_ylabel("Score", fontsize=fontsize_labels)
    ax.set_xticks(x)
    ax.set_xticklabels(df_score_cluster[coluna_cluster].astype(str).tolist(), fontsize=fontsize_labels)
    ax.tick_params(axis="y", labelsize=fontsize_labels)
    ax.grid(True, axis="y", color="grey", linewidth=0.1)

    for spine in ax.spines.values():
        spine.set_edgecolor("black")
        spine.set_linewidth(0.8)

    plt.tight_layout()

    caminho_png = pasta_saida / "scores_por_agrupamento_artigo.png"
    caminho_pdf = pasta_saida / "scores_por_agrupamento_artigo.pdf"
    caminho_csv = pasta_saida / "score_medio_por_cluster_artigo.csv"

    plt.savefig(caminho_png, dpi=300, bbox_inches="tight")
    plt.savefig(caminho_pdf, format="pdf", bbox_inches="tight")
    plt.close()

    df_score_cluster.to_csv(caminho_csv, index=False, encoding="utf-8-sig")

    return caminho_png, caminho_pdf, caminho_csv


# ======================================================================================
# Multicolinearidade
# ======================================================================================

def calcular_multicolinearidade(
    df_rotulado: pd.DataFrame,
    coluna_cluster: str,
    pasta_saida: Path,
    colunas_id: list[str],
) -> dict[str, Path]:
    """
    Gera artefatos de multicolinearidade com base nas variáveis numéricas da base tratada.
    """
    caminhos_saida: dict[str, Path] = {}

    colunas_excluir = set(colunas_id) | {coluna_cluster, "ARV", "PROC"}
    colunas_numericas = detectar_colunas_numericas(
        df=df_rotulado,
        coluna_cluster=coluna_cluster,
        colunas_excluir=list(colunas_excluir),
    )

    if not colunas_numericas:
        return caminhos_saida

    df_num = df_rotulado[colunas_numericas].copy()

    # Remove colunas com NaN ou variância zero
    colunas_validas = []
    for coluna in df_num.columns:
        serie = df_num[coluna]
        if serie.isna().any():
            continue
        if serie.nunique(dropna=True) <= 1:
            continue
        colunas_validas.append(coluna)

    if not colunas_validas:
        return caminhos_saida

    df_num = df_num[colunas_validas]

    matriz_corr = df_num.corr(method="pearson")
    caminho_corr = pasta_saida / "matriz_correlacao.csv"
    matriz_corr.to_csv(caminho_corr, encoding="utf-8-sig")
    caminhos_saida["matriz_correlacao"] = caminho_corr

    registros_pares = []
    colunas = matriz_corr.columns.tolist()

    for i in range(len(colunas)):
        for j in range(i + 1, len(colunas)):
            corr = matriz_corr.iloc[i, j]
            registros_pares.append({
                "variavel_1": colunas[i],
                "variavel_2": colunas[j],
                "correlacao": float(corr),
                "correlacao_abs": float(abs(corr)),
            })

    df_pares = pd.DataFrame(registros_pares).sort_values("correlacao_abs", ascending=False)

    caminho_pares = pasta_saida / "pares_multicolineares.csv"
    df_pares.to_csv(caminho_pares, index=False, encoding="utf-8-sig")
    caminhos_saida["pares_multicolineares"] = caminho_pares

    limiares = [0.50, 0.60, 0.70, 0.80, 0.90, 0.95]
    resumo_limiares = []
    for limiar in limiares:
        qtd = int((df_pares["correlacao_abs"] >= limiar).sum())
        resumo_limiares.append({
            "limiar_correlacao_abs": limiar,
            "quantidade_pares": qtd,
        })

    df_resumo = pd.DataFrame(resumo_limiares)
    caminho_resumo = pasta_saida / "resumo_multicolinearidade_limiares.csv"
    df_resumo.to_csv(caminho_resumo, index=False, encoding="utf-8-sig")
    caminhos_saida["resumo_multicolinearidade_limiares"] = caminho_resumo

    variaveis_heatmap = filtrar_variaveis_existentes(df_num, VARIAVEIS_PERFIL_ARTIGO)
    if len(variaveis_heatmap) < 2:
        variaveis_heatmap = (
            df_pares["variavel_1"].tolist() + df_pares["variavel_2"].tolist()
        )
        variaveis_heatmap = list(dict.fromkeys(variaveis_heatmap))[:20]

    if len(variaveis_heatmap) >= 2:
        matriz_heatmap = matriz_corr.loc[variaveis_heatmap, variaveis_heatmap]

        plt.figure(figsize=(max(8, len(variaveis_heatmap) * 0.5), max(6, len(variaveis_heatmap) * 0.5)))
        plt.imshow(matriz_heatmap.values, vmin=-1, vmax=1, aspect="auto")
        plt.colorbar(label="Correlação de Pearson")
        plt.xticks(
            ticks=np.arange(len(matriz_heatmap.columns)),
            labels=matriz_heatmap.columns,
            rotation=90,
        )
        plt.yticks(
            ticks=np.arange(len(matriz_heatmap.index)),
            labels=matriz_heatmap.index,
        )
        plt.title("Heatmap de multicolinearidade")
        plt.tight_layout()

        caminho_heatmap = pasta_saida / "heatmap_multicolinearidade.png"
        plt.savefig(caminho_heatmap, dpi=300, bbox_inches="tight")
        plt.close()

        caminhos_saida["heatmap_multicolinearidade"] = caminho_heatmap

    return caminhos_saida


# ======================================================================================
# Execução principal
# ======================================================================================

def criar_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Gera a caracterização final dos clusters da Fase 2C."
    )

    parser.add_argument(
        "--pasta-base",
        type=str,
        default="artifacts/clones/fase2c_config_final",
        help="Pasta base onde está a execução oficial da Fase 2C.",
    )

    parser.add_argument(
        "--pasta-execucao",
        type=str,
        default=None,
        help="Pasta específica da execução. Se omitida, usa a subpasta exec_* mais recente.",
    )

    parser.add_argument(
        "--saida",
        type=str,
        default="reports/clones/fase2c_caracterizacao",
        help="Pasta de saída para tabelas e figuras.",
    )

    return parser


def main() -> None:
    parser = criar_parser()
    args = parser.parse_args()

    pasta_base = Path(args.pasta_base)
    pasta_execucao = Path(args.pasta_execucao) if args.pasta_execucao else None
    pasta_saida = garantir_pasta(Path(args.saida))

    pasta_execucao_resolvida = resolver_pasta_execucao(
        pasta_base=pasta_base,
        pasta_execucao=pasta_execucao,
    )

    caminho_base_tratada = pasta_execucao_resolvida / "base_analitica_tratada.csv"
    caminho_resultado = pasta_execucao_resolvida / "resultado_amostras.csv"
    caminho_embedding = pasta_execucao_resolvida / "embedding.csv"

    if not caminho_base_tratada.exists():
        raise FileNotFoundError(f"Arquivo não encontrado: {caminho_base_tratada}")

    if not caminho_resultado.exists():
        raise FileNotFoundError(f"Arquivo não encontrado: {caminho_resultado}")

    if not caminho_embedding.exists():
        raise FileNotFoundError(f"Arquivo não encontrado: {caminho_embedding}")

    df_base_tratada = pd.read_csv(caminho_base_tratada)
    df_resultado = pd.read_csv(caminho_resultado)
    df_embedding = pd.read_csv(caminho_embedding)

    coluna_cluster = detectar_coluna_cluster(df_resultado)
    df_rotulado, colunas_id = alinhar_base_tratada_com_clusters(df_base_tratada, df_resultado)

    print("=" * 80)
    print("FASE 2C — CARACTERIZAÇÃO FINAL DOS CLUSTERS")
    print("=" * 80)
    print(f"Pasta da execução: {pasta_execucao_resolvida}")
    print(f"Coluna de cluster detectada: {coluna_cluster}")
    print(f"Quantidade de linhas em base_analitica_tratada: {len(df_base_tratada)}")
    print(f"Quantidade de linhas em resultado_amostras: {len(df_resultado)}")
    print(f"Quantidade de linhas em embedding: {len(df_embedding)}")
    print(f"Quantidade de linhas na base rotulada final: {len(df_rotulado)}")
    print()

    caminhos_tabelas = gerar_tabelas_caracterizacao(
        df_rotulado=df_rotulado,
        coluna_cluster=coluna_cluster,
        pasta_saida=pasta_saida,
        colunas_id=colunas_id,
    )

    caminho_clusters_2d = gerar_grafico_clusters_2d(
        df_resultado=df_resultado,
        df_embedding=df_embedding,
        coluna_cluster=coluna_cluster,
        pasta_saida=pasta_saida,
    )

    caminho_tamanho = gerar_grafico_tamanho_clusters(
        df_rotulado=df_rotulado,
        coluna_cluster=coluna_cluster,
        pasta_saida=pasta_saida,
    )

    caminho_heatmap = gerar_heatmap_perfil_clusters(
        df_rotulado=df_rotulado,
        coluna_cluster=coluna_cluster,
        pasta_saida=pasta_saida,
        colunas_id=colunas_id,
    )

    mapa_categoricas = detectar_colunas_categoricas_existentes(df_rotulado)

    caminho_especie = None
    caminho_regiao = None

    if "especie" in mapa_categoricas:
        caminho_especie = gerar_grafico_distribuicao_categorica(
            df_rotulado=df_rotulado,
            coluna_cluster=coluna_cluster,
            coluna_categorica=mapa_categoricas["especie"],
            nome_arquivo="grafico_especie_por_cluster.png",
            titulo="Distribuição de espécie por cluster",
            pasta_saida=pasta_saida,
        )

    if "regiao" in mapa_categoricas:
        caminho_regiao = gerar_grafico_distribuicao_categorica(
            df_rotulado=df_rotulado,
            coluna_cluster=coluna_cluster,
            coluna_categorica=mapa_categoricas["regiao"],
            nome_arquivo="grafico_regiao_por_cluster.png",
            titulo="Distribuição de região por cluster",
            pasta_saida=pasta_saida,
        )

    caminho_barplot_artigo_png = None
    caminho_barplot_artigo_pdf = None
    caminho_score_artigo_png = None
    caminho_score_artigo_pdf = None
    caminho_score_artigo_csv = None

    caminho_barplot_artigo_png, caminho_barplot_artigo_pdf = gerar_barplot_perfil_normalizado_artigo(
        df_rotulado=df_rotulado,
        coluna_cluster=coluna_cluster,
        pasta_saida=pasta_saida,
    )

    (
        caminho_score_artigo_png,
        caminho_score_artigo_pdf,
        caminho_score_artigo_csv,
    ) = gerar_score_por_cluster_artigo(
        df_rotulado=df_rotulado,
        coluna_cluster=coluna_cluster,
        pasta_saida=pasta_saida,
    )

    caminhos_multicol = calcular_multicolinearidade(
        df_rotulado=df_rotulado,
        coluna_cluster=coluna_cluster,
        pasta_saida=pasta_saida,
        colunas_id=colunas_id,
    )

    print("Tabelas geradas:")
    for nome, caminho in caminhos_tabelas.items():
        print(f"- {nome}: {caminho}")

    print()
    print("Figuras geradas:")
    if caminho_clusters_2d is not None:
        print(f"- grafico_clusters_2d: {caminho_clusters_2d}")
    print(f"- grafico_tamanho_clusters: {caminho_tamanho}")
    print(f"- heatmap_perfil_clusters: {caminho_heatmap}")
    if caminho_especie is not None:
        print(f"- grafico_especie_por_cluster: {caminho_especie}")
    if caminho_regiao is not None:
        print(f"- grafico_regiao_por_cluster: {caminho_regiao}")
    if caminho_barplot_artigo_png is not None:
        print(f"- grafico_barplot_normalizado_artigo_png: {caminho_barplot_artigo_png}")
    if caminho_barplot_artigo_pdf is not None:
        print(f"- grafico_barplot_normalizado_artigo_pdf: {caminho_barplot_artigo_pdf}")
    if caminho_score_artigo_png is not None:
        print(f"- scores_por_agrupamento_artigo_png: {caminho_score_artigo_png}")
    if caminho_score_artigo_pdf is not None:
        print(f"- scores_por_agrupamento_artigo_pdf: {caminho_score_artigo_pdf}")
    if caminho_score_artigo_csv is not None:
        print(f"- score_medio_por_cluster_artigo_csv: {caminho_score_artigo_csv}")

    if caminhos_multicol:
        print()
        print("Saídas de multicolinearidade:")
        for nome, caminho in caminhos_multicol.items():
            print(f"- {nome}: {caminho}")

    print()
    print("Execução concluída com sucesso.")


if __name__ == "__main__":
    main()