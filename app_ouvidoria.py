"""Ouvidoria Inteligente — aplicação Streamlit."""
import numpy as np
import pandas as pd
import streamlit as st
from ouvidoria import (
    MODELOS, carregar, representar, buscar, dividir, cor_score,
    detectar_duplicatas, matriz_heatmap, grafico_2d, cosseno_matriz
)

st.set_page_config(page_title="Ouvidoria Semântica", page_icon="🏛️", layout="wide")

@st.cache_data
def base():
    return carregar()

@st.cache_data
def embeddings(textos, modelo):
    return representar(tuple(textos), modelo)

def main():
    st.title("🏛️ Ouvidoria Semântica")
    st.caption("Busca por significado, identificação de duplicatas e análise de manifestações longas.")

    df = base()
    modelo_nome = st.sidebar.selectbox("Modelo de embedding", list(MODELOS))
    top_k = st.sidebar.slider("Quantidade de resultados", 1, 10, 5)
    st.sidebar.markdown("**Modelos disponíveis**")
    st.sidebar.caption("Os modelos são multilíngues e baixados na primeira execução.")

    tabs = st.tabs([
        "🔎 Busca Semântica",
        "📋 Base Completa",
        "🗺️ Espaço Vetorial",
        "🧩 Chunking",
    ])

    with tabs[0]:
        consulta = st.text_area(
            "Descreva o problema do cidadão",
            "A rua está com o asfalto muito danificado e cheia de buracos.",
            height=90,
        )
        if st.button("Pesquisar manifestações", type="primary"):
            if not consulta.strip():
                st.warning("Informe uma descrição.")
            else:
                try:
                    resultados = buscar(df, consulta, modelo_nome, top_k)
                    for r in resultados.itertuples():
                        st.markdown(
                            f"### {cor_score(r.score)} {r.id} — {r.score:.3f}"
                        )
                        st.caption(f"Categoria oficial: {r.categoria_oficial}")
                        st.write(r.texto)
                except Exception as exc:
                    st.error(f"Não foi possível gerar a busca: {exc}")

    with tabs[1]:
        st.dataframe(df, use_container_width=True, hide_index=True)
        if st.button("Calcular matriz 40 × 40"):
            try:
                emb = embeddings(df.texto.tolist(), modelo_nome)
                sim = cosseno_matriz(emb)
                st.pyplot(matriz_heatmap(emb))
                st.download_button(
                    "Baixar matriz CSV",
                    pd.DataFrame(sim, index=df.id, columns=df.id).to_csv().encode("utf-8"),
                    "matriz_similaridade.csv",
                    "text/csv",
                )
            except Exception as exc:
                st.error(f"Falha ao calcular a matriz: {exc}")

        limiar = st.slider("Limiar para candidatos a duplicata", 0.50, 0.99, 0.85, 0.01)
        if st.button("Executar detecção de duplicatas"):
            try:
                pares = detectar_duplicatas(df.texto.tolist(), limiar, modelo_nome)
                tabela = [
                    {"ID 1":df.id.iloc[p["i"]], "ID 2":df.id.iloc[p["j"]],
                     "Cosseno":round(p["similaridade"],4)}
                    for p in pares
                ]
                st.dataframe(tabela, use_container_width=True, hide_index=True)
                st.caption("O resultado é uma lista de candidatos; similaridade alta não prova que duas manifestações sejam a mesma ocorrência.")
            except Exception as exc:
                st.error(f"Falha na detecção: {exc}")

    with tabs[2]:
        metodo = st.radio("Projeção", ["PCA", "t-SNE"], horizontal=True)
        if st.button("Projetar corpus"):
            try:
                emb = embeddings(df.texto.tolist(), modelo_nome)
                st.pyplot(grafico_2d(
                    emb, df.categoria_oficial.tolist(), metodo, df.id.tolist()
                ))
                st.info(
                    "As cores representam as categorias oficiais. A proximidade "
                    "em duas dimensões é uma visualização e não substitui a "
                    "similaridade no espaço original."
                )
            except Exception as exc:
                st.error(f"Falha na projeção: {exc}")

    with tabs[3]:
        padrao = df.loc[df.texto.str.len().idxmax(), "texto"]
        texto = st.text_area("Cole uma manifestação longa", padrao, height=220)
        estrategia = st.selectbox("Estratégia", ["Recursivo", "Janela fixa"])
        tamanho = st.slider("chunk_size", 100, 600, 320, 20)
        overlap = st.slider("chunk_overlap", 0, tamanho-1, min(70, tamanho-1), 10)
        if st.button("Gerar chunks"):
            try:
                partes = dividir(texto, tamanho, overlap, estrategia)
                if not partes:
                    st.warning("Texto vazio.")
                else:
                    st.dataframe(
                        [{"Chunk":i+1,"Caracteres":len(p),"Texto":p}
                         for i,p in enumerate(partes)],
                        use_container_width=True, hide_index=True
                    )
                    emb = representar(tuple(partes), modelo_nome)
                    st.dataframe(
                        pd.DataFrame(emb, index=[f"Chunk {i+1}" for i in range(len(partes))]),
                        use_container_width=True
                    )
                    st.pyplot(grafico_2d(
                        emb, ["mesma manifestação"]*len(partes), "PCA",
                        [f"C{i+1}" for i in range(len(partes))]
                    ))
            except Exception as exc:
                st.error(f"Falha no chunking/embedding: {exc}")

if __name__ == "__main__":
    main()
