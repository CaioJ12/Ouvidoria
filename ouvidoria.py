"""Núcleo da Ouvidoria Inteligente.

A interface e os notebooks usam este módulo. O projeto foi estruturado para
separar carregamento de dados, representações esparsas, embeddings, busca,
duplicatas, projeção e chunking.
"""
from __future__ import annotations
from pathlib import Path
from functools import lru_cache
import json
import numpy as np
import pandas as pd

MODELOS = {
    "MiniLM multilíngue": "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
    "DistilUSE multilíngue": "sentence-transformers/distiluse-base-multilingual-cased-v2",
}

BASE = Path(__file__).resolve().parent


def carregar(caminho=BASE/"manifestacoes.json"):
    df = pd.DataFrame(json.loads(Path(caminho).read_text(encoding="utf-8")))
    obrigatorios = {"id","data","categoria_oficial","texto"}
    if not obrigatorios.issubset(df.columns):
        raise ValueError("manifestacoes.json precisa conter id, data, categoria_oficial e texto.")
    if len(df) != 40:
        raise ValueError("O corpus desta atividade deve possuir 40 manifestações.")
    if not df.texto.str.len().between(50,800).all():
        raise ValueError("Cada texto deve ter entre 50 e 800 caracteres.")
    return df


def normalizar_texto(texto):
    import re, unicodedata
    texto = unicodedata.normalize("NFD", texto.casefold())
    texto = "".join(c for c in texto if unicodedata.category(c) != "Mn")
    return re.sub(r"\s+", " ", texto).strip()


@lru_cache(maxsize=4)
def _modelo(nome):
    from sentence_transformers import SentenceTransformer
    if nome not in MODELOS:
        raise ValueError(f"Modelo desconhecido: {nome}")
    return SentenceTransformer(MODELOS[nome])


def representar(textos, nome="MiniLM multilíngue"):
    """Retorna embeddings normalizados; documentos longos são agregados por janelas."""
    model = _modelo(nome)
    limite = getattr(model, "max_seq_length", 256)
    resultados = []
    for texto in textos:
        # Tokenização aproximada por palavras para evitar truncamento silencioso.
        palavras = str(texto).split()
        if len(palavras) <= int(limite * 0.75):
            vec = model.encode([str(texto)], normalize_embeddings=True)[0]
        else:
            janela = max(20, int(limite * 0.65))
            partes = []
            for inicio in range(0, len(palavras), janela):
                partes.append(" ".join(palavras[inicio:inicio+janela]))
            arr = model.encode(partes, normalize_embeddings=True)
            vec = arr.mean(axis=0)
            vec = vec / max(np.linalg.norm(vec), 1e-12)
        resultados.append(vec)
    return np.vstack(resultados)


def bow_tfidf(textos):
    from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
    from sklearn.preprocessing import normalize
    limpos = [normalizar_texto(t) for t in textos]
    bow = CountVectorizer(ngram_range=(1,1), min_df=1).fit_transform(limpos)
    tfidf = TfidfVectorizer(ngram_range=(1,2), min_df=1).fit_transform(limpos)
    return normalize(bow), normalize(tfidf)


def cosseno_matriz(emb):
    arr = np.asarray(emb, dtype=float)
    norm = np.linalg.norm(arr, axis=1, keepdims=True)
    arr = arr / np.maximum(norm, 1e-12)
    return arr @ arr.T


def comparar_pares(df, modelo):
    from sklearn.metrics.pairwise import cosine_similarity
    bow, tfidf = bow_tfidf(df.texto.tolist())
    emb = representar(tuple(df.texto.tolist()), modelo)
    pares = [("M003","M017"),("M008","M022"),("M008","M031")]
    linhas=[]
    idx=df.set_index("id").index
    for a,b in pares:
        ia, ib = idx.get_loc(a), idx.get_loc(b)
        linhas.append({
            "Par": f"{a} × {b}",
            "BoW": float(cosine_similarity(bow[ia], bow[ib])[0,0]),
            "TF-IDF": float(cosine_similarity(tfidf[ia], tfidf[ib])[0,0]),
            "Embeddings": float(np.dot(emb[ia], emb[ib])),
        })
    return pd.DataFrame(linhas), emb


def buscar(df, consulta, modelo, top_k=5):
    """Busca por melhor chunk de cada manifestação; retorna uma linha por ID."""
    emb_docs = representar(tuple(df.texto.tolist()), modelo)
    emb_q = representar((consulta,), modelo)[0]
    scores = emb_docs @ emb_q
    ordem = np.argsort(-scores)[:top_k]
    saida = df.iloc[ordem].copy()
    saida["score"] = scores[ordem]
    return saida


def detectar_duplicatas(textos, limiar=0.85, modelo="MiniLM multilíngue"):
    emb = representar(tuple(textos), modelo)
    sim = emb @ emb.T
    pares=[]
    for i in range(len(textos)):
        for j in range(i+1, len(textos)):
            if sim[i,j] >= limiar:
                pares.append({"i":i,"j":j,"similaridade":float(sim[i,j])})
    return sorted(pares, key=lambda x:-x["similaridade"])


def dividir(texto, tamanho=300, overlap=60, estrategia="Recursivo"):
    if not str(texto).strip():
        return []
    if estrategia == "Recursivo":
        from langchain_text_splitters import RecursiveCharacterTextSplitter
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=tamanho,
            chunk_overlap=overlap,
            separators=["\n\n","\n",". ","; ",", "," ",""],
        )
        return splitter.split_text(str(texto))
    # Estratégia comparativa: janela de caracteres com sobreposição fixa.
    partes=[]
    inicio=0
    texto=str(texto)
    passo=max(1,tamanho-overlap)
    while inicio < len(texto):
        parte=texto[inicio:inicio+tamanho].strip()
        if parte: partes.append(parte)
        inicio += passo
    return partes


def cor_score(score):
    if score > .7: return "🟢"
    if score > .5: return "🟡"
    return "🔴"


def projetar(emb, metodo="PCA"):
    import pandas as pd
    if metodo=="PCA":
        from sklearn.decomposition import PCA
        return PCA(n_components=2, random_state=42).fit_transform(emb)
    from sklearn.manifold import TSNE
    perplexidade=min(10, max(2, len(emb)-1))
    return TSNE(n_components=2, random_state=42, init="pca", learning_rate="auto",
                perplexity=perplexidade).fit_transform(emb)


def matriz_heatmap(emb):
    import matplotlib.pyplot as plt
    import seaborn as sns
    sim=cosseno_matriz(emb)
    fig,ax=plt.subplots(figsize=(9,7))
    sns.heatmap(sim, ax=ax, cmap="viridis", vmin=0, vmax=1, xticklabels=False, yticklabels=False)
    ax.set_title("Matriz de similaridade — embeddings")
    ax.set_xlabel("Manifestações")
    ax.set_ylabel("Manifestações")
    fig.tight_layout()
    return fig


def grafico_2d(emb, categorias, metodo="PCA", ids=None):
    import matplotlib.pyplot as plt
    coords=projetar(emb, metodo)
    fig,ax=plt.subplots(figsize=(9,6))
    cats=pd.Series(categorias)
    for cat in cats.unique():
        mask=cats.to_numpy()==cat
        ax.scatter(coords[mask,0],coords[mask,1],label=cat,alpha=.78)
    if ids is not None:
        for x,y,label in zip(coords[:,0],coords[:,1],ids):
            ax.annotate(label,(x,y),fontsize=7,alpha=.75)
    ax.set_title(f"Espaço vetorial — {metodo}")
    ax.legend()
    fig.tight_layout()
    return fig
