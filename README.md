# Ouvidoria Inteligente — proposta individual

Protótipo acadêmico de triagem semântica de manifestações cidadãs.

## Entregas

1. `analise_comparativa.ipynb` — BoW × TF-IDF × dois modelos de embeddings.
2. `deteccao_duplicatas.ipynb` — matriz, threshold, pares e análise de erros.
3. `chunking_manifestacoes.ipynb` — cinco textos longos, duas configurações e PCA.
4. `app_ouvidoria.py` — quatro abas Streamlit.
5. `RELATORIO.pdf` — síntese em até 5 páginas.

Arquivos de suporte:
- `manifestacoes.json`: 40 manifestações sintéticas.
- `verdade_duplicatas.json`: referência de três pares duplicados (6/40 = 15%).
- `ouvidoria.py`: núcleo reutilizável.
- `resultados/`: artefatos produzidos por análises.

## Instalação

```bash
python -m venv .venv
```

Windows:
```bash
.venv\Scripts\activate
```

```bash
pip install -r requirements-dev.txt
```

## Aplicação

```bash
streamlit run app_ouvidoria.py
```

Na primeira utilização, os modelos de `sentence-transformers` podem ser
baixados. A barra lateral permite selecionar dois modelos multilíngues.

## Notebooks

Abra os três `.ipynb` na raiz do projeto e execute as células em ordem.

## Observação sobre autoria

O corpus e a implementação desta proposta foram escritos especificamente
para este projeto. O arquivo `verdade_duplicatas.json` explicita os pares
usados como referência para avaliação, evitando depender de rótulos ocultos
na aplicação.
