"""Embeddings e índice vetorial (ChromaDB), com metadados.

Também traz um embedding local (TF-IDF) usado nos testes e na comparação de chunking
"offline", sem chave de API. A produção usa o embedding do Gemini.
"""

from pathlib import Path

from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings

from . import config
from .documentos import carregar_documentos, gerar_chunks


class EmbeddingTFIDF(Embeddings):
    """Embedding lexical local (TF-IDF). Só para testes e comparações sem internet."""

    def __init__(self, textos_de_ajuste: list[str]) -> None:
        from sklearn.feature_extraction.text import TfidfVectorizer

        self._vetorizador = TfidfVectorizer(lowercase=True, strip_accents="unicode", ngram_range=(1, 2), sublinear_tf=True)
        self._vetorizador.fit(textos_de_ajuste)

    def embed_documents(self, textos: list[str]) -> list[list[float]]:
        return self._vetorizador.transform(textos).toarray().tolist()

    def embed_query(self, texto: str) -> list[float]:
        return self.embed_documents([texto])[0]


def criar_indice(chunks: list[Document], embeddings: Embeddings, colecao: str,
                 pasta: Path | None = None, recriar: bool = True):
    """Cria (ou recria) uma coleção no Chroma. pasta=None usa memória (testes)."""
    from langchain_chroma import Chroma

    argumentos = dict(collection_name=colecao, embedding_function=embeddings,
                      collection_metadata={"hnsw:space": "cosine"})
    if pasta is not None:
        argumentos["persist_directory"] = str(pasta)
    indice = Chroma(**argumentos)
    if recriar:
        indice.reset_collection()
    indice.add_documents(chunks, ids=[c.metadata["chunk_id"] for c in chunks])
    return indice


def abrir_indice(embeddings: Embeddings, colecao: str, pasta: Path):
    from langchain_chroma import Chroma

    return Chroma(collection_name=colecao, embedding_function=embeddings, persist_directory=str(pasta),
                  collection_metadata={"hnsw:space": "cosine"})


def construir_indices_gemini() -> dict:
    """Gera os chunks das duas estratégias e grava os dois índices em chroma_db/ (precisa da chave)."""
    docs = carregar_documentos(config.PASTA_DOCUMENTOS)
    embeddings = config.criar_embeddings()
    indices = {}
    for estrategia, colecao in (("fixo", "lei_fixo"), ("artigo", "lei_por_artigo")):
        indices[estrategia] = criar_indice(gerar_chunks(docs, estrategia), embeddings, colecao, config.PASTA_CHROMA)
    return indices


if __name__ == "__main__":
    construir_indices_gemini()
    print("Índices criados em", config.PASTA_CHROMA)
