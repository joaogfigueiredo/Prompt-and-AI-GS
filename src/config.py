"""Configurações centrais: caminhos, nomes de modelos e limites.

A chave do Gemini NUNCA fica no código. Ela vem da variável de ambiente
GOOGLE_API_KEY, lida do arquivo .env (copie .env.example para .env e preencha).
"""

import os
from pathlib import Path

from dotenv import load_dotenv

RAIZ = Path(__file__).resolve().parents[1]
load_dotenv(RAIZ / ".env")

PASTA_DOCUMENTOS = RAIZ / "base_documental"
PASTA_CHROMA = RAIZ / "chroma_db"
ARQ_PERGUNTAS = RAIZ / "avaliacao" / "perguntas_ground_truth.json"
ARQ_RAGAS = RAIZ / "avaliacao_ragas.json"

# Modelos do Gemini. Se o Google trocar os nomes, mude no .env sem mexer no código
# (rode `python scripts/listar_modelos.py` para ver os disponíveis na sua conta).
MODELO_LLM = os.getenv("GEMINI_MODELO_LLM", "gemini-2.5-flash")
MODELO_EMBEDDING = os.getenv("GEMINI_MODELO_EMBEDDING", "models/gemini-embedding-001")

K_RECUPERACAO = int(os.getenv("K_RECUPERACAO", "4"))          # trechos buscados por pergunta
LIMIAR_RELEVANCIA = float(os.getenv("LIMIAR_RELEVANCIA", "0.30"))  # abaixo disso: "não sei"
MAX_CHAMADAS_FERRAMENTA = int(os.getenv("MAX_CHAMADAS_FERRAMENTA", "4"))  # limite do agente

COLECAO_PADRAO = "lei_por_artigo"  # a estratégia de chunking escolhida (ver README)


def tem_chave() -> bool:
    return bool(os.getenv("GOOGLE_API_KEY"))


def exigir_chave() -> None:
    if not tem_chave():
        raise RuntimeError(
            "GOOGLE_API_KEY não encontrada. Copie .env.example para .env e cole sua chave do Gemini. "
            "No Google Colab, use Secrets e defina os.environ['GOOGLE_API_KEY']."
        )


def criar_llm():
    """Modelo de linguagem (Gemini). temperature=0 para respostas estáveis."""
    exigir_chave()
    from langchain_google_genai import ChatGoogleGenerativeAI

    return ChatGoogleGenerativeAI(model=MODELO_LLM, temperature=0)


def criar_embeddings():
    """Modelo de embeddings (Gemini)."""
    exigir_chave()
    from langchain_google_genai import GoogleGenerativeAIEmbeddings

    return GoogleGenerativeAIEmbeddings(model=MODELO_EMBEDDING)
