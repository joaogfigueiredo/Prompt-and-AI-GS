"""Configurações centrais: caminhos, modelos e limites.

A chave do Gemini NUNCA fica no código. Ela vem da variável de ambiente GOOGLE_API_KEY:
no computador, do arquivo .env (copie .env.example para .env); no Google Colab, dos Secrets.
"""

import os
import re
import time
import warnings
from pathlib import Path

from langchain_core.embeddings import Embeddings

# Alguns modelos novos do Gemini ignoram a temperatura e avisam; o aviso só polui a saída.
warnings.filterwarnings("ignore", message=".*fixed sampling defaults.*")

try:
    from dotenv import load_dotenv
except ImportError:  # no Colab o .env não é usado
    load_dotenv = None

RAIZ = Path(__file__).resolve().parents[1]
if load_dotenv:
    load_dotenv(RAIZ / ".env")

PASTA_DOCUMENTOS = RAIZ / "2_base_documental"
PASTA_CHROMA = RAIZ / "chroma_db"
ARQ_PERGUNTAS = RAIZ / "avaliacao" / "perguntas_ground_truth.json"
ARQ_RAGAS = RAIZ / "3_avaliacao_ragas.json"

# O Google troca os nomes dos modelos com frequência. Usamos o primeiro da lista que a conta enxerga;
# para forçar um nome, defina GEMINI_MODELO_LLM / GEMINI_MODELO_EMBEDDING.
PREFERENCIA_LLM = ("models/gemini-3.5-flash-lite", "models/gemini-2.5-flash-lite",
                   "models/gemini-2.5-flash", "models/gemini-3.5-flash")
PREFERENCIA_EMBEDDING = ("models/gemini-embedding-001", "models/gemini-embedding-2")

K_RECUPERACAO = int(os.getenv("K_RECUPERACAO", "4"))                       # trechos buscados por pergunta
LIMIAR_RELEVANCIA = float(os.getenv("LIMIAR_RELEVANCIA", "0.65"))          # abaixo disso: "não sei" (calibrado: seção 3)
MAX_CHAMADAS_FERRAMENTA = int(os.getenv("MAX_CHAMADAS_FERRAMENTA", "4"))   # limite do agente (o enunciado pede no máximo 6)

COLECAO_PADRAO = "docs_por_artigo"  # a estratégia de chunking escolhida (ver README)

_modelos_da_conta: set[str] | None = None


def tem_chave() -> bool:
    return bool(os.getenv("GOOGLE_API_KEY"))


def exigir_chave() -> None:
    if not tem_chave():
        raise RuntimeError(
            "GOOGLE_API_KEY não encontrada. No computador, copie .env.example para .env e cole sua chave do Gemini. "
            "No Google Colab, crie o Secret GOOGLE_API_KEY e ative 'Notebook access'."
        )


def _listar_modelos_da_conta() -> set[str]:
    global _modelos_da_conta
    if _modelos_da_conta is None:
        try:
            from google import genai

            _modelos_da_conta = {m.name for m in genai.Client().models.list()}
        except Exception:
            _modelos_da_conta = set()
    return _modelos_da_conta


def _escolher(variavel: str, preferencia: tuple[str, ...]) -> str:
    manual = os.getenv(variavel)
    if manual:
        return manual
    disponiveis = _listar_modelos_da_conta()
    return next((m for m in preferencia if m in disponiveis), preferencia[0])


def modelo_llm() -> str:
    return _escolher("GEMINI_MODELO_LLM", PREFERENCIA_LLM)


def modelo_embedding() -> str:
    return _escolher("GEMINI_MODELO_EMBEDDING", PREFERENCIA_EMBEDDING)


class EmbeddingsResilientes(Embeddings):
    """Repete a chamada quando o plano gratuito responde 429 (limite de requisições por minuto)."""

    def __init__(self, interno: Embeddings, tentativas: int = 6) -> None:
        self.interno, self.tentativas = interno, tentativas

    def _repetir(self, funcao, argumento):
        for tentativa in range(1, self.tentativas + 1):
            try:
                return funcao(argumento)
            except Exception as erro:
                texto = str(erro)
                limite_de_uso = "429" in texto or "RESOURCE_EXHAUSTED" in texto
                if not limite_de_uso or tentativa == self.tentativas:
                    raise
                sugerido = re.search(r"retry in ([\d.]+)s", texto)
                espera = float(sugerido.group(1)) if sugerido else 15.0 * tentativa
                time.sleep(min(espera + 1.0, 65.0))

    def embed_documents(self, textos: list[str]) -> list[list[float]]:
        return self._repetir(self.interno.embed_documents, textos)

    def embed_query(self, texto: str) -> list[float]:
        return self._repetir(self.interno.embed_query, texto)


def criar_llm():
    """Modelo de linguagem (Gemini). temperature=0 para respostas estáveis."""
    exigir_chave()
    from langchain_google_genai import ChatGoogleGenerativeAI

    return ChatGoogleGenerativeAI(model=modelo_llm(), temperature=0)


def criar_embeddings():
    """Modelo de embeddings (Gemini), com repetição automática em caso de limite de uso."""
    exigir_chave()
    from langchain_google_genai import GoogleGenerativeAIEmbeddings

    return EmbeddingsResilientes(GoogleGenerativeAIEmbeddings(model=modelo_embedding()))
