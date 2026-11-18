"""Avaliação: comparação de chunking (sem LLM) e avaliação RAGAS do agente (com Gemini).

  python -m src.avaliacao          # roda o RAGAS e grava avaliacao_ragas.json

O RAGAS depende de bibliotecas (pyarrow/datasets) que podem ser bloqueadas em alguns
computadores. Se acontecer, rode no Google Colab (ver README).
"""

import json
import re
import sys
from datetime import datetime
from pathlib import Path

from . import config


def carregar_perguntas(caminho: Path = config.ARQ_PERGUNTAS) -> list[dict]:
    return json.loads(Path(caminho).read_text(encoding="utf-8"))


def _norm(texto: str) -> str:
    return re.sub(r"\s+", " ", texto.lower()).strip()


# ---------------------------------------------------------------------------- chunking
def cobertura_de_evidencias(indice, perguntas: list[dict], k: int = 4) -> dict:
    """Para cada pergunta, quantas das evidências necessárias aparecem nos k trechos recuperados.

    Uma pergunta está "coberta" quando TODAS as suas evidências estão nos trechos. Mede se a
    estratégia de chunking entrega ao modelo o que ele precisa para responder certo.
    """
    linhas = []
    for p in perguntas:
        achados = indice.similarity_search_with_relevance_scores(p["pergunta"], k=k)
        texto = _norm(" ".join(d.page_content for d, _ in achados))
        presentes = [e for e in p["evidencias"] if _norm(e) in texto]
        linhas.append({"id": p["id"], "tipo": p["tipo"], "evidencias_encontradas": len(presentes),
                       "evidencias_total": len(p["evidencias"]), "coberta": len(presentes) == len(p["evidencias"])})
    n = len(linhas)
    return {
        "k": k, "por_pergunta": linhas,
        "cobertura_geral": sum(l["coberta"] for l in linhas) / n,
        "cobertura_simples": _media(linhas, "simples"),
        "cobertura_armadilha": _media(linhas, "armadilha"),
    }


def _media(linhas: list[dict], tipo: str) -> float:
    grupo = [l for l in linhas if l["tipo"] == tipo]
    return sum(l["coberta"] for l in grupo) / len(grupo) if grupo else float("nan")


# ---------------------------------------------------------------------------- RAGAS
def coletar_respostas(agente, perguntas: list[dict]) -> list[dict]:
    """Roda o agente nas 8 perguntas e junta o que o RAGAS precisa."""
    amostras = []
    for p in perguntas:
        r = agente.responder(p["pergunta"])
        amostras.append({
            "id": p["id"], "tipo": p["tipo"],
            "user_input": p["pergunta"], "response": r.texto,
            "retrieved_contexts": r.contextos or ["(nenhum trecho recuperado)"],
            "reference": p["ground_truth"],
            "ferramentas_usadas": [c["ferramenta"] for c in r.chamadas],
        })
    return amostras


def rodar_ragas(amostras: list[dict], llm, embeddings) -> dict:
    try:
        from ragas import EvaluationDataset, evaluate
        from ragas.metrics import AnswerRelevancy, ContextPrecision, ContextRecall, Faithfulness
    except Exception as erro:  # ImportError ou bloqueio de DLL
        raise RuntimeError(
            "Não foi possível importar o RAGAS neste computador "
            f"({type(erro).__name__}: {erro}). Rode no Google Colab (veja o README)."
        ) from erro

    dataset = EvaluationDataset.from_list([
        {k: a[k] for k in ("user_input", "response", "retrieved_contexts", "reference")} for a in amostras])
    resultado = evaluate(dataset, metrics=[Faithfulness(), AnswerRelevancy(), ContextPrecision(), ContextRecall()],
                         llm=llm, embeddings=embeddings)
    tabela = resultado.to_pandas()
    metricas = [c for c in tabela.columns if c not in ("user_input", "response", "retrieved_contexts", "reference")]

    por_pergunta = []
    for a, (_, linha) in zip(amostras, tabela.iterrows()):
        por_pergunta.append({"id": a["id"], "tipo": a["tipo"], "pergunta": a["user_input"], "resposta": a["response"],
                             "ferramentas_usadas": a["ferramentas_usadas"],
                             "metricas": {m: _num(linha[m]) for m in metricas}})
    medias = {m: _num(tabela[m].mean()) for m in metricas}
    por_tipo = {t: {m: _num(tabela.loc[[a["tipo"] == t for a in amostras], m].mean()) for m in metricas}
                for t in ("simples", "armadilha")}
    return {"gerado_em": datetime.now().isoformat(timespec="seconds"), "modelo_llm": config.MODELO_LLM,
            "modelo_embedding": config.MODELO_EMBEDDING, "n_perguntas": len(amostras),
            "medias": medias, "medias_por_tipo": por_tipo, "por_pergunta": por_pergunta}


def _num(valor) -> float | None:
    try:
        v = float(valor)
        return None if v != v else round(v, 4)  # NaN vira null no JSON
    except (TypeError, ValueError):
        return None


def main() -> None:
    from .agente import AgenteCidadao
    from .indice import abrir_indice
    from .rag import ConsultaRAG

    llm = config.criar_llm()
    embeddings = config.criar_embeddings()
    indice = abrir_indice(embeddings, config.COLECAO_PADRAO, config.PASTA_CHROMA)
    agente = AgenteCidadao(ConsultaRAG(indice, llm), llm)

    amostras = coletar_respostas(agente, carregar_perguntas())
    resultado = rodar_ragas(amostras, llm, embeddings)
    config.ARQ_RAGAS.write_text(json.dumps(resultado, ensure_ascii=False, indent=2), encoding="utf-8")
    print("Médias:", json.dumps(resultado["medias"], ensure_ascii=False))
    print("Salvo em", config.ARQ_RAGAS)


if __name__ == "__main__":
    try:
        main()
    except RuntimeError as erro:
        print(erro, file=sys.stderr)
        sys.exit(2)
