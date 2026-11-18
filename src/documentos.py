"""Carrega a base documental e divide os documentos em trechos (chunks).

Duas estratégias de chunking são comparadas no notebook:

  A) fixo        : cortes a cada ~500 caracteres, sem olhar a estrutura do documento.
  B) por_artigo  : um trecho por artigo ("Art. 1º ..."), mantendo o artigo inteiro junto.
                   Artigos muito longos são subdivididos.

Em ambas, o início de cada trecho recebe o doc_id (e, na B, o artigo), para que a
citação viaje junto com o texto.
"""

import re
from dataclasses import dataclass
from pathlib import Path

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter


@dataclass(frozen=True)
class DocumentoFonte:
    doc_id: str
    titulo: str
    data_publicacao: str
    tema: str
    status: str
    fonte_url: str
    texto: str


def _ler_front_matter(conteudo: str) -> tuple[dict[str, str], str]:
    """Separa o cabeçalho (entre ---) do corpo do documento."""
    partes = conteudo.split("---", 2)
    if len(partes) < 3:
        raise ValueError("Documento sem front matter (--- ... ---).")
    meta = {}
    for linha in partes[1].strip().splitlines():
        if ":" in linha:
            chave, valor = linha.split(":", 1)
            meta[chave.strip()] = valor.strip()
    return meta, partes[2].strip()


def carregar_documentos(pasta: Path) -> list[DocumentoFonte]:
    docs = []
    for arquivo in sorted(Path(pasta).glob("doc_*.md")):
        meta, corpo = _ler_front_matter(arquivo.read_text(encoding="utf-8"))
        docs.append(DocumentoFonte(
            doc_id=meta["doc_id"], titulo=meta["titulo"], data_publicacao=meta["data_publicacao"],
            tema=meta["tema"], status=meta["status"], fonte_url=meta["fonte_url"], texto=corpo,
        ))
    if not docs:
        raise FileNotFoundError(f"Nenhum doc_*.md em {pasta}")
    return docs


def _metadados(doc: DocumentoFonte, estrategia: str, indice: int, artigo: str) -> dict:
    # O Chroma só aceita valores simples (str, int, float, bool).
    return {
        "doc_id": doc.doc_id, "titulo": doc.titulo, "data_publicacao": doc.data_publicacao, "tema": doc.tema,
        "status": doc.status, "fonte_url": doc.fonte_url, "artigo": artigo,
        "estrategia": estrategia, "chunk_id": f"{doc.doc_id}-{estrategia}-{indice:03d}",
    }


_ARTIGO_NO_TEXTO = re.compile(r"Art\. (\d+[ºo]?)")


def _artigos_do_trecho(doc: DocumentoFonte, inicio: int, trecho: str) -> str:
    """Artigo(s) que o trecho contém (ou o último artigo antes dele), só para a citação.

    O TEXTO do chunk fixo continua sem rótulo, para a comparação com o chunking por artigo ser justa;
    este valor vai apenas para o metadado `artigo`, usado na citação [doc_id, artigo].
    """
    achados = list(dict.fromkeys(f"Art. {n}" for n in _ARTIGO_NO_TEXTO.findall(trecho)))
    if not achados:
        anteriores = _ARTIGO_NO_TEXTO.findall(doc.texto[:inicio])
        achados = [f"Art. {anteriores[-1]}"] if anteriores else ["Ementa"]
    return ", ".join(achados)


def chunk_fixo(doc: DocumentoFonte, tamanho: int = 500, sobreposicao: int = 50) -> list[Document]:
    """Estratégia A: tamanho fixo, ignora a estrutura do documento."""
    divisor = RecursiveCharacterTextSplitter(chunk_size=tamanho, chunk_overlap=sobreposicao, add_start_index=True)
    pedacos = divisor.create_documents([doc.texto])
    return [Document(page_content=f"[{doc.doc_id}] {p.page_content}",
                     metadata=_metadados(doc, "fixo", i, _artigos_do_trecho(doc, p.metadata["start_index"], p.page_content)))
            for i, p in enumerate(pedacos)]


_INICIO_ARTIGO = re.compile(r"(?m)^(?=Art\. \d)")
_NUMERO_ARTIGO = re.compile(r"^Art\. (\d+[ºo]?(?:-[A-Z])?)")


def chunk_por_artigo(doc: DocumentoFonte, max_chars: int = 2000) -> list[Document]:
    """Estratégia B: um trecho por artigo, com o rótulo [doc_id | Art. N] no início."""
    blocos = [b.strip() for b in _INICIO_ARTIGO.split(doc.texto) if b.strip()]
    divisor = RecursiveCharacterTextSplitter(chunk_size=max_chars, chunk_overlap=100)
    saida: list[Document] = []
    for bloco in blocos:
        achado = _NUMERO_ARTIGO.match(bloco)
        artigo = f"Art. {achado.group(1)}" if achado else "Ementa"
        partes = [bloco] if len(bloco) <= max_chars else divisor.split_text(bloco)
        for j, parte in enumerate(partes):
            sufixo = "" if j == 0 else " (continuação)"
            rotulo = f"[{doc.doc_id} | {artigo}{sufixo}] "
            saida.append(Document(page_content=rotulo + parte,
                                  metadata=_metadados(doc, "artigo", len(saida), artigo)))
    return saida


ESTRATEGIAS = {"fixo": chunk_fixo, "artigo": chunk_por_artigo}


def gerar_chunks(docs: list[DocumentoFonte], estrategia: str) -> list[Document]:
    funcao = ESTRATEGIAS[estrategia]
    return [c for d in docs for c in funcao(d)]
