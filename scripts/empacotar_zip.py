"""Monta o ZIP de entrega com os 7 itens e confere o que o enunciado exige.

  python scripts/empacotar_zip.py NOME_DO_GRUPO

Recusa empacotar se houver pendência (link ou texto a preencher, ground truth vazio, erro no notebook)
ou se a chave da API aparecer em qualquer arquivo.
"""

import json
import os
import re
import sys
import zipfile
from pathlib import Path

import nbformat

RAIZ = Path(__file__).resolve().parents[1]
TURMA = "1CCR"
ENTREGA = RAIZ / "entrega"

ARQUIVOS = {
    "1_Notebook_GS2026_2.ipynb": RAIZ / "1_Notebook_GS2026_2.ipynb",
    "3_avaliacao_ragas.json": RAIZ / "3_avaliacao_ragas.json",
    "4_README.md": RAIZ / "README.md",
    "5_certificacao.txt": ENTREGA / "5_certificacao.txt",
    "6_Chat_Log.txt": ENTREGA / "6_Chat_Log.txt",
    "7_demo_video_url.txt": ENTREGA / "7_demo_video_url.txt",
}
PASTA_DOCS = RAIZ / "2_base_documental"
PENDENCIAS = ("[COLE AQUI", "RESULTADO_CHUNKING", "RESULTADOS_RAGAS", "a ser preenchida")


def chave_da_api() -> str:
    try:
        from dotenv import dotenv_values

        valor = dotenv_values(RAIZ / ".env").get("GOOGLE_API_KEY")
    except ImportError:
        valor = None
    return (valor or os.getenv("GOOGLE_API_KEY") or "").strip()


def main() -> None:
    if len(sys.argv) < 2:
        sys.exit("Uso: python scripts/empacotar_zip.py NOME_DO_GRUPO")
    grupo = re.sub(r"[^A-Za-z0-9_-]", "_", sys.argv[1])
    problemas = []

    faltando = [nome for nome, caminho in ARQUIVOS.items() if not caminho.exists()]
    docs = sorted(PASTA_DOCS.glob("doc_*.md"))
    if faltando:
        problemas.append(f"arquivos que não existem: {', '.join(faltando)}")
    if len(docs) < 2:
        problemas.append("a base documental precisa de 2 documentos ou mais")
    if problemas:
        sys.exit("NÃO EMPACOTADO:\n- " + "\n- ".join(problemas))

    for nome, caminho in ARQUIVOS.items():
        texto = caminho.read_text(encoding="utf-8")
        for marca in PENDENCIAS:
            if marca in texto:
                problemas.append(f"{nome} ainda tem pendência: '{marca}'")

    notebook = nbformat.read(ARQUIVOS["1_Notebook_GS2026_2.ipynb"], as_version=4)
    for i, celula in enumerate(notebook.cells):
        if celula.cell_type == "code" and any(o.get("output_type") == "error" for o in celula.get("outputs", [])):
            problemas.append(f"o notebook tem erro de execução na célula {i}")
        if celula.cell_type == "code" and any("AVALIAÇÃO NÃO EXECUTADA" in str(o.get("text", "")) for o in celula.get("outputs", [])):
            problemas.append("o RAGAS não foi executado no notebook (célula da seção 7)")

    perguntas = json.loads((RAIZ / "avaliacao" / "perguntas_ground_truth.json").read_text(encoding="utf-8"))
    if any(not p["ground_truth"].strip() for p in perguntas):
        problemas.append("há pergunta sem ground truth")

    chave = chave_da_api()
    candidatos = list(ARQUIVOS.values()) + docs
    if chave:
        for caminho in candidatos:
            if chave in caminho.read_text(encoding="utf-8", errors="ignore"):
                problemas.append(f"A CHAVE DA API APARECE em {caminho.name}. Remova antes de entregar.")
    for caminho in candidatos:
        if re.search(r"AIza[0-9A-Za-z_-]{30,}", caminho.read_text(encoding="utf-8", errors="ignore")):
            problemas.append(f"{caminho.name} contém algo que parece uma chave do Google")

    if problemas:
        sys.exit("NÃO EMPACOTADO:\n- " + "\n- ".join(problemas))

    destino = ENTREGA / f"GS2026_2_{TURMA}_{grupo}.zip"
    with zipfile.ZipFile(destino, "w", zipfile.ZIP_DEFLATED) as zf:
        for nome, caminho in ARQUIVOS.items():
            zf.write(caminho, nome)
        for doc in docs:
            zf.write(doc, f"2_base_documental/{doc.name}")
    with zipfile.ZipFile(destino) as zf:
        topo = sorted({n.split("/")[0] for n in zf.namelist()})
    print("ZIP criado:", destino)
    print("Itens na raiz:", ", ".join(topo))
    print("(nenhuma pendência encontrada e a chave não aparece em nenhum arquivo)")


if __name__ == "__main__":
    main()
