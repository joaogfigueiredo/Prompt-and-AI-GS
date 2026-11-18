"""Lista os modelos do Gemini disponíveis na sua conta (útil se o Google trocar os nomes).

Rode com:  python scripts/listar_modelos.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src import config  # noqa: E402

config.exigir_chave()

from google import genai  # noqa: E402  (vem junto com langchain-google-genai)

cliente = genai.Client()
for modelo in cliente.models.list():
    acoes = ", ".join(getattr(modelo, "supported_actions", None) or [])
    print(f"{modelo.name}  [{acoes}]")
