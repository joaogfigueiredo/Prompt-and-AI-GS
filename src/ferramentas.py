"""Ferramentas do agente.

1. buscar_base_documental            : consulta o RAG (trechos dos documentos do Balcão).
2. calcular_desconto_tarifa_social   : CALCULA quanto a família paga pela energia, com erro controlado.

Erro controlado = a ferramenta nunca "estoura": devolve um texto "ERRO_CONTROLADO: ..." que o
modelo lê e usa para pedir o dado certo ao usuário, em vez de o programa quebrar.
"""

import json

from langchain_core.documents import Document
from langchain_core.tools import tool

from .rag import ConsultaRAG, formatar_contexto

# (limite superior acumulado em kWh, desconto). Acima do último limite: 0% de desconto.
REGRAS_DE_DESCONTO = {
    "vigente": ((80.0, 1.00),),                                    # Portaria 03/2026, art. 2º
    "cartilha_2025": ((30.0, 0.65), (100.0, 0.40), (220.0, 0.10)),  # Cartilha 2025, art. 3º (desatualizada)
}
CONSUMO_MAXIMO_KWH = 5000.0
TARIFA_MAXIMA = 10.0


def calcular_desconto(consumo_kwh: float, tarifa_reais_por_kwh: float, regra: str = "vigente") -> dict:
    """Calcula o valor da energia com a Tarifa Social. Levanta ValueError se a entrada for inválida."""
    if regra not in REGRAS_DE_DESCONTO:
        raise ValueError(f"regra '{regra}' desconhecida. Use: {', '.join(REGRAS_DE_DESCONTO)}.")
    if not 0 <= consumo_kwh <= CONSUMO_MAXIMO_KWH:
        raise ValueError(f"consumo_kwh deve estar entre 0 e {CONSUMO_MAXIMO_KWH:.0f}. Recebido: {consumo_kwh}.")
    if not 0 < tarifa_reais_por_kwh <= TARIFA_MAXIMA:
        raise ValueError(f"tarifa_reais_por_kwh deve estar entre 0 e {TARIFA_MAXIMA:.0f} reais. Recebido: {tarifa_reais_por_kwh}.")

    faixas, anterior, total_sem, total_com = [], 0.0, 0.0, 0.0
    limites = list(REGRAS_DE_DESCONTO[regra]) + [(float("inf"), 0.0)]
    for limite, desconto in limites:
        kwh = max(0.0, min(consumo_kwh, limite) - anterior)
        if kwh > 0:
            sem, com = kwh * tarifa_reais_por_kwh, kwh * tarifa_reais_por_kwh * (1 - desconto)
            faixas.append({"de_kwh": anterior, "ate_kwh": min(consumo_kwh, limite), "kwh": kwh,
                           "desconto": desconto, "valor_a_pagar": round(com, 2)})
            total_sem, total_com = total_sem + sem, total_com + com
        anterior = limite
    return {
        "regra": regra, "consumo_kwh": consumo_kwh, "tarifa_reais_por_kwh": tarifa_reais_por_kwh,
        "valor_sem_desconto": round(total_sem, 2), "valor_a_pagar": round(total_com, 2),
        "economia": round(total_sem - total_com, 2), "faixas": faixas,
        "aviso": "Só a energia, sem tributos e outros encargos da fatura.",
    }


def criar_ferramentas(consulta: ConsultaRAG, registro_docs: list[Document]):
    """Cria as ferramentas ligadas a uma consulta RAG. Os trechos usados vão para registro_docs."""

    @tool
    def buscar_base_documental(pergunta: str) -> str:
        """Busca nos documentos do Balcão da Tarifa Social (Cartilha 2025 e Portaria 03/2026) os trechos que respondem à pergunta.
        Use SEMPRE antes de afirmar qualquer regra da Tarifa Social. Devolve trechos com doc_id, artigo e status."""
        # recuperar_seguro já descarta trechos com prompt injection: eles nunca chegam ao modelo.
        docs, _removidos = consulta.recuperar_seguro(pergunta)
        if not docs:
            return "SEM_RESULTADOS: nenhum trecho relevante nos documentos. Responda NAO_SEI."
        registro_docs.extend(docs)
        return formatar_contexto(docs)

    @tool
    def calcular_desconto_tarifa_social(consumo_kwh: float, tarifa_reais_por_kwh: float, regra: str = "vigente") -> str:
        """Calcula quanto uma família paga pela energia com a Tarifa Social.
        consumo_kwh: consumo do mês em kWh. tarifa_reais_por_kwh: preço do kWh em reais.
        regra: 'vigente' (Portaria 03/2026, padrão) ou 'cartilha_2025' (regra antiga, só para comparação)."""
        try:
            return json.dumps(calcular_desconto(consumo_kwh, tarifa_reais_por_kwh, regra), ensure_ascii=False)
        except ValueError as erro:
            return f"ERRO_CONTROLADO: {erro}"

    return [buscar_base_documental, calcular_desconto_tarifa_social]
