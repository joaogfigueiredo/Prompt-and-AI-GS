"""Interface Gradio: conversa com histórico e painel de fontes.

Rode com:  python -m src.app_gradio     (precisa da chave do Gemini no .env e do índice criado)
"""

import gradio as gr

from . import config
from .agente import AgenteCidadao

AVISO = ("**Citizen Assist AI** é um protótipo educacional e **não é canal oficial**. "
         "Os documentos da base são simulados (cidade fictícia de Vila Serena). "
         "Não digite CPF, endereço ou outros dados pessoais. Confirme sempre na distribuidora ou na ANEEL.")
EXEMPLOS = [
    "Quem pode pedir a Tarifa Social no Balcão de Vila Serena?",
    "Qual o desconto para quem consome até 30 kWh?",
    "Consumo 100 kWh e tarifa de R$ 0,90 por kWh: quanto pago com a Tarifa Social?",
]


def formatar_fontes(fontes: list[dict], chamadas: list[dict]) -> str:
    if not fontes and not chamadas:
        return "**Fontes:** nenhuma para esta resposta."
    linhas = ["**Fontes desta resposta**"]
    for f in fontes:
        origem = f"[{f['url']}]({f['url']})" if str(f["url"]).startswith("http") else f["url"]
        linhas.append(f"- `{f['doc_id']}`, {f['artigo']} ({f['status']}, {f['data']}): {origem}")
    if chamadas:
        linhas.append("\n**Ferramentas usadas:** " + ", ".join(c["ferramenta"] for c in chamadas))
    return "\n".join(linhas)


def criar_app(agente: AgenteCidadao) -> gr.Blocks:
    def enviar(mensagem: str, historico: list[dict]):
        if not mensagem or not mensagem.strip():
            return "", historico, "**Fontes:** digite uma pergunta."
        pares = [(historico[i]["content"], historico[i + 1]["content"])
                 for i in range(0, len(historico) - 1, 2)]
        resultado = agente.responder(mensagem, pares)
        texto = resultado.texto
        if resultado.alertas:
            texto += "\n\n_Proteções ativadas: " + "; ".join(resultado.alertas) + "_"
        historico = historico + [{"role": "user", "content": mensagem}, {"role": "assistant", "content": texto}]
        return "", historico, formatar_fontes(resultado.fontes, resultado.chamadas)

    with gr.Blocks(title="Citizen Assist AI") as app:
        gr.Markdown("# Citizen Assist AI\n" + AVISO)
        conversa = gr.Chatbot(label="Conversa", height=420)
        fontes = gr.Markdown("**Fontes:** aparecem aqui depois da resposta.")
        caixa = gr.Textbox(label="Sua pergunta", placeholder="Ex.: Quem tem direito à Tarifa Social?")
        with gr.Row():
            botao = gr.Button("Enviar", variant="primary")
            limpar = gr.Button("Limpar conversa")
        gr.Examples(EXEMPLOS, inputs=caixa)
        botao.click(enviar, [caixa, conversa], [caixa, conversa, fontes])
        caixa.submit(enviar, [caixa, conversa], [caixa, conversa, fontes])
        limpar.click(lambda: ([], "**Fontes:** aparecem aqui depois da resposta."), None, [conversa, fontes])
    return app


def main() -> None:
    from .indice import abrir_indice
    from .rag import ConsultaRAG

    llm = config.criar_llm()
    indice = abrir_indice(config.criar_embeddings(), config.COLECAO_PADRAO, config.PASTA_CHROMA)
    criar_app(AgenteCidadao(ConsultaRAG(indice, llm), llm)).launch()


if __name__ == "__main__":
    main()
