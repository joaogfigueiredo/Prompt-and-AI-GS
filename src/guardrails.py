"""Os 3 guardrails do assistente.

1. ESCOPO            : só responde sobre Tarifa Social, energia e temas ligados.
2. PROMPT INJECTION  : trechos dos documentos que tentam dar ordens ao modelo são descartados.
3. LGPD              : CPF, NIS, e-mail, telefone e CEP são mascarados antes de chegar ao modelo
                       ou ao histórico, e pedidos de dados de terceiros são recusados.
"""

import re
import unicodedata
from dataclasses import dataclass, field

from langchain_core.documents import Document


def _normalizar(texto: str) -> str:
    sem_acento = "".join(c for c in unicodedata.normalize("NFKD", texto) if not unicodedata.combining(c))
    return re.sub(r"\s+", " ", sem_acento.lower()).strip()


# ----------------------------------------------------------------------------- 1. escopo
_PALAVRAS_ESCOPO = re.compile(
    r"tarifa|energia|eletric|\bluz\b|kwh|quilowatt|cadunico|cadastro unico|\bbpc\b|"
    r"prestacao continuada|baixa renda|distribuidora|aneel|indigena|quilombola|desconto|"
    r"\bfatura|conta de|consumo|medidor|\bcde\b|subclasse|beneficio|salario minimo|lei 1[0-9]"
)
MENSAGEM_FORA_DO_ESCOPO = (
    "Só posso ajudar com a Tarifa Social de Energia Elétrica e temas ligados (direitos, "
    "documentos, descontos e consumo de energia). Para outros assuntos, procure o canal oficial."
)


def dentro_do_escopo(pergunta: str) -> bool:
    return bool(_PALAVRAS_ESCOPO.search(_normalizar(pergunta)))


# ----------------------------------------------------------------------------- 2. injection
_PADROES_INJECTION = [re.compile(p) for p in (
    r"ignor\w* (todas |todos |as |os |suas |seus )?(as )?(instrucoes|regras|ordens|diretrizes|anteriores)",
    r"desconsider\w* (as |os |todas |todos )?(instrucoes|regras|ordens|anteriores|documentos)",
    r"esquec\w* (tudo|as instrucoes|as regras)",
    r"ignore (all |any )?(previous |prior |above )?(instructions|rules)",
    r"disregard (the |all )?(previous|above|prior)",
    r"you are now", r"voce agora e", r"a partir de agora (voce|responda|aja)",
    r"system prompt", r"prompt do sistema", r"prompt de sistema",
    r"revele (o |seu |sua )?(prompt|instrucoes|chave|segredo)",
    r"novas? instrucoes", r"aja como", r"finja (que|ser)",
    r"nao (cite|mencione) (as )?fontes",
)]


def texto_suspeito_de_injection(texto: str) -> bool:
    n = _normalizar(texto)
    return any(p.search(n) for p in _PADROES_INJECTION)


def filtrar_trechos_suspeitos(docs: list[Document]) -> tuple[list[Document], list[Document]]:
    """Separa os trechos seguros dos que parecem conter ordens ao modelo."""
    seguros, removidos = [], []
    for d in docs:
        (removidos if texto_suspeito_de_injection(d.page_content) else seguros).append(d)
    return seguros, removidos


# ----------------------------------------------------------------------------- 3. LGPD
_PADROES_PII = (
    (re.compile(r"\b\d{3}\.?\d{3}\.?\d{3}-?\d{2}\b"), "[CPF/NIS removido]"),
    (re.compile(r"[\w.+-]+@[\w-]+\.[\w.]+"), "[e-mail removido]"),
    (re.compile(r"(?<!\d)(?:\+?55\s?)?\(?\d{2}\)?\s?9?\d{4}-?\d{4}(?!\d)"), "[telefone removido]"),
    (re.compile(r"\b\d{5}-\d{3}\b"), "[CEP removido]"),
)
_PEDIDO_DADOS_DE_TERCEIROS = re.compile(
    r"(descobr|descubr|consult|busc|ach|pesquis|me (de|passe|diga))\w* .{0,40}"
    r"(cpf|endereco|telefone|renda|nis|dados) (de|do|da) (meu |minha |um |uma |o |a )?"
    r"(pessoa|fulano|vizinho|vizinha|terceiro|cliente|alguem|funcionario)"
)
MENSAGEM_DADOS_DE_TERCEIROS = (
    "Não consulto nem forneço dados pessoais de terceiros. Posso explicar as regras da Tarifa Social."
)


def mascarar_dados_pessoais(texto: str) -> str:
    for padrao, troca in _PADROES_PII:
        texto = padrao.sub(troca, texto)
    return texto


# ----------------------------------------------------------------------------- junção
@dataclass
class ResultadoGuardrail:
    permitido: bool
    pergunta_segura: str
    motivo: str = ""
    mensagem: str = ""
    alertas: list[str] = field(default_factory=list)


def aplicar_guardrails_entrada(pergunta: str) -> ResultadoGuardrail:
    """Roda os guardrails na pergunta do usuário, nesta ordem: LGPD, injection, escopo."""
    segura = mascarar_dados_pessoais(pergunta.strip())
    alertas = []
    if segura != pergunta.strip():
        alertas.append("lgpd: dados pessoais mascarados antes de chegar ao modelo")

    if _PEDIDO_DADOS_DE_TERCEIROS.search(_normalizar(pergunta)):
        return ResultadoGuardrail(False, segura, "lgpd_terceiros", MENSAGEM_DADOS_DE_TERCEIROS, alertas)

    if texto_suspeito_de_injection(pergunta):
        return ResultadoGuardrail(False, segura, "prompt_injection",
                                  "Não posso seguir instruções que tentam mudar meu funcionamento. "
                                  "Pergunte sobre a Tarifa Social de Energia.", alertas)

    if not dentro_do_escopo(pergunta):
        return ResultadoGuardrail(False, segura, "fora_do_escopo", MENSAGEM_FORA_DO_ESCOPO, alertas)

    return ResultadoGuardrail(True, segura, alertas=alertas)
