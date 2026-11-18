# Prompt and AI: Citizen Assist AI (RAG + agente + guardrails)

Entrega da Global Solution 2026.2 (FIAP, 1º CC). Um assistente que responde sobre a **Tarifa Social de Energia Elétrica** a partir de documentos oficiais, cita a fonte, diz "não sei" quando não tem base, usa ferramentas (busca e cálculo), tem 3 guardrails e é avaliado com RAGAS.

## A armadilha da base documental

Dois documentos reais, publicados em fontes oficiais, que **se contradizem** porque o segundo substituiu parte do primeiro:

| doc_id | O que é | Status | Fonte |
|---|---|---|---|
| `LEI_12212_2010` | Lei 12.212/2010, texto original | **Desatualizado em parte** | [Senado](https://legis.senado.leg.br/norma/575854/publicacao/15749932) |
| `LEI_15235_2025` | Lei 15.235/2025, que altera a anterior | **Vigente** | [Planalto](https://www.planalto.gov.br/ccivil_03/_ato2023-2026/2025/lei/l15235.htm) |

| Assunto | 2010 (desatualizado) | 2025 (vigente) |
|---|---|---|
| Desconto até 30 kWh | 65% | 100% (até 80 kWh) |
| Faixa de 31 a 100 kWh | 40% | 100% até 80 kWh, 0% depois |
| Faixa de 101 a 220 kWh | 10% | revogada |
| Indígenas e quilombolas: 100% até | 50 kWh | 80 kWh |

Um RAG ingênuo pode responder "65%" porque o trecho antigo também é parecido com a pergunta. O nosso prompt manda usar o documento `VIGENTE`, avisar que a regra antiga mudou e citar os dois.

## Como rodar

### No seu computador

```bash
pip install -r requirements.txt
copy .env.example .env          # e cole sua chave do Gemini em GOOGLE_API_KEY
python -m unittest              # 35 testes, rodam sem internet e sem chave
python scripts/gerar_notebook.py   # gera notebook.ipynb (com a chave, roda também as partes do Gemini)
python -m src.indice            # cria os índices no ChromaDB (chroma_db/)
python -m src.app_gradio        # abre a interface
python -m src.avaliacao         # RAGAS -> avaliacao_ragas.json
```

### No Google Colab (recomendado para o RAGAS)

Alguns computadores bloqueiam as bibliotecas do RAGAS (`pyarrow`/`datasets`); no Windows com controle de aplicativos isso acontece. No Colab funciona:

1. Envie a pasta (ou o ZIP) para o Colab e abra `notebook.ipynb`.
2. Primeira célula: `!pip install -q -r requirements.txt`.
3. Em **Secrets** (ícone de chave), crie `GOOGLE_API_KEY` com a sua chave e, numa célula: `import os; from google.colab import userdata; os.environ["GOOGLE_API_KEY"] = userdata.get("GOOGLE_API_KEY")`.
4. Rode tudo (*Run all*). O notebook cria os índices, roda o agente e o RAGAS e grava `avaliacao_ragas.json`.

> Os nomes dos modelos (`gemini-2.5-flash` e `models/gemini-embedding-001`) podem mudar. Se der erro de modelo, rode `python scripts/listar_modelos.py` e ajuste `GEMINI_MODELO_LLM` e `GEMINI_MODELO_EMBEDDING` no `.env`.

## Estrutura

```
Prompt and AI/
├── README.md
├── notebook.ipynb                 a entrega: roteiro completo, executável
├── requirements.txt
├── .env.example                   modelo do arquivo da chave (o .env NÃO vai para o GitHub nem para o ZIP)
├── avaliacao_ragas.json           (gerado pelo RAGAS)
├── base_documental/
│   ├── doc_01_lei_12212_2010_original.md
│   ├── doc_02_lei_15235_2025.md
│   └── fontes_brutas/             HTML original baixado das fontes oficiais
├── avaliacao/
│   └── perguntas_ground_truth.json    8 perguntas (4 simples + 4 armadilha) com ground truth
├── src/
│   ├── config.py                  caminhos, modelos e limites
│   ├── documentos.py              carregamento e as 2 estratégias de chunking
│   ├── indice.py                  embeddings e ChromaDB com metadados
│   ├── guardrails.py              escopo, prompt injection e LGPD
│   ├── rag.py                     pipeline LCEL com citação e "não sei"
│   ├── ferramentas.py             busca nos documentos e cálculo do desconto
│   ├── agente.py                  agente com limite de chamadas e erro controlado
│   ├── app_gradio.py              interface com histórico e fontes
│   └── avaliacao.py               comparação de chunking e RAGAS
├── scripts/
│   ├── gerar_notebook.py
│   └── listar_modelos.py
└── tests/test_offline.py          35 testes sem internet
```

## Decisões de projeto

**Chunking (2 estratégias comparadas).** A) tamanho fixo de ~500 caracteres; B) um trecho por artigo, com o rótulo `[doc_id | Art. N]`. Escolhida a **B**: cada trecho já traz a citação, incisos e parágrafos ficam juntos (na A, os incisos da regra de 80 kWh são cortados em dois trechos) e, medindo a cobertura de evidências nas 8 perguntas com um embedding local (TF-IDF), a B não perde da A (cobertura de 88% nas duas; a pergunta Q1 falha nas duas porque a busca lexical não liga "quem tem direito" ao art. 2º). A comparação definitiva, com embeddings do Gemini, está no notebook (seção 2b) e a justificativa pela qualidade da resposta vem do RAGAS.

**Metadados no Chroma:** `doc_id`, `titulo`, `data_publicacao`, `status`, `fonte_url`, `artigo`, `estrategia`, `chunk_id`.

**"Não sei":** dois caminhos. (1) Se nenhum trecho passa do `LIMIAR_RELEVANCIA` (0,30), o modelo nem é chamado. (2) O prompt manda responder `NAO_SEI` quando os trechos não bastam. Calibre o limiar com a célula "Calibrando o limiar" do notebook.

**Agente.** Duas ferramentas: `buscar_base_documental` e `calcular_desconto_tarifa_social` (a conta é feita em código, não pelo modelo). **Erro controlado:** entradas inválidas viram `ERRO_CONTROLADO: ...`, que o modelo lê e usa para pedir o dado certo. **Limite:** no máximo `MAX_CHAMADAS_FERRAMENTA` (4) chamadas; ao estourar, o agente é forçado a responder.

**Guardrails.**
1. *Escopo:* só Tarifa Social e temas ligados; o resto é recusado sem chamar o modelo.
2. *Prompt injection na fonte:* trechos recuperados que contêm ordens ao modelo ("ignore as instruções...") são descartados antes do prompt; os documentos entram no prompt dentro de `<documento>` e a regra 1 do prompt diz que são dados, não instruções.
3. *LGPD:* CPF/NIS, e-mail, telefone e CEP são mascarados antes do modelo, do histórico e na saída; pedidos de dados de terceiros são recusados.

## Resultados

### Já medido (sem Gemini)
- 35 testes automatizados passam (chunking, guardrails, ferramentas, RAG e agente com modelo de teste, interface).
- Cobertura de evidências com embedding local: fixo 88% e por artigo 88% (k = 2 e k = 4), armadilha 100% nas duas.

### Medido com Gemini e RAGAS (`avaliacao_ragas.json`, 2026-11-18)

Modelo LLM: `models/gemini-3.5-flash-lite` · Embedding: `models/gemini-embedding-001` · 8 perguntas (4 simples + 4 armadilha).

| Métrica | Simples (4) | Armadilha (4) | Geral |
|---|---|---|---|
| Faithfulness | 0.548 | 0.608 | 0.578 |
| Answer relevancy | 0.783 | 0.841 | 0.803 |
| Context precision | 0.611 | 0.625 | 0.619 |
| Context recall | 0.750 | 1.000 | 0.875 |

**Por que cada métrica ficou como ficou:**

- **Context recall 1.0 nas armadilhas, 0.75 nas simples:** As perguntas de armadilha (Q5–Q8) pedem exatamente os artigos onde a lei mudou — esses trechos têm vocabulário específico e são bem recuperados. A queda nas simples vem de Q3 ("o que fazer ao mudar de residência"), que o guardrail de escopo recusou por entender como fora do tema; resultado: recall 0.0 nessa pergunta puxou a média para baixo.

- **Context precision 0.61–0.63:** O RAG recupera tanto trechos vigentes (Lei 15.235/2025) quanto os desatualizados (Lei 12.212/2010) porque ambos são semanticamente parecidos com as perguntas. Trechos irrelevantes entram no contexto e reduzem a precisão — especialmente nas armadilhas onde a "regra antiga" é deliberadamente parte da base.

- **Faithfulness 0.55–0.61:** Q3 recusada sem contexto (faithfulness 0.0) puxa a média das simples para baixo. Nas armadilhas, o modelo às vezes menciona a regra antiga junto com a nova mesmo quando o prompt pede para usar só a VIGENTE, gerando afirmações não totalmente suportadas pelos trechos recuperados.

- **Answer relevancy 0.80–0.84:** Boa, mas limitada pelo aviso obrigatório "não é canal oficial" que encerra todas as respostas — o RAGAS desconta por ser texto que não responde à pergunta. Nas armadilhas a relevância é mais alta porque a resposta é mais direta (corrigi/negou a premissa errada da pergunta).

**Melhoria concreta:** implementar *reranking por status* — após a busca vetorial, reordenar os trechos recuperados colocando `status=VIGENTE` primeiro no contexto. Isso reduziria a confusão do modelo entre a regra antiga e a nova, aumentando faithfulness nas armadilhas e context precision em geral, sem custo adicional de chamadas à API.

## Limitações

- Só 2 documentos: o assistente não cobre decretos, resoluções da ANEEL nem as regras do CadÚnico.
- A base reproduz trechos da lei; resoluções da ANEEL podem detalhar a aplicação e não estão incluídas.
- O guardrail de escopo é por palavras-chave: perguntas legítimas com vocabulário diferente podem ser recusadas (falso negativo), e esse é o custo de ser previsível e barato.
- O RAGAS usa o próprio Gemini como juiz, então as notas têm ruído.
- Protótipo educacional, **não é canal oficial**.

## Pendências da entrega

- [x] Rodar com a chave: `avaliacao_ragas.json` e a tabela de resultados acima
- [ ] Certificação (confirmar com o professor o que é e anexar ao ZIP)
- [ ] Vídeo de até 3 minutos
- [ ] Gerar o ZIP **sem** o `.env` e sem a pasta `chroma_db/`
