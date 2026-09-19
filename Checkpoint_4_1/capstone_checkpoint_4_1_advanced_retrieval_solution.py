r"""Capstone Checkpoint 4.1 — Advanced Retrieval Implementation (starter).
Jupytext-style cell markers (# %% / # %% [markdown]) — runnable as a
plain script AND openable as cells in VS Code/PyCharm/Jupytext.
"""

# %% [markdown]
# # Capstone Checkpoint 4.1 — Advanced Retrieval Implementation
# **MO-LLM Module 4/Required Capstone Checkpoint (120 minutes)**
#
# ## What this checkpoint is
#
# This checkpoint advances your capstone system from a *single-pass* retriever to an
# **advanced retriever**. You will use the baseline retrieval system you built in
# Checkpoint 2.1 and evaluated in Checkpoint 3.1 to identify where single-pass retrieval
# falls short and apply the Module 4 techniques to fix it:
#
# - **Lab 4.1 — multistep retrieval (query decomposition):** Rewrite a complex
#   question into focused sub-queries, retrieve for each, and merge the results.
# - **Lab 4.2 — graph-based retrieval:** Organize your corpus as a graph of
#   relationships (citations, shared topics, links, same author/entity) and pull in
#   *related* documents a keyword/vector search would miss.
#
# You then **measure the improvement** against your 3.1 baseline. The graded
# deliverable is a **500-750 word written report** (final section). This script is a
# runnable demonstration of both techniques on a tiny sample corpus so you can see
# them work before adapting them to your real system.
#
# **Learning outcomes (Module 4):**
# 1. Identify common retrieval failures — e.g., complex multistep questions and too
#    much loosely-related context degrading the answer.
# 2. Explain *why* those failures occur with single-pass retrieval.
# 3. Implement an advanced retrieval strategy (query decomposition and/or graph
#    traversal) that addresses them.
# 4. Measure the improvement against your baseline using your 3.1 evaluation set.

# %% [markdown]
# ## Step 1 — Keep your capstone scenario
#
# Use the **same scenario** you chose in Checkpoint 1.1 and have built on since.
#
# | Scenario | Corpus | Natural relationships to exploit in a graph |
# |---|---|---|
# | **Research Paper Navigator** | ~150 research paper PDFs (`Labs/CapstoneDatasets/ResearchPapers/`) | citations, shared authors, shared topics/keywords, "published-before" |
# | **Wikipedia Retrieval Engine** | ~2,400 Wikipedia HTML articles (`Labs/CapstoneDatasets/Wikipedia/`) | hyperlinks between articles, shared categories, mentioned entities |
#
# Multistep decomposition shines on cross-document questions ("Compare X and Y,"
# "how did an idea evolve"); graph retrieval shines when the answer needs documents
# that are *related* to a hit but don't themselves match the query terms.

# %% [markdown]
# ## Setup (~5 min)
#
# 1. **Python 3.11 or 3.12**
# 2. `pip install langchain-openai langchain-core python-dotenv networkx`
# 3. Use the OpenRouter API key provided for this course. This checkpoint uses
#   the `openai/gpt-5.4-mini` model, with usage covered by course credits.
# 4. Create a `.env` file next to this script: `OPENROUTER_API_KEY=sk-or-v1-...`
#
# This script runs on a tiny built-in sample corpus, so you do **not** need your full
# dataset indexed to complete it. You *will* refer to your real 2.1 system and 3.1
# evaluation results when you write the report.

# %%
from __future__ import annotations

import warnings
warnings.filterwarnings("ignore")

import json
import os
import re
from datetime import datetime
from pathlib import Path
from typing import Any

import networkx as nx
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

from rank_bm25 import BM25Okapi
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document
from pypdf import PdfReader

# %%
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
LLM_MODEL = "openai/gpt-5.4-mini"  # Latest small OpenAI model, fast; covered by course credits
TEMPERATURE = 0.0
LOG_PATH = Path.cwd() / "checkpoint_4_1_responses.log"

JUDGE_MODEL = "openai/gpt-oss-120b"
EMBEDDING_MODEL = "openai/text-embedding-3-small"
CHROMA_DIR = "chroma_db"
TOP_K = 3

# === SET THIS to the scenario you chose in Checkpoint 1.1 ===
SCENARIO = "research_papers"   # "research_papers" or "wikipedia"

DECOMPOSE_SYSTEM = (
    "You are a query decomposition assistant for a document retrieval system. "
    "Break the user's question into 2-4 focused sub-queries that together cover "
    "everything needed to answer it. Each sub-query should target a distinct aspect. "
    'Return ONLY a JSON array of strings, e.g., ["sub-query 1", "sub-query 2"].'
)

ANSWER_SYSTEM = (
    "You are a helpful assistant. Answer the question using only the provided "
    "documents, and quote from them where you can. If the documents do not contain "
    "the answer, say so rather than guessing."
)


# %%
def check_api_key() -> str:
    load_dotenv()
    key = os.getenv("OPENROUTER_API_KEY")
    if not key:
        raise RuntimeError(
            "OPENROUTER_API_KEY not set. Grab a free key at "
            "https://openrouter.ai/keys, put it in a .env file next to this "
            "script, and rerun."
        )
    return key


def make_llm() -> ChatOpenAI:
    return ChatOpenAI(
        model=LLM_MODEL,
        temperature=TEMPERATURE,
        api_key=check_api_key(),
        base_url=OPENROUTER_BASE_URL,
    )


def log_response(label: str, prompt: str, response: str) -> None:
    ts = datetime.now().isoformat(timespec="seconds")
    entry = (
        f"[{ts}]  {label}  SCENARIO={SCENARIO}  MODEL={LLM_MODEL}\n"
        f"PROMPT:   {prompt}\n"
        f"RESPONSE: {response}\n"
        f"{'-' * 72}\n"
    )
    with LOG_PATH.open("a", encoding="utf-8") as fh:
        fh.write(entry)

KNOWN_AUTHORS = [
    "Armando Solar-Lezama",
    "Rishabh Singh",
    "Kevin Ellis",
    "Martin Rinard",
    "Marsha Chechik",
    "Rajeev Alur",
    "Sumit Gulwani"
]

TOPIC_KEYWORDS = {

    "program_synthesis": [
        "program synthesis",
        "synthesis"
    ],

    "verification": [
        "verification",
        "formal verification"
    ],

    "ai_software_engineering": [
        "software engineering",
        "AI for software engineering"
    ]
}
def extract_authors(text):

    found = []

    lower_text = text.lower()

    for author in KNOWN_AUTHORS:
        if author.lower() in lower_text:
            found.append(author)

    return found


def extract_topics(text):

    found = []

    lower_text = text.lower()

    for topic, keywords in TOPIC_KEYWORDS.items():

        if any(
            keyword.lower() in lower_text
            for keyword in keywords
        ):
            found.append(topic)

    return found
# %% [markdown]
# ## A tiny sample corpus (stands in for your real one)
#
# Six short "documents" with relationships so both techniques are demonstrable
# without indexing your full corpus. Each has an ID, text, a `topics` list, and
# `links` to related documents (i.e., think citations for papers, or hyperlinks for
# Wikipedia). Your real system would derive these from your actual data.

# %%
# Path to your real corpus
PDF_FOLDER = Path(
    r"C:\MIT\MIT_RAG_Capstone_Project\Checkpoint_1_1\Capstone_Checkpoint_1_1_starter\ResearchPapers"
)


def load_pdf_corpus():
    """
    Load all PDFs from the research paper corpus.
    Uses only the first few pages of each paper to avoid
    slow extraction and problematic PDFs.
    """
    docs = []

    pdf_files = list(PDF_FOLDER.glob("*.pdf"))

    print(f"Found {len(pdf_files)} PDF files")

    for idx, pdf_file in enumerate(pdf_files, start=1):

        print(f"[{idx}/{len(pdf_files)}] Loading {pdf_file.name}")

        try:
            reader = PdfReader(str(pdf_file))

            text_parts = []

            # Only first 5 pages
            for page_num, page in enumerate(reader.pages[:5]):

                try:
                    text = page.extract_text()

                    if text:
                        text_parts.append(text)

                except Exception as e:
                    print(
                        f"   skipped page {page_num+1}: {e}"
                    )

            full_text = "\n".join(text_parts)

            # limit stored text
            full_text = full_text[:10000]

            docs.append(
                {
                    "id": pdf_file.stem,
                    "text": full_text,
                    "authors": extract_authors(full_text),
                    "topics": extract_topics(full_text)
                }
            )

            print("   loaded")

        except Exception as e:
            print(f"Failed to load {pdf_file.name}: {e}")

    return docs

SAMPLE_DOCS = load_pdf_corpus()

DOC_BY_ID = {d["id"]: d for d in SAMPLE_DOCS}



def keyword_score(query: str, text: str) -> float:
    """A tiny, dependency-free relevance score: shared word count. This stands in for
    your real hybrid retriever, so the retrieval step needs no extra dependencies. The
    demo still calls the LLM, which requires an OpenRouter API key."""
    q = set(re.findall(r"[a-z0-9]+", query.lower()))
    t = set(re.findall(r"[a-z0-9]+", text.lower()))
    return float(len(q & t))


def baseline_retrieve(query: str, k: int = 3) -> list[tuple[str, float]]:
    """Single-pass retrieval: Score every doc once, take the top k. (id, score)."""
    scored = [(d["id"], keyword_score(query, d["text"])) for d in SAMPLE_DOCS]
    scored.sort(key=lambda x: x[1], reverse=True)
    return [(i, s) for i, s in scored[:k] if s > 0]


# %% [markdown]
# ## Step 2 — Multistep retrieval (provided, adapted from Lab 4.1)
#
# `decompose_query` asks the LLM to split the question into sub-queries. For each
# sub-query, we retrieve the top documents and **sum** each document's score across
# the sub-queries. A document relevant to several parts of the question rises to the
# top. (In your real system, replace `baseline_retrieve` with your 2.1 retriever.)

# %%
def decompose_query(llm: ChatOpenAI, query: str) -> list[str]:
    messages = [SystemMessage(content=DECOMPOSE_SYSTEM), HumanMessage(content=query)]
    raw = llm.invoke(messages).content.strip()
    if raw.startswith("```"):
        raw = raw.strip("`")
        raw = raw[raw.find("["): raw.rfind("]") + 1] if "[" in raw else raw
    try:
        parsed = json.loads(raw)
        if isinstance(parsed, list) and all(isinstance(x, str) for x in parsed) and parsed:
            return parsed
    except (json.JSONDecodeError, ValueError):
        pass
    return [query]


def multistep_retrieve(llm: ChatOpenAI, query: str, k: int = 3) -> list[str]:
    sub_queries = decompose_query(llm, query)
    print(f"Decomposed to: {sub_queries}")
    score_map: dict[str, float] = {}
    for sq in sub_queries:
        for doc_id, score in baseline_retrieve(sq, 3 * k):
            score_map[doc_id] = score_map.get(doc_id, 0.0) + score
    ranked = sorted(score_map.items(), key=lambda kv: kv[1], reverse=True)
    return [doc_id for doc_id, _ in ranked[:k]]


# %% [markdown]
# ## Step 3 — Graph-based retrieval (provided, adapted from Lab 4.2)
#
# Build a graph whose nodes are documents and topics, with edges for "links to" and
# "relates to topic." Retrieval seeds with the baseline hits, then pulls in each
# seed's linked documents and topic-siblings as extra context — deduplicated.

# %%


def build_graph(docs):

    G = nx.Graph()

    for doc in docs:

        paper_id = doc["id"]

        G.add_node(
            paper_id,
            node_type="paper"
        )

        authors = extract_authors(
            doc["text"]
        )

        topics = extract_topics(
            doc["text"]
        )

        for author in authors:

            G.add_node(
                author,
                node_type="author"
            )

            G.add_edge(
                paper_id,
                author,
                edge_type="written_by"
            )

        for topic in topics:

            G.add_node(
                topic,
                node_type="topic"
            )

            G.add_edge(
                paper_id,
                topic,
                edge_type="has_topic"
            )

    return G


def graph_retrieve(
    graph,
    seed_docs,
    max_results=10
):

    expanded = set(seed_docs)

    for paper in seed_docs:

        if not graph.has_node(paper):
            continue

        neighbors = list(
            graph.neighbors(paper)
        )

        for neigh in neighbors:

            neigh_type = graph.nodes[neigh].get(
                "node_type"
            )

            if neigh_type in [
                "author",
                "topic"
            ]:

                second_hop = list(
                    graph.neighbors(neigh)
                )

                for related in second_hop:

                    if graph.nodes[related].get(
                        "node_type"
                    ) == "paper":

                        expanded.add(related)

    return list(expanded)[:max_results]


# %% [markdown]
# ## Step 4 — Your advanced-retrieval plan (TODO)
#
# Design how you will apply these techniques to **your** capstone corpus. Return a
# dictionary with the keys below. This is the plan you will implement in your real system
# and describe in the report. Keep it concrete and specific to your scenario.

# %%

# %% [markdown]
# ## Step 5 — Run baseline vs. advanced and capture the evidence
#
# This demonstration compares single-pass retrieval with the two advanced retrieval strategies 
# using the sample corpus and a representative query. It also logs the result. Read the
# retrieved sets: The advanced strategies should surface related documents that the
# baseline misses. Then run the same comparison in your real system for your report.

# %%
def answer_from_docs(llm: ChatOpenAI, query: str, doc_ids: list[str]) -> str:
    context = "\n\n".join(f"[{i}] {DOC_BY_ID[i]['text']}" for i in doc_ids if i in DOC_BY_ID)
    messages = [
        SystemMessage(content=ANSWER_SYSTEM),
        HumanMessage(content=f"Documents:\n{context}\n\nQuestion: {query}"),
    ]
    return llm.invoke(messages).content


MODULE4_EVAL_SET = [

{
    "question":
    "How do the papers describe scalable program synthesis techniques?",

    "grading_notes":
    "Must discuss modular verification, specification decomposition, type-directed synthesis, search reduction, or learned representations.",

    "techniques":
    ["query_decomposition", "multi_hop"]
},

{
    "question":
    "How has Armando Solar-Lezama's research evolved from program synthesis toward AI for software engineering?",

    "grading_notes":
    "Must describe the progression from program synthesis and verification toward broader AI for software engineering themes using evidence from multiple papers.",

    "techniques":
    ["query_decomposition", "graph_retrieval", "multi_hop"]
},

{
    "question":
    "Across Armando Solar-Lezama's papers, how do program synthesis, formal verification, and AI for software engineering connect to one another as part of a broader research agenda?",

    "grading_notes":
    "Must synthesize evidence across multiple papers and explain relationships between these research areas.",

    "techniques":
    ["graph_retrieval", "relationship_aware", "multi_hop"]
},

{
    "question":
    "Which research themes appear repeatedly across Armando Solar-Lezama papers even when different terminology is used, and how are those themes connected?",

    "grading_notes":
    "Must identify recurring themes and connect evidence from multiple documents.",

    "techniques":
    ["graph_retrieval", "relationship_aware"]
}
]

def judge_answer(
    question,
    grading_notes,
    answer
):
    llm = ChatOpenAI(
        model=JUDGE_MODEL,
        temperature=0,
        api_key=check_api_key(),
        base_url=OPENROUTER_BASE_URL,
    )

    rubric = f"""
Question:
{question}

Expected:
{grading_notes}

Student Answer:
{answer}

Respond ONLY as JSON:

{{
  "score": 0-10,
  "pass": true or false,
  "reason": "brief explanation"
}}
"""

    response = llm.invoke(
        [HumanMessage(content=rubric)]
    ).content

    try:
        return json.loads(response)
    except:
        return {
            "score": 0,
            "pass": False,
            "reason": "Judge parse failed"
        }

def compare_baseline_vs_multistep():

    llm = make_llm()

    baseline_passes = 0
    advanced_passes = 0
    baseline_total_score = 0
    advanced_total_score = 0
    for item in MODULE4_EVAL_SET:

        question = item["question"]

        print("\n" + "=" * 80)
        print("Question: " + question)

        print(
            "Techniques: "
            + ", ".join(item["techniques"])
        )
        baseline_docs = [
            doc_id
            for doc_id, _
            in baseline_retrieve(question, TOP_K)
        ]

        baseline_answer = answer_from_docs(
            llm,
            question,
            baseline_docs
        )

        baseline_result = judge_answer(
            question,
            item["grading_notes"],
            baseline_answer
        )

        if baseline_result["pass"]:
            baseline_passes += 1

        baseline_total_score += baseline_result["score"]

        advanced_docs = multistep_retrieve(
            llm,
            question,
            TOP_K
        )

        advanced_answer = answer_from_docs(
            llm,
            question,
            advanced_docs
        )

        advanced_result = judge_answer(
            question,
            item["grading_notes"],
            advanced_answer
        )
        log_response(
            "BASELINE",
            question,
            baseline_answer
        )

        log_response(
            f"ADVANCED ({', '.join(item['techniques'])})",
            question,
            advanced_answer
        )
        if advanced_result["pass"]:
            advanced_passes += 1

        advanced_total_score += advanced_result["score"]

        print(
            f"Baseline Pass={baseline_result['pass']} "
            f"Score={baseline_result['score']}"
        )

        print(
            f"Advanced Pass={advanced_result['pass']} "
            f"Score={advanced_result['score']}"
        )

    baseline_rate = (
        baseline_passes / len(MODULE4_EVAL_SET)
    ) * 100

    advanced_rate = (
        advanced_passes / len(MODULE4_EVAL_SET)
    ) * 100


  
    baseline_avg = (
        baseline_total_score /
        len(MODULE4_EVAL_SET)
    )

    advanced_avg = (
        advanced_total_score /
        len(MODULE4_EVAL_SET)
    )

    print("\n")
    print("=" * 80)

    print(
        f"BASELINE AVG SCORE: {baseline_avg:.2f}"
    )

    print(
        f"ADVANCED AVG SCORE: {advanced_avg:.2f}"
    )

    print(
        f"BASELINE PASS RATE: {baseline_rate:.1f}%"
    )

    print(
        f"ADVANCED PASS RATE: {advanced_rate:.1f}%"
    )

    print(
        f"PASS RATE IMPROVEMENT: "
        f"{advanced_rate - baseline_rate:.1f}%"
    )
    return baseline_rate, advanced_rate


print("\n")
print("=" * 80)
print("MODULE 4 EVALUATION")
print("=" * 80)

compare_baseline_vs_multistep()


# %% [markdown]
# ## Step 6
#
# The checkpoint deliverable is your completed Capstone Checkpoint 4.1 worksheet, not
# code. Using evidence from your real system, cover:
#
# 1. **System overview** State your scenario and your 2.1 baseline retriever.
# 2. **Failure diagnosis** Describe the meaningful retrieval failure or limitation identified through
#    your Checkpoint 3.1 evaluation and explain why it occurs.
# 3. **Data redesign** Describe the node/edge schema you defined to expose relationships in
#    your corpus (citations/authors/topics, or links/categories/entities), and why you chose it. 
# 4. **Advanced retrieval implemented** Provide a query decomposition and/or graph traversal,
#    with code or log evidence. Label primary vs. context documents in the prompt.
# 5. **Comparative evaluation** Re-run your 3.1 evaluation on baseline vs advanced,
#    **comparing a similar number of retrieved documents**, and report before/after.
# 6. **Analysis & reflection** Discuss where it helped, where it hurt (redundant/over-broad
#    context), the cost/latency trade-off, and remaining limitations.
#
