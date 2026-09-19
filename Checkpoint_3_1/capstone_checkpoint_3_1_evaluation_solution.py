r"""Capstone Checkpoint 3.1 — Evaluation Infrastructure and Baseline Diagnosis (starter).
Jupytext-style cell markers (# %% / # %% [markdown]) — runnable as a
plain script AND openable as cells in VS Code / PyCharm / Jupytext.

This demonstration system is not your capstone system and 
does not use the Research Paper Navigator or Wikipedia corpus.
"""

# %% [markdown]
# # Capstone Checkpoint 3.1 — Evaluation Infrastructure and Baseline Diagnosis
# **MO-LLM Module 3 / Required Capstone Checkpoint (120 minutes)**
#
# ## What this checkpoint is
#
# #
# This mirrors **Lab 3.1** (an LLM-judge that scores answers pass/fail against grading
# notes), applied to your capstone system. 
# You have a baseline retrieval system from Checkpoint 2.1. In this checkpoint, 
# you will use a structured evaluation approach to measure baseline performance, diagnose strengths 
# and weaknesses, and examine whether the evaluation framework detects problematic outputs.
# The starter script includes a small demonstration corpus and retriever to illustrate the 
# evaluation workflow. Apply the same evaluation approach to your selected capstone scenario and 
# baseline retrieval system. The graded deliverable is the completed Capstone Checkpoint 3.1 worksheet.
#
# **Learning outcomes (Module 3):**
# 1. Define evaluation metrics that reflect the real-world performance requirements of an LLM-powered retrieval system.
# 2. Identify key variables that influence system performance during evaluation and development.
# 3. Use language models to support evaluation tasks while avoiding common pitfalls.
# 4. Evaluate the performance of a retrieval-augmented system during development using a structured evaluation framework.

# %% [markdown]
# ## Step 1 — Keep your capstone scenario
#
# Use the **same scenario and baseline retriever** from Checkpoints 1.1 and 2.1.
#
# | Scenario | Corpus |
# |---|---|
# | **Research Paper Navigator** | ~150 research-paper PDFs (`Labs/CapstoneDatasets/ResearchPapers/`) |
# | **Wikipedia Retrieval Engine** | ~2,400 Wikipedia HTML articles (`Labs/CapstoneDatasets/Wikipedia/`) |

# %% [markdown]
# ## Setup (~5 min)
#
# 1. **Python 3.11 or 3.12.**
# 2. `pip install langchain-openai langchain-core python-dotenv`
# 3. Use the OpenRouter API key provided for this program.
# 4. Create a `.env` file next to this script: `OPENROUTER_API_KEY=sk-or-v1-...`
#
# Runs on a tiny built-in sample corpus, so you do not need to prepare your own dataset.
# It still requires an OpenRouter API key to run the LLM (it is not offline or free of API
# calls). You apply the same evaluation approach to your real system for the report.

# %%
from __future__ import annotations

import warnings
warnings.filterwarnings("ignore")

import os
import re
from datetime import datetime
from pathlib import Path

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
LLM_MODEL = "openai/gpt-5.4-mini"  # latest small OpenAI model, fast; covered by course credits
JUDGE_MODEL = "openai/gpt-oss-120b"
EMBEDDING_MODEL = "openai/text-embedding-3-small"
CHROMA_DIR = "chroma_db"
TEMPERATURE = 0
TOP_K = 3
LOG_PATH = Path.cwd() / "checkpoint_3_1_evaluation.log"

SCENARIO = "research_papers"   # "research_papers" or "wikipedia"

ANSWER_SYSTEM = (
    "You are a helpful assistant. Answer the question using ONLY the provided "
    "documents, and quote from them where you can. If the documents do not contain "
    "the answer, say so rather than guessing."
)
JUDGE_SYSTEM = (
    "You are a strict evaluator. You are given an ANSWER and GRADING NOTES describing "
    "what a correct answer must contain. Reply with exactly one word: 'pass' if the "
    "answer satisfies the grading notes, or 'fail' if it does not."
)


# %%
def check_api_key() -> str:
    load_dotenv()
    key = os.getenv("OPENROUTER_API_KEY")
    if not key:
        raise RuntimeError(
            "OPENROUTER_API_KEY is not set. Use the OpenRouter API key "
            "provided for this program, put it in a .env file next to this "
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

def make_judge_llm() -> ChatOpenAI:
    return ChatOpenAI(
        model=JUDGE_MODEL,
        temperature=0,
        api_key=check_api_key(),
        base_url=OPENROUTER_BASE_URL,
    )

def log(label: str, text: str) -> None:
    ts = datetime.now().isoformat(timespec="seconds")
    with LOG_PATH.open("a", encoding="utf-8") as fh:
        fh.write(f"[{ts}] {label}\n{text}\n{'-' * 72}\n")


# %% [markdown]
# ## Demonstration corpus + retriever (provided to illustrate the evaluation workflow)

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
                }
            )

            print("   loaded")

        except Exception as e:
            print(f"Failed to load {pdf_file.name}: {e}")

    return docs

SAMPLE_DOCS = load_pdf_corpus()

DOC_BY_ID = {d["id"]: d for d in SAMPLE_DOCS}


def _tokens(text: str) -> set[str]:
    return set(re.findall(r"[a-z0-9]+", text.lower()))

# Build BM25 index AFTER _tokens exists
bm25_documents = [
    list(_tokens(doc["text"]))
    for doc in SAMPLE_DOCS
]

bm25 = BM25Okapi(bm25_documents)
DOC_BY_ID = {d["id"]: d for d in SAMPLE_DOCS}


def get_embeddings():
    return OpenAIEmbeddings(
        model=EMBEDDING_MODEL,
        api_key=check_api_key(),
        base_url=OPENROUTER_BASE_URL,
    )
documents = [
    Document(
        page_content=d["text"],
        metadata={"id": d["id"]}
    )
    for d in SAMPLE_DOCS
]

vector_db = Chroma.from_documents(
    documents,
    get_embeddings(),
    persist_directory=CHROMA_DIR
)


def normalize(scores, invert=False):

    if not scores:
        return []

    low = min(scores)
    high = max(scores)

    if high == low:
        return [0.5] * len(scores)

    norm = [
        (s - low) / (high - low)
        for s in scores
    ]

    if invert:
        norm = [1.0 - s for s in norm]

    return norm

def retrieve(query: str, k: int = TOP_K):

    # ---------- BM25 ----------
    bm25_scores = bm25.get_scores(
    list(_tokens(query))
    )

    bm25_norm = normalize(
        list(bm25_scores)
    )

    # ---------- Vector ----------
    vector_results = vector_db.similarity_search_with_score(
        query,
        k=len(SAMPLE_DOCS)
    )

    vector_distance_map = {}

    for doc, distance in vector_results:
        vector_distance_map[
            doc.metadata["id"]
        ] = distance

    vector_scores = []

    for paper in SAMPLE_DOCS:

        vector_scores.append(
            vector_distance_map.get(
                paper["id"],
                9999
            )
        )

    vector_norm = normalize(
        vector_scores,
        invert=True
    )

    # ---------- Hybrid ----------
    combined = []

    for idx, paper in enumerate(SAMPLE_DOCS):

        score = (
            0.5 * bm25_norm[idx]
            + 0.5 * vector_norm[idx]
        )

        combined.append(
            (
                paper["id"],
                float(score)
            )
        )

    combined.sort(
        key=lambda x: x[1],
        reverse=True
    )

    return combined[:k]

def answer(llm: ChatOpenAI, query: str, doc_ids: list[str]) -> str:
    context = "\n\n".join(f"[{i}] {DOC_BY_ID[i]['text']}" for i in doc_ids if i in DOC_BY_ID)
    messages = [
        SystemMessage(content=ANSWER_SYSTEM),
        HumanMessage(content=f"Documents:\n{context}\n\nQuestion: {query}"),
    ]
    return llm.invoke(messages).content


# %% [markdown]
# ## Step 2 — The evaluation metric (provided)
#
# A simple **LLM-judge** that returns pass/fail by checking an answer against grading
# notes — the same idea as Lab 3.1's DiscreteMetric, written directly here so the
# checkpoint needs no extra packages. Tuning this metric (stricter notes, a 'partial'
# level, a stronger judge model) is part of the diagnosis.

# %%
"""def judge(llm: ChatOpenAI, answer_text: str, grading_notes: str) -> str:
    messages = [
        SystemMessage(content=JUDGE_SYSTEM),
        HumanMessage(content=f"ANSWER:\n{answer_text}\n\nGRADING NOTES:\n{grading_notes}\n\nVerdict (pass/fail):"),
    ]
    verdict = llm.invoke(messages).content.strip().lower()
    return "pass" if "pass" in verdict else "fail"
"""
def judge(answer_text: str,
          grading_notes: str) -> str:

    llm = make_judge_llm()

    messages = [
        SystemMessage(content=JUDGE_SYSTEM),
        HumanMessage(
            content=f"ANSWER:\n{answer_text}\n\n"
                    f"GRADING NOTES:\n{grading_notes}\n\n"
                    f"Verdict (pass/fail):"
        ),
    ]

    verdict = llm.invoke(messages).content.strip().lower()

    return "pass" if "pass" in verdict else "fail"

# %% [markdown]
# ## Step 3 — Your evaluation set (TODO)
#
# Define the test set your evaluation runs on. Each item is a question plus
# **grading notes** — a short description of what a correct answer must contain (the
# judge checks the answer against these). Good evaluation sets include questions you
# expect to pass AND questions that probe known weaknesses.
#
# Return a list of 3-5 dicts: `{"question": "...", "grading_notes": "..."}`.

# %%
def my_eval_set() -> list[dict]:
    """Evaluation set for the Research Paper Navigator scenario.
    Includes factual, author-based, conceptual, and intentionally
    difficult questions to evaluate baseline performance.
    """

    return [
        {
            "question": "Quote the first sentence of the paper Improved modeling of RNA-binding protein motifs in an interpretable neural model of RNA splicing.",
            #"grading_notes": "Must return the first sentence from the Gupta2024 paper and clearly quote it.",
            "grading_notes": "Sequence-specific RNA-binding proteins (RBPs) play central roles in splicing decisions."
        },

        {
            "question": "List all papers where Armando Solar-Lezama is an author or co-author.",
            "grading_notes": "Must identify papers containing Armando Solar-Lezama as an author or co-author and provide corresponding paper titles.",
            #"grading_notes": "Special Issue on Syntax-Guided Synthesis Preface, Challenges and Paths Towards AI for Software Engineering, Position: Future Research and Challenges Remain Towards AI for Software Engineering, Liquid Information Flow Control"
        },

        {
            "question": "Based on Armando Solar-Lezama's papers in the collection, what is his primary research focus?",
            "grading_notes": "Must mention either AI for software engineering, program synthesis, syntax-guided synthesis, software analysis or formal verification.",
            #"grading_notes": "Must mention either AI for software engineering, software analysis or formal verification.",
            #"grading_notes": "Based on the provided documents, Armando Solar-Lezama’s papers here are about **software engineering / software analysis, especially string constraints and AI for software engineering**",
        },

        {
            "question": "How do the papers describe scalable program synthesis techniques?",
            "grading_notes": "Must discuss scalability through modular verification, specification decomposition, type-directed synthesis, search reduction, or learned representations."
        },

        {
            "question": "Summarize all research areas represented across every paper written by Armando Solar-Lezama in the collection.",
            "grading_notes":"Answer should identify multiple research areas across all retrieved papers. Partial answers should fail."
        }
    ]


# %% [markdown]
# Run the demonstration code to understand the evaluation workflow. Then apply the same evaluation 
# design to your own capstone baseline system, using its retrieval and answer-generation functions. 
# Record results from your capstone system in the worksheet.
#
# ## Step 4 — Run the baseline evaluation
#
# Answers each question with the baseline retriever, scores it with the judge, and
# reports the pass rate. This is your baseline diagnosis: the failures are what you
# analyse in the report.

# %%
def run_evaluation() -> None:
    llm = make_llm()
    eval_set = my_eval_set()
    passes = 0
    print(f"Checkpoint 3.1 — baseline evaluation  |  scenario: {SCENARIO}\n")
    for i, item in enumerate(eval_set, 1):
        hits = retrieve(item["question"], TOP_K)
        ans = answer(llm, item["question"], [doc_id for doc_id, _ in hits]) if hits else "(no documents retrieved)"
        #verdict = judge(llm, ans, item["grading_notes"])
        verdict = judge(ans, item["grading_notes"])
        passes += verdict == "pass"
        print("=" * 72)
        print(f"Q{i}: {item['question']}")
        print(f"  retrieved={hits}  verdict={verdict.upper()}")
        print(f"  answer: {ans}")
        log(f"Q{i}: {item['question']}", f"retrieved={hits}\nverdict={verdict}\nanswer={ans}")
    print("=" * 72)
    print(f"Baseline pass rate: {passes}/{len(eval_set)}")


# %% [markdown]
# ## Step 5 — Validate the framework: can it catch a manipulated answer? (provided)
#
# A good evaluation framework must FAIL a wrong answer, not just pass good ones. This
# takes a question your corpus can answer, produces a correct answer, then feeds the
# judge a deliberately manipulated (false) answer — and checks that the judge flags it.
# This is your "evaluation framework validation" evidence for the report.

# %%
def validate_framework() -> None:
    llm = make_llm()
    q = "What is BM25?"
    notes = "States that BM25 is a keyword / term-frequency ranking function for documents."
    good = answer(llm, q, [doc_id for doc_id, _ in retrieve(q)])
    manipulated = "BM25 is a deep neural network that generates images from text prompts."
    #good_verdict = judge(llm, good, notes)
    #manip_verdict = judge(llm, manipulated, notes)
    good_verdict = judge(good, notes)
    manip_verdict = judge(manipulated, notes)
    print("\n--- Framework validation ---")
    print(f"  correct answer   -> {good_verdict.upper()}   (expected PASS)")
    print(f"  manipulated answer -> {manip_verdict.upper()}   (expected FAIL)")
    print("  The framework works if it PASSES the correct answer and FAILS the manipulated one.")
    log("FRAMEWORK VALIDATION", f"good={good_verdict} manipulated={manip_verdict}")


# %%
run_evaluation()
validate_framework()

# %% [markdown]
# ## Step 6 — Your written responses in the Capstone Checkpoint 3.1 worksheet
#
# Complete the Capstone Checkpoint 3.1 worksheet using evidence from your capstone evaluation.
# Address the seven sections: system overview, evaluation design, testing approach, baseline results, 
# performance analysis, evaluation framework validation, and reflection and next steps.
#
# 1. **System overview** — your scenario and your 2.1 baseline retriever.
# 2. **Evaluation design** — your criteria and metric (what "correct" means; how the
#    judge decides pass/fail; any thresholds).
# 3. **Testing approach** — how you built your evaluation set and ran it.
# 4. **Baseline results** — the pass/fail outcomes and where the system falls short.
# 5. **Performance analysis** — what the results reveal about strengths, weaknesses,
#    and failure modes.
# 6. **Evaluation framework validation** — show your framework detects a degraded or
#    manipulated output (use the Step 5 result, or your own).
# 7. **Reflection and next steps** — limitations of your evaluation and what you'll
#    improve (this motivates the advanced retrieval in Checkpoint 4.1).
