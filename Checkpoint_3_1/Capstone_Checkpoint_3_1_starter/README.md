# Capstone Checkpoint 3.1

## Evaluation Infrastructure and Baseline Diagnosis

This starter script demonstrates how to evaluate a retrieval-augmented generation (RAG) system with an LLM judge. It retrieves documents, asks an LLM to answer using only those documents, and scores the answer as `pass` or `fail` against grading notes.

The demonstration uses a small built-in corpus. It is not the Research Paper Navigator or Wikipedia corpus. For the capstone submission, apply the same evaluation approach to the scenario and baseline retriever from Checkpoint 2.1.

## Prerequisites

- Python 3.11 or 3.12
- An OpenRouter API key provided for the course
- Internet access for the OpenRouter API calls

Install the required packages:

```powershell
python -m pip install langchain-openai langchain-core python-dotenv
```

## Configuration

Create a `.env` file in this folder, next to the Python script:

```text
OPENROUTER_API_KEY=sk-or-v1-your-key-here
```

Do not commit or share this file. The script loads the key with `python-dotenv` and sends requests through the OpenRouter-compatible API at `https://openrouter.ai/api/v1`.

## Complete the starter task

Open `capstone_checkpoint_3_1_evaluation_starter.py` and implement `my_eval_set()`.

Return 3 to 5 dictionaries in this format:

```python
def my_eval_set() -> list[dict]:
    return [
        {
            "question": "What is BM25?",
            "grading_notes": "States that BM25 is a keyword or term-frequency ranking function for documents.",
        },
    ]
```

Build an evaluation set that includes:

- Questions the baseline should answer correctly.
- At least one question expected to expose a baseline weakness.
- Grading notes that state the information a correct answer must contain.
- Questions appropriate to your chosen capstone scenario.

The supplied demonstration retriever uses token overlap and the supplied answer function uses the six `SAMPLE_DOCS`. When evaluating your capstone system, replace or connect these functions to the retrieval and answer-generation implementation from Checkpoint 2.1.

## Run the evaluation

From this folder, run:

```powershell
python .\capstone_checkpoint_3_1_evaluation_starter.py
```

The script will:

1. Load the API key and create the chat model client.
2. Run each item in `my_eval_set()` through retrieval, answer generation, and judging.
3. Print retrieved document IDs, answers, verdicts, and the baseline pass rate.
4. Run framework validation using a correct answer and a deliberately false answer.
5. Write timestamped results to `checkpoint_3_1_evaluation.log`.

Framework validation is successful when the correct answer receives `PASS` and the manipulated answer receives `FAIL`.

## Expected output

The exact answer text and verdicts can vary because an LLM is used. The final output should include a baseline pass rate and a validation summary similar to:

```text
Baseline pass rate: 3/4

--- Framework validation ---
  correct answer   -> PASS   (expected PASS)
  manipulated answer -> FAIL   (expected FAIL)
```

Review the generated log alongside the printed output when recording results. Do not treat the pass rate as a universal quality score: it depends on the questions, grading notes, retriever, answer model, and judge model.

## Worksheet deliverable

Use the evidence from your capstone evaluation to complete [Required_Capstone_Checkpoint_3_1_Worksheet.docx](../Required_Capstone_Checkpoint_3_1_Worksheet.docx). Address these seven sections:

1. System overview: scenario and Checkpoint 2.1 baseline retriever.
2. Evaluation design: correctness criteria, grading notes, judge, and thresholds.
3. Testing approach: evaluation-set construction and execution process.
4. Baseline results: pass/fail outcomes and observed gaps.
5. Performance analysis: strengths, weaknesses, and failure modes.
6. Evaluation framework validation: evidence that degraded output is detected.
7. Reflection and next steps: limitations and improvements for Checkpoint 4.1.

## Important implementation notes

- `TOP_K` controls the number of retrieved documents passed to the answer model.
- `LLM_MODEL` and `TEMPERATURE` define the answer and judge model configuration.
- `SCENARIO` is descriptive in the starter and does not switch the demo corpus automatically.
- The script uses Jupytext-style `# %%` markers, so it can run as a normal Python file or be opened as notebook-like cells in VS Code.
- The script makes multiple paid or credit-consuming API calls. Keep the evaluation set small while developing.