# 🧠 AmbedkarGPT – RAG System (Assignment 1 & 2)

A Retrieval-Augmented Generation (RAG) setup using:

- Local ChromaDB
- Sentence Transformers (MiniLM)
- Ollama Mistral 7B
- Pure Python (no LangChain)
- Reproducible testing for Assignment 2

This repo has two working parts:

- Assignment 1 → RAG system
- Assignment 2 → Testing different text-splitting methods & checking results

Made easy for a recruiter to clone, run, and check.

## 📦 Project Structure

```
AMBEDKARGPT-INTERN-TASK
│
├── app.py                 # LLM answers + main logic
├── pipeline.py            # Load docs + split + embed
├── vectorstore.py         # Chroma search + retrieval
├── config.py              # All settings
├── utils.py               # Helper tools
├── chroma_store/          # Local vector DB
│
├── speech.txt             # Content for Assignment 1
│
├── assignment2/
│   ├── corpus/            # 6 docs corpus
│   ├── evaluation.py      # Different methods evaluation script 
│   ├── test_dataset.json  # 25 questions (given)
│   ├── results/           # Logs for each test
│   ├── results_analysis.md# Deep dive on splitting methods
│
└── README.md              # This file
```

## 🟦 Assignment 1 — RAG System

### ✔ What it does
- Load docs → Split text → Turn into embeddings (MiniLM)
- Store in local Chroma
- Find top matches
- Generate answers with Ollama Mistral 7B
- Ask questions via command line

### ▶️ How to Run Assignment 1
1. Activate python virtual environment:
   ```
   python -m venv .venv
   .venv\Scripts\activate
   ```
2. Install Dependencies:
   ```
   pip install -r requirements.txt
   ```

3. Start Ollama & get Mistral:
   ```
   ollama pull mistral
   ```

4. Run the app:
   ```
   python app.py
   ```

5. Ask questions like:
   ```
   Enter question: What is the remedy for the caste system?
   ```

## 🟧 Assignment 2 — Testing Framework

Tests three text-splitting ways:

- Small: 250 chars / 50 overlap
- Medium: 550 chars / 80 overlap
- Large: 900 chars / 100 overlap

Checks these metrics:

- Hit Rate
- MRR
- Precision@3
- ROUGE-L
- BLEU
- Semantic Similarity (Cosine)
- Faithfulness (how grounded answers are)

## 📌 Running the Evaluation (Assignment 2)

1. Make sure Ollama is installed and the Mistral model is available:
   ```bash
   ollama pull mistral
   ```

2. Now run the evaluation script:
   ```bash
   cd assignment2
   python evaluation.py
   ```

### 📁 Output Files Generated

Running the above command will produce:

```
assignment2/
│
├── results/
│   ├── results_small.json
│   ├── results_medium.json
│   ├── results_large.json
│
└── test_results_real.json   ← Combined summary  file
```

### 📄 Deep Analysis
Check `assignment2/results_analysis.md` for:

- Which splitting method won
- Metric tables
- What went wrong & why
- Tips to make RAG better

## 🏆 Quick Wins

Small chunks (250/50) was produced better results in this dataset, especially on:

- Precision
- Faithfulness
- Clean retrieval

Medium & Large were okay on word-matching (ROUGE/BLEU) but slightly worse on staying real and adding noise.

**Final tip:** Go with 250/50 splitting to cut hallucinations and get top retrieval



---

