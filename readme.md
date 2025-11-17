# 🧠 AmbedkarGPT – RAG System (Assignment 1 & 2)

A simple Retrieval-Augmented Generation (RAG) setup using:

- Local ChromaDB
- Sentence Transformers (MiniLM)
- Ollama Mistral 7B
- Pure Python (no LangChain)
- Reproducible testing for Assignment 2

This repo has two working parts:

- Assignment 1 → Basic RAG system
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
│   ├── corpus/            # 6 docs (speech1.txt ... speech6.txt)
│   ├── evaluation.py      # Auto-testing script
│   ├── test_dataset.json  # 25 questions (given)
│   ├── results/           # Logs for each test
│   ├── results_analysis.md# Deep dive on splitting methods
│   ├── plot_metrics.py    # Makes charts
│   └── plots/             # Saved charts
│
└── README.md              # This file
```

## 🟦 Assignment 1 — Basic RAG System

### ✔ What it does
- Load docs → Split text → Turn into embeddings (MiniLM)
- Store in local Chroma
- Find top matches
- Generate answers with Ollama Mistral 7B
- Ask questions via command line

### ▶️ How to Run Assignment 1

1. Install stuff:
   ```
   pip install -r requirements.txt
   ```

2. Start Ollama & get Mistral:
   ```
   ollama pull mistral
   ```

3. Run the app:
   ```
   python app.py
   ```

4. Ask questions like:
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

### 📌 Quick Run (Mock Mode) — No Ollama needed
```
cd assignment2
$env:MOCK_MODE = "1"
python evaluation.py --out mock_results.json
```

### 📌 Full Run (Real LLM)
Make sure Ollama Mistral is running.
```
cd assignment2
Remove-Item Env:\MOCK_MODE
python evaluation.py --use_mock 0 --out test_results_real.json
```

### 📊 Make Charts
```
python plot_metrics.py
```
Charts saved to `assignment2/plots/`.

### 📄 Deep Analysis
Check `assignment2/results_analysis.md` for:

- Which splitting method won
- Metric tables
- What went wrong & why
- Tips to make RAG better

## 🏆 Quick Wins

Small chunks (250/50) rocked it, especially on:

- Precision@3
- Faithfulness
- Clean retrieval

Medium & Large were okay on word-matching (ROUGE/BLEU) but worse on staying real and adding noise.

**Final tip:** Go with 250/50 splitting to cut hallucinations and get top retrieval.

## 📌 Short Answers 

### **Q: Which chunking strategy works best for the corpus?**  
The **small chunk size (250 chars, 50 overlap)** performed the best.  
It gave the highest Precision@3 (0.613) and the strongest Faithfulness score (0.625).  
Overall, small chunks retrieve more focused context and reduce hallucination.

---

### **Q: What is the system’s current accuracy score?**  
Here is the quick accuracy snapshot:

- **Hit Rate:** 0.84  
- **Precision@3:** 0.613  
- **Faithfulness:** 0.625  

These three metrics together represent the system’s practical accuracy and how grounded its answers are.

---

### **Q: What are the most common failure types?**  
The main issues observed:

1. **Retrieval noise** – especially with larger chunks, where unrelated text appears in top-k results.  
2. **Paraphrasing mismatch** – the model answers correctly but uses different wording than the ground truth, lowering ROUGE/BLEU.  
3. **Partial or unsupported statements** – answers sometimes mix correct info with details not directly supported by retrieved chunks.

---

### **Q: What specific improvements would boost performance?**  
Recommended improvements (priority order):

1. **Add a cross-encoder reranker** – retrieve top-50, then rerank. Major boost for Precision@K and faithfulness.  
2. **Use stronger embedding models** such as `all-mpnet-base-v2` or `e5` for better semantic retrieval.  
3. **Improve the prompt** so the model sticks closer to context wording and cites evidence.  
4. **Adopt hybrid retrieval (BM25 + dense)** for better lexical + semantic coverage.  
5. **Add an answer verification step** to filter out unsupported or hallucinated statements.

---

