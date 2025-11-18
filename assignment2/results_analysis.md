# Assignment 2 — Results Analysis

This report looks at three ways to split text (**Small: 250 chars/50 overlap**, **Medium: 550/80**, **Large: 900/100**) for our RAG system. We used local Chroma, MiniLM embeddings, and Ollama Mistral 7B.

## Goal
Find the **best way to split text** for getting good retrieval and answers.

## Quick Summary
**Small chunks** did the best on key metrics: top **Precision@3 (0.613)**, **Faithfulness (0.625)**, and good semantic similarity. Medium and Large chunks were better on ROUGE/BLEU but worse on staying grounded and being precise.

**My pick:** Go with **small chunks (250/50)** for the best mix.

## Metrics Table

| Metric             | Small (250/50) | Medium (550/80) | Large (900/100) |
|--------------------|----------------|-----------------|-----------------|
| Hit Rate          | 0.84          | 0.88           | 0.88           |
| MRR               | 0.753         | 0.787          | 0.793          |
| **Precision@3**   | **0.613**     | 0.493          | 0.387          |
| ROUGE-L           | 0.286         | 0.332          | 0.350          |
| BLEU              | 0.102         | 0.118          | 0.132          |
| Cosine Similarity | 0.517         | 0.560          | 0.554          |
| **Faithfulness**  | **0.625**     | 0.541          | 0.498          |

## What I Found

### **Small Chunks (250/50)**
- **Good stuff:** Top **Precision@3 (0.613)** means more useful chunks in the top 3. **Faithfulness (0.625)** keeps answers stuck to the facts, cutting down on made-up stuff. Semantic similarity (0.517) is okay.
- **Why it works:** Short chunks give focused info, so the LLM gets cleaner input and finds stuff better.
- **Downsides:** A bit lower on word matching (**ROUGE-L: 0.286**) because Mistral rephrases things.

### **Medium Chunks (550/80)**
- **Good stuff:** Best semantic similarity (**0.560**) and high hit rate (0.88), so it finds stuff well overall.
- **Bad stuff:** Lower **Precision@3 (0.493)** and **Faithfulness (0.541)** because chunks mix in extra stuff, adding noise.
- **Why:** Medium size grabs more context but makes it less relevant.

### **Large Chunks (900/100)**
- **Good stuff:** Top **ROUGE-L (0.350)** and **BLEU (0.132)** from matching words directly in long text.
- **Bad stuff:** Lowest **Precision@3 (0.387)** and **Faithfulness (0.498)**, with more risk of making stuff up.
- **Why:** Too much extra junk in chunks messes up finding and answering.

### Common Problems
- **Noise in Retrieval:** Big chunks bring in off-topic stuff, confusing the LLM and dropping faithfulness.
- **Rephrasing Effect:** Mistral's way of talking lowers ROUGE/BLEU scores, so semantic stuff (**cosine, faithfulness**) is better for checking quality.
- **Half-Relevant Stuff:** Medium chunks sometimes give incomplete info, leading to mixed answers.

## My Suggestions
- **Main one:** Use **small chunks (250/50)** for this data to get the best retrieval and grounded answers.
- **Tweak Prompts:** Change prompts to use exact words (like "Quote directly from the context") to boost ROUGE/BLEU without losing faithfulness.
- **Better Embeddings:** Try stronger models like **all-mpnet-base-v2** for closer matches and higher cosine scores.
- **Add Reranking:** Grab top-50 chunks, rerank with a cross-encoder, pick top-3—it'll really help **Precision@3**.
- **Cite Sources:** Add ways to show where answers come from for easier checking.

## How to Run
- Full test: `python evaluation.py`

## Files Made
- `test_results_real.json`: Main numbers
- `results/`: Logs for each test

## Wrap-Up
**Small chunks** give the best mix for finding and grounding. **Retrieval scores (Precision@3, Faithfulness)** matter more than word-matching ones in real RAG setups, as they show how reliable answers are in practice.


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

### Q: What improvements would boost performance?

1. **Cross-encoder reranker** – Get top-50 chunks, rerank them, and pick the best ones. This improves Precision@K and keeps answers more accurate.
2. **Better embedding models** – Using models like `mpnet` or `e5` will give better semantic matching.
3. **Stronger prompt** – Asking the model to stick closely to the context reduces mistakes.
4. **Hybrid retrieval (BM25 + dense)** – Combines keyword search and semantic search for better coverage.
5. **Note on dataset** – Our documents are very short (only one paragraph each). Because of this, some methods like reranking or hybrid retrieval **will not** show their full benefit. These improvements work much better when the dataset is bigger and more detailed.
