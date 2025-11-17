# assignment2/evaluation.py
import os
import json
import argparse
import time
from glob import glob
from collections import defaultdict, OrderedDict

import numpy as np
import nltk
nltk.download("punkt", quiet=True)
from nltk.tokenize import word_tokenize
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction

from rouge_score import rouge_scorer

# reuse your project modules
# make sure Python path includes project root; if running from assignment2/, adjust accordingly
import sys
sys.path.append("..")  # allow importing pipeline.py, vectorstore.py, app.py in project root

from pipeline import chunk_text, embed_chunks, embed_query, load_text
from vectorstore import build_vectorstore, retrieve, clear_collection, get_collection
from app import format_prompt, generate_answer

# ---------------------------------------------------------
# Helpers: metrics implementations
# ---------------------------------------------------------
def hit_rate_per_question(retrieved_sources, gold_sources):
    """retrieved_sources: list of source filenames in rank order"""
    gold = set(gold_sources or [])
    for src in retrieved_sources:
        if src in gold:
            return 1
    return 0

def reciprocal_rank(retrieved_sources, gold_sources):
    gold = set(gold_sources or [])
    for i, src in enumerate(retrieved_sources, start=1):
        if src in gold:
            return 1.0 / i
    return 0.0

def precision_at_k(retrieved_sources, gold_sources, k):
    gold = set(gold_sources or [])
    topk = retrieved_sources[:k]
    if k == 0:
        return 0.0
    return sum(1 for s in topk if s in gold) / float(k)

# ROUGE-L using rouge_scorer
rouge = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
smooth_fn = SmoothingFunction().method1

def rouge_l_score(hypothesis, reference):
    if not hypothesis or not reference:
        return 0.0
    sc = rouge.score(reference, hypothesis)
    # return fmeasure (between 0 and 1)
    return sc['rougeL'].fmeasure

def bleu_score(hypothesis, reference):
    # simple sentence-level BLEU with smoothing
    if not hypothesis or not reference:
        return 0.0
    ref_tokens = [word_tokenize(reference.lower())]
    hyp_tokens = word_tokenize(hypothesis.lower())
    try:
        score = sentence_bleu(ref_tokens, hyp_tokens, smoothing_function=smooth_fn)
    except Exception:
        score = 0.0
    return float(score)

def cosine_sim(a, b):
    if a is None or b is None:
        return 0.0
    a = np.array(a, dtype=float)
    b = np.array(b, dtype=float)
    if np.linalg.norm(a) == 0 or np.linalg.norm(b) == 0:
        return 0.0
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))

# Faithfulness heuristic:
# compute cosine between generated answer embedding and concatenated retrieved chunks embedding
def faithfulness_score(answer_emb, retrieved_chunk_embs):
    if not retrieved_chunk_embs:
        return 0.0
    # average retrieved chunk embeddings (they are normalized in pipeline)
    avg = np.mean(np.array(retrieved_chunk_embs, dtype=float), axis=0)
    return cosine_sim(answer_emb, avg)

# Answer relevance: cosine between answer and ground truth embeddings
def answer_relevance(answer_emb, gt_emb):
    return cosine_sim(answer_emb, gt_emb)

# Index building helper (keeps mapping: chunk_id -> source doc)
def build_index_from_corpus(corpus_dir, chunk_size, overlap):
    file_list = sorted(glob(os.path.join(corpus_dir, "*.txt")))
    if not file_list:
        raise FileNotFoundError(f"No .txt files found in {corpus_dir}")

    all_chunks = []
    chunk_meta = []  # same order - each entry: {"source": filename, "local_chunk_idx": i}
    for fpath in file_list:
        text = load_text(fpath)
        # chunk_text uses config defaults if none passed; we pass explicit
        chunks = chunk_text(text, chunk_size=chunk_size, overlap=overlap)
        for i, c in enumerate(chunks):
            all_chunks.append(c)
            chunk_meta.append({
                "source": os.path.basename(fpath),
                "local_chunk_idx": i
            })

    # embeddings
    embeddings = embed_chunks(all_chunks)  # normalized vectors

    # Build vectorstore (this uses vectorstore.build_vectorstore, which generates ids chunk_0..)
    # But we must clear previous index first
    try:
        coll = get_collection()
        clear_collection(coll)
    except Exception:
        pass

    build_vectorstore(all_chunks, embeddings)

    # build mapping id -> source based on order (vecstore assigned chunk_0 .. chunk_n)
    id_map = {}
    for idx, meta in enumerate(chunk_meta):
        id_map[f"chunk_{idx}"] = meta["source"]

    return all_chunks, chunk_meta, id_map, embeddings

# ---------------------------------------------------------
# Main evaluation loop
# ---------------------------------------------------------
def run_evaluation(corpus_dir, test_dataset_path, out_path, chunk_sizes, top_k=3, use_mock=True, persist_prefix="assignment2"):
    # load test dataset
    with open(test_dataset_path, "r", encoding="utf-8") as f:
        tests = json.load(f)["test_questions"]

    results_all = OrderedDict()
    os.makedirs("results", exist_ok=True)

    for label, (chunk_size, overlap) in chunk_sizes.items():
        print(f"\n--- Running experiment: {label} (chunk={chunk_size}, overlap={overlap}) ---")
        start_time = time.time()

        # Build index for this experiment
        chunks, chunk_meta, id_map, chunk_embeddings = build_index_from_corpus(corpus_dir, chunk_size, overlap)

        per_question = []
        metrics_acc = {
            "hit_count": 0,
            "mrr_sum": 0.0,
            "precision_at_k_sum": 0.0,
            "rouge_l_sum": 0.0,
            "bleu_sum": 0.0,
            "cosine_sum": 0.0,
            "relevance_sum": 0.0,
            "faithfulness_sum": 0.0,
            "n": 0
        }

        # Precompute embeddings for ground truths if present
        gt_embeddings = {}
        for item in tests:
            if item.get("ground_truth"):
                gt_embeddings[item["id"]] = embed_query(item["ground_truth"])

        # For each test question:
        for item in tests:
            qid = item["id"]
            question = item["question"]
            ground_truth = item.get("ground_truth", "")
            gold_sources = item.get("source_documents", [])

            # query embedding
            q_emb = embed_query(question)

            # retrieve top_k
            hits = retrieve(q_emb, top_k=top_k)

            # map hits -> sources (using id_map built earlier)
            retrieved_sources = []
            retrieved_chunk_embs = []
            retrieved_info = []
            for rank, h in enumerate(hits, start=1):
                hid = h.get("id")
                src = id_map.get(hid, None)
                retrieved_sources.append(src)
                # find embedding for this chunk index
                try:
                    idx = int(hid.replace("chunk_", ""))
                    retrieved_chunk_embs.append(chunk_embeddings[idx])
                except Exception:
                    retrieved_chunk_embs.append(None)
                retrieved_info.append({
                    "id": hid,
                    "content": h.get("content"),
                    "score": h.get("score"),
                    "source": src,
                    "rank": rank
                })

            # build prompt
            prompt = format_prompt(question, retrieved_info)

            # generate answer (mock or real)
            if use_mock:
                # mock return: easiest is to return ground truth when available,
                # otherwise return the first retrieved chunk content (shortened)
                if ground_truth:
                    gen_answer = ground_truth
                elif retrieved_info:
                    gen_answer = retrieved_info[0]["content"][:200]
                else:
                    gen_answer = "Not available in the provided text."
            else:
                # call real LLM via your app.generate_answer(prompt, use_mock=False)
                gen_answer = generate_answer(prompt, use_mock=False)

            # compute metrics
            is_hit = bool(hit_rate_per_question(retrieved_sources, gold_sources))
            rr = reciprocal_rank(retrieved_sources, gold_sources)
            prec_k = precision_at_k(retrieved_sources, gold_sources, top_k)

            # textual metrics
            rscore = rouge_l_score(gen_answer, ground_truth)
            bscore = bleu_score(gen_answer, ground_truth)
            # embedding-based metrics
            # compute embeddings for gen_answer and ground_truth
            ans_emb = embed_query(gen_answer) if gen_answer else None
            gt_emb = gt_embeddings.get(qid, None)
            cos_sim = cosine_sim(ans_emb, gt_emb) if (ans_emb is not None and gt_emb is not None) else 0.0
            relevance = answer_relevance(ans_emb, gt_emb) if (ans_emb is not None and gt_emb is not None) else 0.0
            faith = faithfulness_score(ans_emb, [e for e in retrieved_chunk_embs if e is not None])

            # aggregate
            metrics_acc["n"] += 1
            metrics_acc["hit_count"] += int(is_hit)
            metrics_acc["mrr_sum"] += rr
            metrics_acc["precision_at_k_sum"] += prec_k
            metrics_acc["rouge_l_sum"] += rscore
            metrics_acc["bleu_sum"] += bscore
            metrics_acc["cosine_sum"] += cos_sim
            metrics_acc["relevance_sum"] += relevance
            metrics_acc["faithfulness_sum"] += faith

            per_question.append({
                "id": qid,
                "question": question,
                "ground_truth": ground_truth,
                "answerable": item.get("answerable", True),
                "retrieved": retrieved_info,
                "generated_answer": gen_answer,
                "metrics": {
                    "is_hit": is_hit,
                    "reciprocal_rank": rr,
                    "precision_at_k": prec_k,
                    "rouge_l": rscore,
                    "bleu": bscore,
                    "cosine": cos_sim,
                    "relevance": relevance,
                    "faithfulness": faith
                }
            })

        # compute aggregated metrics
        N = metrics_acc["n"] or 1
        agg = {
            "hit_rate": metrics_acc["hit_count"] / float(N),
            "mrr": metrics_acc["mrr_sum"] / float(N),
            f"precision_at_{top_k}": metrics_acc["precision_at_k_sum"] / float(N),
            "mean_rouge_l": metrics_acc["rouge_l_sum"] / float(N),
            "mean_bleu": metrics_acc["bleu_sum"] / float(N),
            "mean_cosine": metrics_acc["cosine_sum"] / float(N),
            "mean_relevance": metrics_acc["relevance_sum"] / float(N),
            "mean_faithfulness": metrics_acc["faithfulness_sum"] / float(N),
            "num_questions": N,
            "chunk_size": chunk_size,
            "overlap": overlap
        }

        exp_result = {
            "experiment": label,
            "chunk_size": chunk_size,
            "overlap": overlap,
            "top_k": top_k,
            "aggregated": agg,
            "per_question": per_question,
            "time_seconds": time.time() - start_time
        }

        # save per-experiment result
        fname = os.path.join("results", f"results_{label}.json")
        with open(fname, "w", encoding="utf-8") as fo:
            json.dump(exp_result, fo, indent=2)

        results_all[label] = agg
        print(f" -> Saved experiment results to {fname}")

    # Save merged summary
    merged = {
        "summary": results_all,
        "experiments": list(chunk_sizes.keys())
    }
    out_file = out_path or "test_results.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(merged, f, indent=2)
    print(f"\nAll experiments done. Summary written to {out_file}")


# ---------------------------------------------------------
# CLI
# ---------------------------------------------------------
def parse_chunk_sizes(arg):
    # default string format: small:250:50,medium:550:80,large:900:100
    parts = arg.split(",")
    res = {}
    for p in parts:
        name, size, overlap = p.split(":")
        res[name] = (int(size), int(overlap))
    return res

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus_dir", type=str, default="corpus", help="Corpus folder path (6 txt files)")
    parser.add_argument("--test_dataset", type=str, default="test_dataset.json", help="test dataset json")
    parser.add_argument("--out", type=str, default="test_results.json", help="output summary file")
    parser.add_argument("--top_k", type=int, default=3, help="top_k retrieval")
    parser.add_argument("--use_mock", type=int, default=1, help="1=mock mode (fast), 0=use real LLM")
    parser.add_argument("--chunk_sizes", type=str,
                        default="small:250:50,medium:550:80,large:900:100",
                        help="Comma-separated experiments name:size:overlap")
    args = parser.parse_args()

    chunk_sizes = parse_chunk_sizes(args.chunk_sizes)
    use_mock = bool(args.use_mock) or (os.getenv("MOCK_MODE", "0") == "1")

    # ensure working dir
    os.makedirs("results", exist_ok=True)

    run_evaluation(
        corpus_dir=args.corpus_dir,
        test_dataset_path=args.test_dataset,
        out_path=args.out,
        chunk_sizes=chunk_sizes,
        top_k=args.top_k,
        use_mock=use_mock
    )
