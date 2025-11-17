import os
os.environ["ANONYMIZED_TELEMETRY"] = "false"
os.environ["CHROMA_TELEMETRY_DISABLED"] = "true"


import json
import time
import os
import subprocess
from config import TOP_K, LLM_MODEL
from pipeline import load_text, chunk_text, embed_chunks, embed_query
from vectorstore import build_vectorstore, retrieve
from utils import save_log_json

# Disable ChromaDB telemetry to remove those warnings


# ----------------------------------------------------------
# Prompt builder
# ----------------------------------------------------------
def format_prompt(question, chunks):
    """Clean English prompt for Mistral 7B"""

    context = ""
    for i, hit in enumerate(chunks):
        context += f"\n[Chunk {i+1}] {hit['content']}\n"

    prompt = f"""
You are an AI assistant. Answer the question strictly using the information provided in the context below.

If the answer is not present in the context, reply exactly with:
"Not available in the provided text."

Do not add extra knowledge. Do not guess.

Context:
{context}

Question: {question}

Provide a clear answer in 2–3 sentences.
    """

    return prompt.strip()


# LLM Generator (Mock + Real)
def generate_answer(prompt, use_mock=False):
    """LLM se answer nikalna (mock mode helpful for evaluator)"""

    if use_mock:
        # Mock output: evaluator bina Ollama ke test kar payega
        return "Mock answer (LLM disabled). Context ke base par simplified response."

    # Real Ollama call with proper encoding handling
    try:
        # Ollama call via CLI with proper encoding for Windows
        result = subprocess.run(
            ["ollama", "run", LLM_MODEL],
            input=prompt,
            text=True,
            capture_output=True,
            timeout=60,
            encoding='utf-8',  # Force UTF-8 encoding
            errors='ignore'    # Ignore encoding errors
        )

        if result.returncode != 0:
            return f"Ollama error: {result.stderr}"
        
        return result.stdout.strip()

    except subprocess.TimeoutExpired:
        return "LLM timeout: Took too long to generate response."
    except Exception as e:
        return f"LLM error: {str(e)}"


# ----------------------------------------------------------
# Ask() – Full RAG pipeline
# ----------------------------------------------------------
def ask(question, top_k=TOP_K, use_mock=False):
    """End-to-end RAG process (retrieve → generate → log)"""

    # Query embedding
    query_vec = embed_query(question)

    # Retrieval
    hits = retrieve(query_vec, top_k=top_k)

    # If no relevant chunks found
    if not hits:
        answer = "No relevant information found in the document."
        # Log empty result
        log = {
            "question": question,
            "retrieved_chunks": [],
            "answer": answer,
            "timestamp": time.time()
        }
        save_log_json(log)
        return answer

    # Prompt
    prompt = format_prompt(question, hits)

    # LLM Answer
    answer = generate_answer(prompt, use_mock=use_mock)

    # Log saving
    log = {
        "question": question,
        "retrieved_chunks": hits,
        "answer": answer,
        "timestamp": time.time()
    }
    save_log_json(log)

    return answer


# ----------------------------------------------------------
# Initialize System
# ----------------------------------------------------------
def initialize_system():
    """Load document, process, and build vectorstore"""
    print("🔄 Loading and processing document...")
    
    try:
        # Step 1: Load & prepare document
        text = load_text("speech.txt")
        chunks = chunk_text(text)
        embeddings = embed_chunks(chunks)

        # Step 2: Build vectorstore
        build_vectorstore(chunks, embeddings)
        print("✅ Vector store built successfully!")
        return True
    except Exception as e:
        print(f"❌ System initialization failed: {e}")
        return False


# ----------------------------------------------------------
# CLI (simple interactive mode)
# ----------------------------------------------------------
if __name__ == "__main__":
    print("🔹 Ambedkar RAG System (Assignment 1)")
    print("Type 'exit' to quit.\n")

    # Initialize system
    if not initialize_system():
        print("⚠️  Continuing with existing vector store...")

    # Step 3: Interactive loop
    while True:
        try:
            q = input("\n❓ Enter question: ").strip()
            if q.lower() == "exit":
                break
            if not q:
                continue

            # First try with real LLM, fallback to mock if needed
            ans = ask(q, use_mock=False)
            print("\n🟩 Answer:", ans)
            
        except KeyboardInterrupt:
            print("\n\n👋 Exiting...")
            break
        except Exception as e:
            print(f"\n❌ Error: {e}")
            print("🔄 Trying with mock mode...")
            try:
                ans = ask(q, use_mock=True)
                print("\n🟩 Answer (Mock):", ans)
            except:
                print("❌ System failed completely.")