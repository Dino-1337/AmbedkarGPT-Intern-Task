import time
import subprocess
from config import TOP_K, LLM_MODEL
from pipeline import load_text, chunk_text, embed_chunks, embed_query
from vectorstore import build_vectorstore, retrieve
from utils import save_log_json


def format_prompt(question, chunks):
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


def generate_answer(prompt):
    try:
        result = subprocess.run(
            ["ollama", "run", LLM_MODEL],
            input=prompt,
            text=True,
            capture_output=True,
            timeout=60,
            encoding='utf-8',
            errors='ignore'
        )

        if result.returncode != 0:
            return f"Ollama error: {result.stderr}"

        return result.stdout.strip()

    except subprocess.TimeoutExpired:
        return "LLM timeout: Took too long to generate response."
    except Exception as e:
        return f"LLM error: {str(e)}"


def ask(question, top_k=TOP_K):
    query_vec = embed_query(question)
    hits = retrieve(query_vec, top_k=top_k)

    if not hits:
        answer = "No relevant information found in the document."
        log = {
            "question": question,
            "retrieved_chunks": [],
            "answer": answer,
            "timestamp": time.time()
        }
        save_log_json(log)
        return answer

    prompt = format_prompt(question, hits)
    answer = generate_answer(prompt)

    log = {
        "question": question,
        "retrieved_chunks": hits,
        "answer": answer,
        "timestamp": time.time()
    }
    save_log_json(log)

    return answer


def initialize_system():
    print("🔄 Loading and processing document...")

    try:
        text = load_text("speech.txt")
        chunks = chunk_text(text)
        embeddings = embed_chunks(chunks)
        build_vectorstore(chunks, embeddings)
        print("✅ Vector store built successfully!")
        return True
    except Exception as e:
        print(f"❌ System initialization failed: {e}")
        return False


if __name__ == "__main__":
    print("🔹 Ambedkar RAG System (Assignment 1)")
    print("Type 'exit' to quit.\n")

    if not initialize_system():
        print("⚠️ Continuing with existing vector store...")

    while True:
        try:
            q = input("\n❓ Enter question: ").strip()
            if q.lower() == "exit":
                break
            if not q:
                continue

            ans = ask(q)
            print("\n🟩 Answer:", ans)

        except KeyboardInterrupt:
            print("\n\n👋 Exiting...")
            break
        except Exception as e:
            print(f"\n❌ Error: {e}")
            print("System failed.")
