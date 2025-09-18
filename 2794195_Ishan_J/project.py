# python3.13 -m pip install langchain langchain-community faiss-cpu sentence-transformers requests

import os
import csv
import time
import warnings
import requests
import xml.etree.ElementTree as ET
from requests.packages.urllib3.exceptions import InsecureRequestWarning

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import embeddings
from langchain.memory import ConversationBufferMemory


warnings.simplefilter("ignore", InsecureRequestWarning)


PERPLEXITY_API_KEY = os.environ.get(
    "PERPLEXITY_API_KEY", 
    "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"
)
PERPLEXITY_URL = "https://api.perplexity.ai/chat/completions"

HEADERS = {
    "Content-Type": "application/json",
    "Authorization": f"Bearer {PERPLEXITY_API_KEY}",
}

# ----------------- Helper: Safe POST with retries -----------------
def safe_post(url, headers, json_body, retries=3, timeout=120):
    for i in range(retries):
        try:
            resp = requests.post(url, headers=headers, json=json_body, verify=False, timeout=timeout)
            resp.raise_for_status()
            return resp
        except Exception as e:
            print(f"⚠️ Request failed (attempt {i+1}/{retries}): {e}")
            time.sleep(3 * (i+1))
    return None

# ----------------- Embeddings wrapper using Perplexity -----------------
class PerplexityEmbeddings(Embeddings):
    def __init__(self, model="sonar-pro"):
        self.model = model

    def embed_documents(self, texts):
        vectors = []
        for text in texts:
            vec = self.get_embedding(text)
            vectors.append(vec)
        return vectors

    def embed_query(self, text):
        return self.get_embedding(text)

    def get_embedding(self, text):
        # Build chat prompt to get a "vector-like" response
        body = {
            "model": self.model,
            "messages": [{"role": "user", "content": f"Embed this text for semantic search: {text}"}],
        }
        resp = safe_post(PERPLEXITY_URL, HEADERS, body)
        if resp is None:
            print("⚠️ Failed to get embedding from Perplexity, returning zeros.")
            return [0.0] * 768  # fallback vector
        try:
            # Convert text to pseudo-vector by taking char codes (hacky!)
            # Because Perplexity does not return real embeddings
            emb_text = resp.json()["choices"][0]["message"]["content"]
            vec = [float(ord(c) % 256) / 255 for c in emb_text[:768]]  # 768-dim
            if len(vec) < 768:
                vec += [0.0] * (768 - len(vec))
            return vec
        except Exception as e:
            print("⚠️ Could not parse embedding:", e)
            return [0.0] * 768

# ----------------- XML -> CSV -----------------
def xml_to_csv(xf, cf):
    t = ET.parse(xf)
    r = t.getroot()
    rows, hdrs = [], []
    for ch in r:
        row = []
        if not hdrs:
            hdrs = [el.tag for el in ch]
        for el in ch:
            row.append(el.text)
        rows.append(row)
    with open(cf, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(hdrs)
        w.writerows(rows)

# ----------------- Load CSV -----------------
def load_csv(cf):
    d = []
    with open(cf, "r", encoding="utf-8") as f:
        rd = csv.reader(f)
        next(rd)  # skip header
        for row in rd:
            d.append(" | ".join(row))
    return d

# ----------------- Main -----------------
def main():
    xml_file = "Sample.xml"  # make sure it exists
    csv_file = "out.csv"

    xml_to_csv(xml_file, csv_file)
    docs = load_csv(csv_file)

    # Chunk text
    splitter = RecursiveCharacterTextSplitter(chunk_size=50, chunk_overlap=0)  # smaller chunks
    all_text = "\n".join(docs)
    doc_chunks = splitter.create_documents([all_text])

    # Create embeddings
    emb = PerplexityEmbeddings(model="sonar-pro")
    vs = FAISS.from_documents(doc_chunks, embedding=emb)

    # Conversation memory
    mem = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

    # Chat loop
    while True:
        q = input("ask> ").strip()
        if q.lower() in ("quit", "exit"):
            break

        # Retrieve top-k docs
        retriever = vs.as_retriever(search_type="similarity", search_kwargs={"k": 3})
        docs_ret = retriever.get_relevant_documents(q)
        contexts = [d.page_content for d in docs_ret]

        context_text = "\n\n".join([f"Document {i+1}:\n{c}" for i, c in enumerate(contexts)])
        prompt = f"Here are some retrieved documents:\n\n{context_text}\n\nAnswer the question using ONLY the above documents. If answer is not present, say you don't know.\n\nQuestion: {q}"

        # Include conversation memory
        system_messages = []
        mem_state = mem.load_memory()
        if mem_state and "chat_history" in mem_state:
            try:
                hist_msgs = mem_state["chat_history"]
                hist_text = "\n".join([str(m) for m in hist_msgs])
                system_messages.append({"role": "system", "content": "Conversation history:\n" + hist_text})
            except Exception:
                pass

        messages = system_messages + [{"role": "user", "content": prompt}]

        # Get answer from Perplexity
        body = {"model": "sonar-pro", "messages": messages}
        resp = safe_post(PERPLEXITY_URL, HEADERS, body)
        if resp is None:
            print("No answer returned.")
            continue

        try:
            answer = resp.json()["choices"][0]["message"]["content"]
        except Exception:
            answer = resp.text

        print("\n=== ANSWER ===")
        print(answer)
        print("\n=== SOURCES ===")
        for i, c in enumerate(contexts):
            print(f"src {i+1}: {c[:200]}...")

        try:
            mem.save_context({"input": q}, {"output": answer})
        except Exception:
            pass

if __name__ == "__main__":
    if PERPLEXITY_API_KEY.startswith("pplx-"):
        print("⚠️ Using placeholder API key; replace with your own for production!")
    main()
