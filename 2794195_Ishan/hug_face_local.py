import xml.etree.ElementTree as ET
import csv
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.memory import ConversationBufferMemory
from langchain_huggingface import HuggingFaceEmbeddings, HuggingFacePipeline
from transformers import pipeline

# ---------------- XML to CSV ----------------
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
    with open(cf, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(hdrs)
        w.writerows(rows)

# ---------------- CSV Load ----------------
def load_csv(cf):
    d = []
    with open(cf, "r") as f:
        rd = csv.reader(f)
        h = next(rd)
        for row in rd:
            d.append(" | ".join(row))
    return d

# ---------------- Main ----------------
xmlf = "Sample.xml"
csvf = "out.csv"
xml_to_csv(xmlf, csvf)
docs = load_csv(csvf)

# chunking
splitter = RecursiveCharacterTextSplitter(chunk_size=200, chunk_overlap=30)
all_text = "\n".join(docs)
doc_chunks = splitter.create_documents([all_text])

# ---------------- Local Embeddings ----------------
emb = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# store chunks in FAISS
vs = FAISS.from_documents(doc_chunks, emb)

# ---------------- Memory ----------------
mem = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

# ---------------- Local LLM (CPU only) ----------------
# Flan-T5 models are lightweight and run on CPU without GPU
generator = pipeline(
    "text2text-generation",
    model="google/flan-t5-base",  # you can try flan-t5-small for faster results
    device=-1,                    # force CPU
    max_new_tokens=200
)

llm = HuggingFacePipeline(pipeline=generator)

# ---------------- RAG Chain ----------------
qa = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=vs.as_retriever(search_type="similarity", search_kwargs={"k":3}),
    chain_type="stuff",
    memory=mem,
    return_source_documents=True
)

# ---------------- Chat Loop ----------------
while True:
    q = input("ask> ")
    if q.lower() == "quit": break
    res = qa({"query": q})
    print("ans:", res["result"])
    for i, d in enumerate(res["source_documents"]):
        print("src", i+1, ":", d.page_content[:120], "...")
