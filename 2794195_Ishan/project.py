#python3.13 -m pip install langchain langchain-community langchain-openai faiss-cpu sentence_transformers

import xml.etree.ElementTree as ET
import csv
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_openai import ChatOpenAI
from langchain.chains import RetrievalQA
from langchain.memory import ConversationBufferMemory
from langchain_openai import AzureOpenAIEmbeddings

# xml to csv
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

# csv load
def load_csv(cf):
    d = []
    with open(cf, "r") as f:
        rd = csv.reader(f)
        h = next(rd)
        for row in rd:
            d.append(" | ".join(row))
    return d

# ---- main ----
xmlf = "Sample.xml"
csvf = "out.csv"
xml_to_csv(xmlf, csvf)
docs = load_csv(csvf)

# chunk
splitter = RecursiveCharacterTextSplitter(chunk_size=200, chunk_overlap=30)
all_text = "\n".join(docs)
doc_chunks = splitter.create_documents([all_text])

# embeddings + vectorstore
emb = AzureOpenAIEmbeddings(
    model="azure_ai/genailab-mass-embedding-3-small",   # replace with your embedding deployment name
    api_key="your_azure_key",
    azure_endpoint="https://cin-genailab-mass-litellm-ca.victoriousground-d739afd7.centralindia.azurecontainerapps.io"
)
vs = FAISS.from_documents(doc_chunks, emb)

# memory
mem = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

# llm
llm = ChatOpenAI(
    base_url = "https://cin-genailab-mass-litellm-ca.victoriousground-d739afd7.centralindia.azurecontainerapps.io",
    model = "azure_ai/genailab-mass-DeepSeek-V3-0324",
    api_key = "sk-V8Gh9_JfolaAlT0WkpYncw",
    temperature = 0,
)

# rag chain
qa = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=vs.as_retriever(search_type="similarity", search_kwargs={"k":3}),
    chain_type="stuff",
    memory=mem,
    return_source_documents=True
)

# chat loop
while True:
    q = input("ask> ")
    if q == "quit": break
    res = qa({"query": q})
    print("ans:", res["result"])
    # optional show retrieved
    for i, d in enumerate(res["source_documents"]):
        print("src", i+1, ":", d.page_content[:120], "...")
