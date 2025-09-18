import pandas as pd
import tempfile
import os
import httpx
import tiktoken
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.chat_models import ChatOpenAI
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
import csv
import xml.etree.ElementTree as ET

os.environ["TIKTOKEN_CACHE_DIR"] = "./encodings"
os.environ["OPENAI_API_KEY"] = "sk-v54iRos9P6bGLnygS7rgoQ"
client = httpx.Client(verify=False)

def parse_xml(xmlfile):
    tree = ET.parse(xmlfile)
    root = tree.getroot()
    rows = []

    for elem in root.iter():
        row = {}
        if elem.text and elem.text.strip():
            row[elem.tag] = elem.text.strip()
        for attr, value in elem.attrib.items():
            row[f"{elem.tag}_{attr}"] = value
        if row:
            rows.append(row)
    return rows


def save(rows, filename):
    names = set()
    for row in rows:
        names.update(row.keys())
    names = list(names)

    with open(filename, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=names)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def read(path):
    data = pd.read_csv(path)
    data = data.fillna("")
    texts = []
    for i in range(len(data)):
        row = data.iloc[i]
        parts = []
        for col in data.columns:
            parts.append(f"[{col}] {str(row[col])}")
        content = " | ".join(parts)
        texts.append(content)
    return texts


def chunk(texts):
    splitter = RecursiveCharacterTextSplitter(chunk_size=200, chunk_overlap=0)
    docs = []
    for text in texts:
        small_chunks = splitter.split_text(text)
        for chunk in small_chunks:
            docs.append(Document(page_content=chunk, metadata={"source": "CSV Row"}))
    return docs


def vector(docs):
    embeddings = OpenAIEmbeddings()
    vectorstore = FAISS.from_documents(docs, embeddings)
    return vectorstore


def ask_question(vectorstore, question):


    llm = ChatOpenAI(
        base_url="https://genailab.tcs.in",
        model="azure/genailab-maas-gpt-4o",
        api_key="sk-v54iRos9P6bGLnygS7rgoQ",
        http_client=client,
        temperature=0,
    )

    docs = vectorstore.similarity_search(question, k=3)
    context = "\n\n".join(doc.page_content for doc in docs)

    prompt1 = """I want to get answer from my csv files and gain insights. 
    You are expert at analysing csv files answer the 
    following question with the best of your ability in a clear and consise way
    Always mention the source of your informationn""" 
    str = """Answer question based on the context given and if a definitive answer is not present say no""" 
    prompt = prompt1+question+str+context 
    answer = llm.invoke(prompt)
    return answer.content


def main():
    xml = parse_xml("Sample.xml")
    save(xml, "out.csv")
    texts = read("out.csv")
    docs = chunk(texts)
    vectorstore = vector(docs)

    while True:
        question = input("Ask> ")
        if question == "exit":
            break
        answer = ask_question(vectorstore, question)
        print("\nAnswer:", answer)



main() 

