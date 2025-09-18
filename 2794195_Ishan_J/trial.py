import pandas as pd
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chat_models import ChatOpenAI
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter

import csv
import xml.etree.ElementTree as ET


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
    fieldnames = set()
    for row in rows:
        fieldnames.update(row.keys())
    fieldnames = list(fieldnames)

    with open(filename, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def read(path):
    data = pd.read_csv(path)
    data = data.fillna("")
    texts = []

    # Row-level text representation
    for i in range(len(data)):
        row = data.iloc[i]
        parts = []
        for col in data.columns:
            parts.append(col + ": " + str(row[col]))
        content = " | ".join(parts)
        texts.append(content)

    return data, texts


def adaptive_chunk(texts, data, chunk_size=300, overlap=50):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, chunk_overlap=overlap
    )

    docs = []

    # 1. Row-level chunks (adaptive length)
    for text in texts:
        row_chunks = splitter.split_text(text)
        docs.extend([Document(page_content=chunk) for chunk in row_chunks])

    # 2. Column-level summaries (helps for column queries)
    for col in data.columns:
        col_summary = f"Column {col}: " + ", ".join(data[col].astype(str).tolist()[:50])
        # limit to first 50 values to avoid huge chunks
        col_chunks = splitter.split_text(col_summary)
        docs.extend([Document(page_content=chunk) for chunk in col_chunks])

    # 3. Global summary (very high-level view)
    global_summary = f"CSV has {len(data)} rows and {len(data.columns)} columns: {list(data.columns)}"
    docs.append(Document(page_content=global_summary))

    return docs


def vector(docs):
    embeddings = OpenAIEmbeddings()
    vectorstore = FAISS.from_documents(docs, embeddings)
    return vectorstore


def ask_question(vectorstore, question):
    llm = ChatOpenAI(
        base_url="https://cin-genailab-mass-litellm-ca.victoriousground-d739afd7.centralindia.azurecontainerapps.io",
        model="azure_ai/genailab-mass-DeepSeek-V3-0324",
        api_key="sk-V8Gh9_JfolaAlT0WkpYncw",
        temperature=0,
    )
    docs = vectorstore.similarity_search(question, k=3)
    context = "\n\n".join(doc.page_content for doc in docs)
    prompt1 = """I want to get answer from my csv files and gain insights.
    You are expert at analysing csv files.
    Answer the following question with the best of your ability in a clear
    and concise way.\n\n"""
    prompt = prompt1 + "Context:\n" + context + "\n\nQuestion: " + question

    answer = llm.invoke(prompt)
    return answer.content


def main():
    xml = parse_xml("Sample.xml")
    save(xml, "out.csv")
    path = "out.csv"
    data, texts = read(path)
    docs = adaptive_chunk(texts, data)
    vectorstore = vector(docs)

    while True:
        question = input("Ask> ")
        if question.lower() == "exit":
            break
        answer = ask_question(vectorstore, question)
        print("\nAnswer:", answer)


if __name__ == "__main__":
    main()
import pandas as pd
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chat_models import ChatOpenAI
from langchain.schema import Document

import csv
import xml.etree.ElementTree as ET

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
    fieldnames = set()
    for row in rows:
        fieldnames.update(row.keys())
    fieldnames = list(fieldnames)

    with open(filename, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)



def read(path):
    data = pd.read(path)        
    data = data.fillna("")        
    texts = []

    for i in range(len(data)):
        row = data.iloc[i]          
        parts = []
        for col in data.columns:
            parts.append(col + ": " + str(row[col]))   
        content = " | ".join(parts) 
        texts.append(content)       

    return texts

def vector(texts):
    embeddings = OpenAIEmbeddings()
    docs = [Document(page_content=text) for text in texts]
    vectorstore = FAISS.from_documents(docs, embeddings)
    return vectorstore


def ask_question(vectorstore, question):
    llm = ChatOpenAI(
        base_url = "https://cin-genailab-mass-litellm-ca.victoriousground-d739afd7.centralindia.azurecontainerapps.io",
        model = "azure_ai/genailab-mass-DeepSeek-V3-0324",
        api_key = "sk-V8Gh9_JfolaAlT0WkpYncw",
        temperature = 0,
    )
    docs = vectorstore.similarity_search(question, k=3)
    context = "\n\n".join(doc.page_content for doc in docs)
    prompt1 = """I want to get answer from my csv files and gain insights.
    You are expert at analysic csv files
    answer the following question with the best of your ability in a clear
    and consise way"""
    prompt = prompt1+question

    answer = llm.invoke(prompt)
    return answer.content

def main():
    xml = parse_xml('Sample.xml')
    save(xml, 'out.csv')
    path = "out.csv" 
    texts = read(path)
    vectorstore = vector(texts)

    while True:
        question = input()
        if question=="exit":
            break
        answer = ask_question(vectorstore, question)

main()
