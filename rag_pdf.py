from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_huggingface import HuggingFaceEmbeddings
from pinecone import Pinecone
import os
from dotenv import load_dotenv
import uuid

load_dotenv()

embeddings = HuggingFaceEmbeddings(model_name="intfloat/e5-large-v2")

pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
index = pc.Index(os.getenv("PINECONE_INDEX_NAME"))

def load_split_pdf(file_path):
    loader = PyPDFLoader(file_path)
    docs = loader.load()

    splitter = RecursiveCharacterTextSplitter(chunk_size=800,chunk_overlap=150)
    chunks = splitter.split_documents(docs)
    vectors = []
    texts = [
        f"passage: {doc.page_content}" for doc in chunks
    ]

    embeddings_list = embeddings.embed_documents(texts)

    for i, vector in enumerate(embeddings_list):
        vectors.append({
            "id": f"{uuid.uuid4()}",
            "values": vector,
            "metadata": {
                "text": chunks[i].page_content
            }
        })

    index.upsert(vectors)
    return len(vectors)

def retrieve_context(query, top_k=3):
    query_embedding = embeddings.embed_query(f"query: {query}")

    results = index.query(
        vector=query_embedding,top_k=top_k,include_metadata=True
    )

    contexts = [
        match["metadata"]["text"]
        for match in results["matches"]
    ]

    return "\n\n".join(contexts)
