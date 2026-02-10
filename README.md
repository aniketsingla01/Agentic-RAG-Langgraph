# Agentic RAG with LangGraph

An agentic Retrieval-Augmented Generation (RAG) chatbot built using LangGraph, Streamlit, and LLMs, capable of autonomous routing across PDFs, web search, Wikipedia, ArXiv, and direct reasoning.

## Features
- 📄 PDF-based question answering using vector search
- 🤖 Agentic routing with LangGraph
- 🌐 External tools: Wikipedia, ArXiv, Web Search
- 🧠 Autonomous decision-making (PDF / Tool / LLM-only)
- 💬 Interactive Streamlit chat UI
- 🔐 Secure deployment using environment variables

## Architecture Overview
1. User submits a question
2. Router decides best source (PDF / Tool / LLM)
3. LangGraph executes the selected path
4. Context is retrieved (if needed)
5. LLM generates a grounded response

## Run Locally
git clone https://github.com/aniketsingla01/agentic-rag-langgraph.git
cd agentic-rag-langgraph
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt

##create .env
GROQ_API_KEY=your_key
PINECONE_API_KEY=your_key
PINECONE_INDEX_NAME=your_index

##run the code
streamlit run app.py

##Project Deploy Link
https://agentic-rag-langgraph-mqmy2wmfeawxat3dqpy5kv.streamlit.app/
