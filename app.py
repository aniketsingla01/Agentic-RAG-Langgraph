from rag_pdf import load_split_pdf
from graph import build_agent_graph

import streamlit as st
from langchain_groq import ChatGroq
import tempfile

from dotenv import load_dotenv
load_dotenv()

if "show_uploader" not in st.session_state:
    st.session_state.show_uploader = True


st.set_page_config(page_title="GenAi Chatbot Project", layout="centered")
st.title("GenAi Chatbot")

llm = ChatGroq(model="openai/gpt-oss-120b", temperature=0.6)
graph = build_agent_graph(llm)

SYSTEM_PROMPT = """
You are a helpful GenAI assistant.
- Answer clearly and concisely
- Do not hallucinate unknown facts
- If information is missing, say you don't know
- Be friendly and professional
"""

with st.sidebar:
    st.title("⚙️ Controls")

    if st.button("🧹 New Chat"):
        st.session_state.messages = [
            {"role": "system", "content": SYSTEM_PROMPT}
        ]
        st.session_state.pdf_uploaded = False
        st.session_state.show_uploader = False
        if "pdf_uploader" in st.session_state:
            del st.session_state["pdf_uploader"]
        st.rerun()


if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role":"system","content": SYSTEM_PROMPT}
    ]

if "pdf_uploaded" not in st.session_state:
    st.session_state.pdf_uploaded = False

uploaded_file = st.file_uploader("Upload a PDF file", type="pdf", key="pdf_uploader")

# if uploaded_file is None:
#     st.session_state.pdf_uploaded = False

if uploaded_file and not st.session_state.pdf_uploaded:
    with st.spinner("Processing PDF..."):
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(uploaded_file.read())
            pdf_path = tmp.name
        
        chunks_count = load_split_pdf(pdf_path)
        st.success(f"PDF Processed Successfully!!! {chunks_count} chunks created.")
        st.session_state.pdf_uploaded = True

st.divider()

for msg in st.session_state.messages:
    if msg["role"] != "system":
        with st.chat_message(msg["role"]):
            st.write(msg["content"])

if not st.session_state.show_uploader:
    st.session_state.show_uploader = True


user_input = st.chat_input("Ask me Anything😼!!!")

if user_input:
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.write(user_input)

    result = graph.invoke({
        "question": user_input,
        "route": None,
        "context": None,
        "answer": None
    })

    assistant_reply = result["answer"]
    source_used = result["route"]

    st.session_state.messages.append(
        {"role": "assistant", "content": assistant_reply}
    )

    with st.chat_message("assistant"):
        st.caption(f"🧠 Source used: **{source_used}**")
        st.write(assistant_reply)
