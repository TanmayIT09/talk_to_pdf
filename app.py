# talk_to_pdf_app.py
# Modular single-file Streamlit app with chat-style UI, per-file context,
# history trimming, and cold-start optimizations

import os
import hashlib
import streamlit as st
import pdfplumber
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.prompts import PromptTemplate
from langchain.chains.question_answering import load_qa_chain
from langchain_community.chat_models import ChatOllama
from langchain_community.embeddings import OllamaEmbeddings

# =============================
# Configuration
# =============================
CHROMA_BASE_DIR = "chroma_db"
MAX_HISTORY_TURNS = 6
CHUNK_SIZE = 500
CHUNK_OVERLAP = 100

# =============================
# Utility Functions
# =============================

def compute_file_hash(file_bytes: bytes) -> str:
    return hashlib.md5(file_bytes).hexdigest()


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


# =============================
# Cached Resources (Cold-start optimized)
# =============================

@st.cache_resource(show_spinner=False)
def load_embeddings():
    return OllamaEmbeddings(model="nomic-embed-text")


@st.cache_resource(show_spinner=False)
def load_llm():
    return ChatOllama(model="llama3.1", temperature=0.3)


@st.cache_resource(show_spinner=False)
def load_text_splitter():
    return RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP
    )


@st.cache_resource(show_spinner=False)
def load_qa_chain_cached():
    prompt_template = """
You are a helpful assistant.
Use the provided context to answer the question.
If the context is insufficient, say so clearly.

Context:
{context}

Question:
{question}

Answer:
"""

    prompt = PromptTemplate(
        template=prompt_template,
        input_variables=["context", "question"]
    )

    return load_qa_chain(
        load_llm(),
        chain_type="stuff",
        prompt=prompt
    )


# =============================
# Vector Store Logic
# =============================

def build_or_load_vectorstore(text_chunks, persist_dir):
    embeddings = load_embeddings()

    if os.path.exists(persist_dir):
        return Chroma(
            embedding_function=embeddings,
            persist_directory=persist_dir
        )

    ensure_dir(persist_dir)
    db = Chroma.from_texts(
        texts=text_chunks,
        embedding=embeddings,
        persist_directory=persist_dir
    )
    db.persist()
    return db


# =============================
# PDF Processing
# =============================

def extract_text_from_pdf(uploaded_file) -> str:
    text = ""
    with pdfplumber.open(uploaded_file) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text
    return text


# =============================
# Session State Initialization
# =============================

def init_session_state():
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    if "active_file_hash" not in st.session_state:
        st.session_state.active_file_hash = None

    if "vector_db" not in st.session_state:
        st.session_state.vector_db = None


# =============================
# Chat Rendering Logic
# =============================

def render_chat_history():
    for msg in st.session_state.chat_history:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])


def handle_user_query(user_query: str):
    # Display user message immediately
    with st.chat_message("user"):
        st.markdown(user_query)

    st.session_state.chat_history.append({
        "role": "user",
        "content": user_query
    })

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            docs = st.session_state.vector_db.similarity_search(user_query, k=4)
            chain = load_qa_chain_cached()

            response = chain(
                {
                    "input_documents": docs,
                    "question": user_query
                },
                return_only_outputs=True
            )

            answer = response.get("output_text", "")
            st.markdown(answer)

    st.session_state.chat_history.append({
        "role": "assistant",
        "content": answer
    })

    # Trim history
    max_msgs = MAX_HISTORY_TURNS * 2
    if len(st.session_state.chat_history) > max_msgs:
        st.session_state.chat_history = st.session_state.chat_history[-max_msgs:]


# =============================
# Main App
# =============================

def main():
    st.set_page_config(
        page_title="Talk to PDF",
        page_icon="📄",
        layout="centered"
    )

    st.title("📄 Chat with your PDF")
    st.caption("Local, open-source | Context retained per file")

    init_session_state()

    uploaded_file = st.file_uploader("Upload a PDF", type="pdf")

    if uploaded_file is None:
        st.info("Upload a PDF to start chatting.")
        return

    file_bytes = uploaded_file.read()
    current_hash = compute_file_hash(file_bytes)

    # Reset state if new file is uploaded
    if st.session_state.active_file_hash != current_hash:
        st.session_state.chat_history = []
        st.session_state.vector_db = None
        st.session_state.active_file_hash = current_hash

    persist_dir = os.path.join(CHROMA_BASE_DIR, current_hash)

    # Build vector DB only once per file
    if st.session_state.vector_db is None:
        with st.spinner("Processing PDF and building knowledge base..."):
            pdf_text = extract_text_from_pdf(uploaded_file)
            splitter = load_text_splitter()
            chunks = splitter.split_text(pdf_text)

            st.session_state.vector_db = build_or_load_vectorstore(
                chunks,
                persist_dir
            )

    st.success("PDF ready. Ask your questions below 👇")

    # Render previous messages
    render_chat_history()

    # Chat input
    user_query = st.chat_input("Ask something about the PDF...")

    if user_query:
        handle_user_query(user_query)


# =============================
# Entry Point
# =============================

if __name__ == "__main__":
    main()
