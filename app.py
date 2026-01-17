# talk_to_pdf_app.py
# Single-file Streamlit app with chat-style UI, per-file context, history trimming
# Optimized for Streamlit Cloud cold starts

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

# -----------------------------
# Streamlit configuration
# -----------------------------
st.set_page_config(page_title="Talk to PDF", page_icon="📄", layout="centered")

# -----------------------------
# Constants
# -----------------------------
CHROMA_BASE_DIR = "chroma_db"  # parent dir for all files
MAX_HISTORY_TURNS = 6          # trim chat history to last N turns
CHUNK_SIZE = 500
CHUNK_OVERLAP = 100

# -----------------------------
# Utility helpers
# -----------------------------

def file_hash(file_bytes: bytes) -> str:
    """Create a stable hash for the uploaded file to isolate context per file."""
    return hashlib.md5(file_bytes).hexdigest()


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


# -----------------------------
# Cached resources (cold start optimized)
# -----------------------------

@st.cache_resource(show_spinner=False)
def load_embeddings():
    # Local, open-source embeddings (no quota issues)
    return OllamaEmbeddings(model="nomic-embed-text")


@st.cache_resource(show_spinner=False)
def load_llm():
    # Local LLM via Ollama
    return ChatOllama(model="llama3.1", temperature=0.3)


@st.cache_resource(show_spinner=False)
def get_text_splitter():
    return RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP
    )


# -----------------------------
# Vector store handling (per file)
# -----------------------------

def get_vectorstore_for_file(text_chunks, persist_dir):
    embeddings = load_embeddings()

    # If already exists, just load
    if os.path.exists(persist_dir):
        return Chroma(
            embedding_function=embeddings,
            persist_directory=persist_dir
        )

    # Else create and persist
    ensure_dir(persist_dir)
    db = Chroma.from_texts(
        texts=text_chunks,
        embedding=embeddings,
        persist_directory=persist_dir
    )
    db.persist()
    return db


# -----------------------------
# QA Chain
# -----------------------------

def get_qa_chain():
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

    llm = load_llm()
    return load_qa_chain(llm, chain_type="stuff", prompt=prompt)


# -----------------------------
# Session state initialization
# -----------------------------

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []  # list of {role, content}

if "active_file_hash" not in st.session_state:
    st.session_state.active_file_hash = None

if "vector_db" not in st.session_state:
    st.session_state.vector_db = None


# -----------------------------
# UI
# -----------------------------

st.title("📄 Chat with your PDF")
st.caption("Local, open-source | Context retained per file")

uploaded_file = st.file_uploader("Upload a PDF", type="pdf")

# -----------------------------
# Handle new file upload
# -----------------------------

if uploaded_file is not None:
    file_bytes = uploaded_file.read()
    current_hash = file_hash(file_bytes)

    # If a new file is uploaded, reset context
    if st.session_state.active_file_hash != current_hash:
        st.session_state.chat_history = []
        st.session_state.vector_db = None
        st.session_state.active_file_hash = current_hash

    persist_dir = os.path.join(CHROMA_BASE_DIR, current_hash)

    # Build vector DB only once per file
    if st.session_state.vector_db is None:
        with st.spinner("Processing PDF and building knowledge base..."):
            with pdfplumber.open(uploaded_file) as pdf:
                full_text = ""
                for page in pdf.pages:
                    text = page.extract_text()
                    if text:
                        full_text += text

            splitter = get_text_splitter()
            chunks = splitter.split_text(full_text)

            st.session_state.vector_db = get_vectorstore_for_file(
                chunks,
                persist_dir
            )

    st.success("PDF ready. Ask your questions below 👇")

    # -----------------------------
    # Chat UI
    # -----------------------------

    for msg in st.session_state.chat_history:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    user_query = st.chat_input("Ask something about the PDF...")

    if user_query:
        # Add user message
        st.session_state.chat_history.append({
            "role": "user",
            "content": user_query
        })

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                db = st.session_state.vector_db
                docs = db.similarity_search(user_query, k=4)

                chain = get_qa_chain()
                response = chain(
                    {
                        "input_documents": docs,
                        "question": user_query
                    },
                    return_only_outputs=True
                )

                answer = response.get("output_text", "")
                st.markdown(answer)

        # Add assistant message
        st.session_state.chat_history.append({
            "role": "assistant",
            "content": answer
        })

        # -----------------------------
        # Trim history (keep last N turns)
        # -----------------------------
        max_msgs = MAX_HISTORY_TURNS * 2
        if len(st.session_state.chat_history) > max_msgs:
            st.session_state.chat_history = st.session_state.chat_history[-max_msgs:]

else:
    st.info("Upload a PDF to start chatting.")
