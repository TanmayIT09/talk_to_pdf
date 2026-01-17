# talk_to_pdf_app.py
# Modular single-file Streamlit app with:
# - Chat-style UI
# - Per-file context
# - Page citations
# - Memory summarization
# - Config-based LLM switch (Ollama / Cloud-ready)

import os
import hashlib
import streamlit as st
import pdfplumber
from typing import List

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.prompts import PromptTemplate
from langchain.chains.question_answering import load_qa_chain
from langchain.chains.summarize import load_summarize_chain

# LLM / Embeddings (local)
from langchain_community.chat_models import ChatOllama
from langchain_community.embeddings import OllamaEmbeddings

# =============================
# Configuration (MODEL SWITCH)
# =============================

LLM_PROVIDER = "ollama"  # options: "ollama" | "cloud" (future)
OLLAMA_CHAT_MODEL = "llama3.1"
OLLAMA_EMBED_MODEL = "nomic-embed-text"

CHROMA_BASE_DIR = "chroma_db"
CHUNK_SIZE = 500
CHUNK_OVERLAP = 100

MAX_CHAT_TURNS = 8          # raw turns before summarization
SUMMARY_TRIGGER_TURNS = 6  # when to summarize

# =============================
# Utility Functions
# =============================

def compute_file_hash(file_bytes: bytes) -> str:
    return hashlib.md5(file_bytes).hexdigest()


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


# =============================
# Cached Resources
# =============================

@st.cache_resource(show_spinner=False)
def load_embeddings():
    if LLM_PROVIDER == "ollama":
        return OllamaEmbeddings(model=OLLAMA_EMBED_MODEL)
    raise NotImplementedError("Cloud embeddings not configured")


@st.cache_resource(show_spinner=False)
def load_llm():
    if LLM_PROVIDER == "ollama":
        return ChatOllama(model=OLLAMA_CHAT_MODEL, temperature=0.3)
    raise NotImplementedError("Cloud LLM not configured")


@st.cache_resource(show_spinner=False)
def load_text_splitter():
    return RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP
    )


@st.cache_resource(show_spinner=False)
def load_qa_chain_cached():
    prompt = PromptTemplate(
        template="""
You are a helpful assistant.
Use ONLY the provided context to answer the question.
If the answer is not in the context, say so.

Context:
{context}

Question:
{question}

Answer:
""",
        input_variables=["context", "question"]
    )
    return load_qa_chain(load_llm(), chain_type="stuff", prompt=prompt)


@st.cache_resource(show_spinner=False)
def load_summary_chain():
    return load_summarize_chain(load_llm(), chain_type="stuff")


# =============================
# Vector Store Logic
# =============================

def build_or_load_vectorstore(documents, persist_dir):
    embeddings = load_embeddings()

    if os.path.exists(persist_dir):
        return Chroma(
            embedding_function=embeddings,
            persist_directory=persist_dir
        )

    ensure_dir(persist_dir)
    db = Chroma.from_documents(
        documents=documents,
        embedding=embeddings,
        persist_directory=persist_dir
    )
    db.persist()
    return db


# =============================
# PDF Processing with Page Metadata
# =============================

def load_pdf_as_documents(uploaded_file) -> List:
    from langchain.schema import Document

    documents = []
    with pdfplumber.open(uploaded_file) as pdf:
        for i, page in enumerate(pdf.pages):
            text = page.extract_text()
            if text:
                documents.append(
                    Document(
                        page_content=text,
                        metadata={"page": i + 1}
                    )
                )
    return documents


# =============================
# Session State
# =============================

def init_session_state():
    st.session_state.setdefault("chat_history", [])
    st.session_state.setdefault("conversation_summary", "")
    st.session_state.setdefault("active_file_hash", None)
    st.session_state.setdefault("vector_db", None)


# =============================
# Memory Summarization
# =============================

def maybe_summarize_history():
    if len(st.session_state.chat_history) < SUMMARY_TRIGGER_TURNS * 2:
        return

    summary_chain = load_summary_chain()

    from langchain.schema import Document
    docs = [
        Document(page_content=f"{m['role']}: {m['content']}")
        for m in st.session_state.chat_history
    ]

    summary = summary_chain.run(docs)
    st.session_state.conversation_summary = summary
    st.session_state.chat_history = []


# =============================
# Chat Rendering & Handling
# =============================

def render_chat_history():
    if st.session_state.conversation_summary:
        with st.expander("Conversation summary so far"):
            st.write(st.session_state.conversation_summary)

    for msg in st.session_state.chat_history:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])


def format_answer_with_citations(answer: str, docs: List) -> str:
    pages = sorted({d.metadata.get("page") for d in docs if d.metadata.get("page")})
    if pages:
        citation = ", ".join(f"Page {p}" for p in pages)
        return f"{answer}\n\n---\n📌 **Source:** {citation}"
    return answer


def handle_user_query(user_query: str):
    with st.chat_message("user"):
        st.markdown(user_query)

    st.session_state.chat_history.append({"role": "user", "content": user_query})

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            docs = st.session_state.vector_db.similarity_search(user_query, k=4)
            chain = load_qa_chain_cached()

            response = chain(
                {
                    "input_documents": docs,
                    "question": f"{st.session_state.conversation_summary}\n{user_query}"
                },
                return_only_outputs=True
            )

            answer = response.get("output_text", "")
            answer = format_answer_with_citations(answer, docs)
            st.markdown(answer)

    st.session_state.chat_history.append({"role": "assistant", "content": answer})
    maybe_summarize_history()


# =============================
# Main App
# =============================

def main():
    st.set_page_config(page_title="Talk to PDF", page_icon="📄", layout="centered")
    st.title("📄 Chat with your PDF")
    st.caption("Local RAG • Page citations • Memory summarization")

    init_session_state()

    uploaded_file = st.file_uploader("Upload a PDF", type="pdf")

    if not uploaded_file:
        st.info("Upload a PDF to start chatting")
        return

    file_bytes = uploaded_file.read()
    file_hash = compute_file_hash(file_bytes)

    if st.session_state.active_file_hash != file_hash:
        st.session_state.chat_history = []
        st.session_state.conversation_summary = ""
        st.session_state.vector_db = None
        st.session_state.active_file_hash = file_hash

    persist_dir = os.path.join(CHROMA_BASE_DIR, file_hash)

    if st.session_state.vector_db is None:
        if os.path.exists(persist_dir):
            st.session_state.vector_db = Chroma(
                persist_directory=persist_dir,
                embedding_function=load_embeddings()
            )
        else:
            with st.spinner("Processing PDF (first time only)..."):
                progress = st.progress(0.0)

                progress.progress(0.2, text="Extracting text from PDF...")
                documents = load_pdf_as_documents(uploaded_file)

                progress.progress(0.5, text="Splitting text into chunks...")
                splitter = load_text_splitter()
                split_docs = splitter.split_documents(documents)

                progress.progress(0.8, text="Generating embeddings & saving vector store...")
                st.session_state.vector_db = build_or_load_vectorstore(
                    split_docs, persist_dir
                )

                progress.progress(1.0, text="Done ✅")

    st.success("PDF ready. Ask your questions 👇")

    render_chat_history()

    user_query = st.chat_input("Ask something about the PDF...")
    if user_query:
        handle_user_query(user_query)


# =============================
# Entry Point
# =============================

if __name__ == "__main__":
    main()
