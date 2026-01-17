# 📄 Talk to PDF – Local Chatbot using Ollama + LangChain + Streamlit

A local, open-source **PDF chatbot** that lets you upload a PDF and chat with it using an LLM.
No paid APIs, no quotas — everything runs on your own machine.

---

## ✨ Features

* Chat-style UI (ChatGPT-like)
* Context retained **per PDF**
* Automatically resets context when a new PDF is uploaded
* Local vector database using **Chroma**
* Open-source LLM & embeddings via **Ollama**
* Optimized for fast Streamlit reruns

---

## 🧠 Tech Stack

* **Streamlit** – UI
* **LangChain** – orchestration
* **Ollama** – local LLM runtime
* **Llama 3.1** – chat model
* **nomic-embed-text** – embeddings
* **ChromaDB** – vector store
* **pdfplumber** – PDF text extraction

---

## 📋 Prerequisites

### 1️⃣ Python

* Python **3.9 – 3.11** recommended
  Check version:

```bash
python --version
```

---

### 2️⃣ Ollama (Required)

Ollama runs the LLM **locally**.

#### Install Ollama

👉 [https://ollama.com/download](https://ollama.com/download)

Verify installation:

```bash
ollama --version
```

---

### 3️⃣ Pull Required Models

Run these **once**:

```bash
ollama pull llama3.1
ollama pull nomic-embed-text
```

You can test Ollama:

```bash
ollama run llama3.1
```

---

## 🚀 Project Setup

### 1️⃣ Clone the Repository

```bash
git clone https://github.com/<your-username>/talk_to_pdf.git
cd talk_to_pdf
```

---

### 2️⃣ Create Virtual Environment (Recommended)

#### Windows

```bash
python -m venv venv
venv\\Scripts\\activate
```

#### macOS / Linux

```bash
python3 -m venv venv
source venv/bin/activate
```

---

### 3️⃣ Install Dependencies

```bash
pip install -r requirements.txt
```

If `requirements.txt` doesn’t exist yet, create one with:

```txt
streamlit
pdfplumber
langchain
langchain-community
chromadb
ollama
```

---

## ▶️ Run the Application

```bash
streamlit run talk_to_pdf_app.py
```

The app will open in your browser at:

```
http://localhost:8501
```

---

## 🧪 How to Use

1. Upload a PDF
2. Wait for vector indexing (first time only)
3. Ask questions in the chat input
4. Continue chatting — context is retained
5. Upload a new PDF → context automatically resets

---

## 📂 Project Structure

```
talk_to_pdf/
│
├── talk_to_pdf_app.py     # Main Streamlit app (single file)
├── chroma_db/             # Vector DB (auto-created)
├── requirements.txt
├── README.md
└── .gitignore
```

---

## ⚠️ Important Notes

### 🔹 Streamlit Cloud

* Ollama **does NOT run on Streamlit Cloud**
* This project is meant for:

  * Local laptop usage
  * Self-hosted VM / server

If you want a **cloud-deployable version**, you’ll need:

* Gemini / OpenAI / Groq instead of Ollama

(I can help you convert it.)

---

## 🧹 Cleanup (Optional)

To reset all stored embeddings:

```bash
rm -rf chroma_db
```

---

## 🛠️ Common Issues

### ❌ `Ollama not found`

* Ensure Ollama is installed
* Restart terminal after installation

### ❌ Slow first response

* First query loads model into memory
* Subsequent queries are much faster

---

## 📌 Roadmap Ideas

* Conversation summarization instead of trimming
* Multi-PDF support
* Source citations in answers
* Hybrid local + cloud LLM mode

---

## 👤 Author

Built by **Tanmay Srivastava**
Feel free to fork, improve, and share 🚀
