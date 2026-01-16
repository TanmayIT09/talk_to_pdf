import os
from PyPDF2 import PdfReader
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import CharacterTextSplitter, RecursiveCharacterTextSplitter

#Vector DB
from langchain.vectorstores import Chroma

#Embeddings
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import google.generativeai as genai
from langchain_google_genai import ChatGoogleGenerativeAI

#Chains
from langchain.chains.question_answering import load_qa_chain
from langchain.chains import RetrievalQA

# Prompts
from langchain.prompts import PromptTemplate

from dotenv import load_dotenv
import streamlit as st
import pdfplumber
import shutil

from langchain_community.chat_models import ChatOllama
from langchain_community.embeddings import OllamaEmbeddings

# Streamlit secrets support
if os.path.exists(".env"):
    from dotenv import load_dotenv
    load_dotenv()

os.getenv("GOOGLE_API_KEY")
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

def get_conversational_chain():
    # prompt_template = """
    # Answer the question as detailed as possible from the provided context, make sure to provide all the details, if the answer is not in
    # provided context just say, "answer is not available in the context", don't provide the wrong answer\n\n
    # Context:\n {context}?\n
    # Question: \n{question}\n
    # Answer:
    # """
    prompt_template = """
    You are an expert assistant. Use the following context to answer the question if it is relevant. 
    If the context is not helpful, you may also use your own knowledge to provide a complete and helpful answer.  

    Context: {context}
    Question: {question}
    Answer:
    """
    # model = ChatGoogleGenerativeAI(model="gemini-1.5-flash",
    #                              temperature=0.3)
    model = ChatOllama(
    model="llama3.1",
    temperature=0.3
    )
    prompt = PromptTemplate(template = prompt_template, input_variables = ["context", "question"])
    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)
    return chain

# Initialize Streamlit app
st.title("PDF Text Similarity Search")

# Upload PDF file
uploaded_file = st.file_uploader("Upload a PDF file", type="pdf")

if uploaded_file is not None:
    # Extract text from the PDF
    with pdfplumber.open(uploaded_file) as pdf:
        all_text = ""
        for page in pdf.pages:
            all_text += page.extract_text()

    # Split text into chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
    text_chunks = text_splitter.split_text(all_text)

    # Create Sentence Transformers embedding function
    #embeddings = GoogleGenerativeAIEmbeddings(model = "models/embedding-001")
    embeddings = OllamaEmbeddings(model="nomic-embed-text")

    # Store vector representations in Chroma DB
    vector_db_path = "./chroma_db"  # Adjust the path as needed

    if os.path.exists(vector_db_path):
        shutil.rmtree(vector_db_path)   # delete old embeddings

    #db = Chroma.from_texts(text_chunks, embedding=embeddings, persist_directory=vector_db_path)
    db = Chroma.from_texts(text_chunks, embedding=embeddings)

    # Perform similarity search (user query)
    user_query = st.text_input("Enter your query:")
    if user_query:
        similar_documents = db.similarity_search(user_query)
        # Display similar documents
        chain = get_conversational_chain()

        response = chain({"input_documents":similar_documents, "question": user_query}, return_only_outputs=True)
    
        print(response)
        st.write("Reply: ", response["output_text"])
        #st.write("Similar documents:")
        #for doc_id, score in similar_documents:
        #    st.write(f"Document ID: {doc_id}, Similarity Score: {score:.4f}")
