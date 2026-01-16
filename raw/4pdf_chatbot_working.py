import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import google.generativeai as genai
#from langchain.vectorstores import FAISS
#from langchain_community.vectorstores import FAISS
import chromadb
#from langchain.vectorstores import Chroma
from langchain_community.vectorstores import Chroma
#from langchain.document_loaders import PyPDFLoader
from langchain_community.document_loaders import PyPDFLoader
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv
from langchain.text_splitter import CharacterTextSplitter

from pathlib import Path
import shutil

load_dotenv()
os.getenv("GOOGLE_API_KEY")
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))


def load_chunk_persist_pdf():
    pdf_folder_path = os.path.join("temp")
    documents = []
    for file in os.listdir(pdf_folder_path):
        if file.endswith('.pdf'):
            pdf_path = os.path.join(pdf_folder_path, file)
            loader = PyPDFLoader(pdf_path)
            documents.extend(loader.load())
    text_splitter = CharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    chunked_documents = text_splitter.split_documents(documents)
    
    client = chromadb.PersistentClient(path="./db")
    collection = client.get_or_create_collection(name="pdf_chat_collec")

    vectordb = Chroma.from_documents(
        documents=chunked_documents,
        embedding=GoogleGenerativeAIEmbeddings(model = "models/embedding-001"),
        persist_directory="./db"
    )
    vectordb.persist()
    return vectordb


def get_conversational_chain():
    prompt_template = """
    Answer the question as detailed as possible from the provided context and from uploaded pdf file.
    Make sure to provide all the relevant details.
    If the answer is not in provided context or in uploaded file, just say, "answer is not available in the context".
    Don't provide the wrong answer\n\n
    Context:\n {context}?\n
    Question: \n{question}\n
    Answer:
    """
    model = ChatGoogleGenerativeAI(model="gemini-pro",
                             temperature=0.3)
    prompt = PromptTemplate(template = prompt_template, input_variables = ["context", "question"])
    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)
    return chain


def main():
    st.set_page_config("PDF Based QnA")
    st.header("Chat with PDF using Google's Gemini Pro")
    temp_path=os.path.join("temp")
    if not os.path.exists(temp_path):
        os.mkdir(temp_path)
    print(f"Temp path is: {temp_path}")

    query = st.text_input("Ask a Question from the PDF Files")

    with st.sidebar:
        st.title("Menu:")      
        uploaded_file = st.file_uploader("Upload your PDF Files and Click on the Submit & Process Button", accept_multiple_files=True)
        if uploaded_file:
            print(f"Type of uploaded_file: {type(uploaded_file)}")
            for file in uploaded_file:
                with open(os.path.join("temp",file.name),"wb") as f:
                    f.write(file.getbuffer())

        if st.button("Submit & Process"):
            with st.spinner("Processing..."):
                vectordb = load_chunk_persist_pdf()
                chain = get_conversational_chain()
                matching_docs = vectordb.similarity_search(query)
                response = chain({"input_documents":matching_docs, "question": query}, return_only_outputs=True)
                print(response)
                st.write("Reply: ", response["output_text"])
            shutil.rmtree(os.path.join("temp"))
            st.success("Done")
    

if __name__ == "__main__":
    main()