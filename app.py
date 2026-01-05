import warnings
warnings.filterwarnings("ignore")

from dotenv import load_dotenv
import os
import streamlit as st
from PyPDF2 import PdfReader
from langchain_groq import ChatGroq
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings 
from langchain_community.vectorstores import FAISS
from langchain.chains import ConversationalRetrievalChain

load_dotenv()
# function of loading text from the file
def load_pdf(file):
  reader = PdfReader(file)
  text = ""

  for page in reader.pages:
    text += page.extract_text()

  return text
st.set_page_config(page_title="AskMyPDF - Intelligent PDF Question Answering System")
st.title("My Pdf Question Answering App ;)")

uploaded_file = st.file_uploader("Upload your PDF", type="pdf")

if uploaded_file:
  pdf_text = load_pdf(uploaded_file)
  text_splitter = CharacterTextSplitter(chunk_size = 500, chunk_overlap = 50)
  chunks = text_splitter.split_text(pdf_text)
  embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
  docsearch = FAISS.from_texts(chunks, embeddings)
  llm = ChatGroq(
    api_key=os.getenv("GROQ_API_KEY"),
    model_name="llama-3.3-70b-versatile",
    temperature=0,
    max_tokens=256
  )
  qa_chain = ConversationalRetrievalChain.from_llm(llm = llm,retriever = docsearch.as_retriever())
  st.success("PDF uploaded successfully\n--------------------\nQ&A System is ready ;)")

  if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

  question = st.text_input("Ask a question about the PDF!")

  if question:
    result = qa_chain(
      {
        "question": question, 
        "chat_history": st.session_state.chat_history
      })
    st.session_state.chat_history.append((question, result["answer"]))
    st.write("**Answer**", result["answer"])