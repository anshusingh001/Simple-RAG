import os
import streamlit as st
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_classic.chains.question_answering import load_qa_chain
from langchain_classic.prompts import PromptTemplate
import json
import hashlib

load_dotenv()
API_KEY = os.getenv("GOOGLE_API_KEY")
CACHE_FILE = "query_cache.json"

def load_cache() -> dict:
    if os.path.exists(CACHE_FILE):
        try:
            with open(CACHE_FILE, "r") as f:
                return dict(json.load(f))
        except json.JSONDecodeError:
            return {}
    return {}

def save_cache(cache_data):
    with open(CACHE_FILE, "w") as f:
        json.dump(cache_data, f, indent=4)

def get_cache_key(question, pdf_names):
    # Normalize inputs for consistent hashing
    q_norm = question.strip().lower()
    
    if pdf_names:
        # Sort pdf names to ensure order doesn't matter
        sorted_pdfs = sorted([str(p).strip().lower() for p in pdf_names])
        p_norm = ",".join(sorted_pdfs)
    else:
        p_norm = ""
        
    raw_key = f"{q_norm}|{p_norm}"
    return hashlib.md5(raw_key.encode("utf-8")).hexdigest()

def get_pdf_text(pdf_docs):
    print("Backend: Extracting text from PDFs...")
    text = ""
    for pdf in pdf_docs:
        reader = PdfReader(pdf)
        for page in reader.pages:
            text += page.extract_text() or ""
    print(f"Backend: Extraction complete. Total characters: {len(text)}")
    return text

def get_text_chunks(text):
    print("Backend: Splitting text into chunks...")
    splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=200)
    chunks = splitter.split_text(text)
    print(f"Backend: Created {len(chunks)} chunks.")
    return chunks

@st.cache_resource
def get_embeddings():
    print("Backend: Initializing embeddings model...")
    return GoogleGenerativeAIEmbeddings(
        model="models/gemini-embedding-001", google_api_key=API_KEY
    )

def build_vector_store(pdf_docs):
    """Build and returns the FAISS index."""
    print("Backend: Building vector store...")
    text = get_pdf_text(pdf_docs)
    chunks = get_text_chunks(text)
    embeddings = get_embeddings()
    vector_store = FAISS.from_texts(chunks, embeddings)
    print("Backend: Vector store build complete.")
    return vector_store, embeddings

@st.cache_resource
def get_llm():
    print("Backend: Initializing Gemini LLM...")
    return ChatGoogleGenerativeAI(
        model="gemini-2.5-flash", temperature=0.1, google_api_key=API_KEY
    )

@st.cache_resource
def get_chat_prompt():
    print("Backend: Creating chat prompt template...")
    prompt_template = """
    Answer the question as accurately as possible using only the provided context.
    Summarize the answer crisply and give it.
    Structure your answer with clear headings and bullet points where appropriate.
    
    Previous Conversation Context (if any):
    {chat_history}

    Current Context:
    {context}

    Question:
    {question}

    Answer:
    """
    return PromptTemplate(
        template=prompt_template, input_variables=["context", "question", "chat_history"]
    )

@st.cache_resource
def get_qa_chain():
    print("Backend: Loading QA chain...")
    llm = get_llm()
    prompt = get_chat_prompt()
    return load_qa_chain(llm, chain_type="stuff", prompt=prompt)

def get_answer(user_question, vector_store, chat_history_str, pdf_names=None):
    """Generates an answer using the provided vector store and chat history."""
    print(f"Backend: Generating answer for question: '{user_question}'")
    
    # Check cache first
    cache = load_cache()
    cache_key = get_cache_key(user_question, pdf_names)
    
    if cache_key in cache:
        print("Backend: Cache HIT. Returning cached answer.")
        return cache[cache_key]
        
    print("Backend: Cache MISS. Performing similarity search.")
    docs = vector_store.similarity_search(user_question)
    chain = get_qa_chain()
    
    response = chain.invoke(
        {
            "input_documents": docs, 
            "question": user_question,
            "chat_history": chat_history_str
        }
    )
    
    answer = response.get("output_text", "No answer returned.")
    
    # Save to cache
    cache[cache_key] = answer
    save_cache(cache)
    
    print("Backend: Answer generation complete. Cache updated.")
    return answer