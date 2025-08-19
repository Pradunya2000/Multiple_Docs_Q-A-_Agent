# app.py

import os
import shutil
import streamlit as st
from datetime import datetime
from typing import List, Optional

# --- Configuration Section ---
# A4F API Configuration
A4F_API_KEY = "ddc-a4f-67b11dc36daf4f128b78e5b126b62966"
A4F_BASE_URL = "https://api.a4f.co/v1"

# Model Names
LLM_MODEL_NAME = "provider-3/gpt-4"
EMBEDDING_MODEL_NAME = "provider-2/text-embedding-3-small"

# ChromaDB Settings
# Streamlit Cloud's file system is ephemeral, so we'll create these dirs
CHROMA_DB_DIR = "chroma_db"
COLLECTION_NAME = "pdf_documents"

# Other Settings
CHUNK_SIZE = 750
CHUNK_OVERLAP = 150
UPLOAD_FOLDER = "docs"

# --- Import & Environment Setup ---
# The logic from all your files is now here, so we only need to import what's required for the whole app.
from langchain.document_loaders import PyPDFLoader, TextLoader, UnstructuredWordDocumentLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.embeddings import OpenAIEmbeddings
from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA

# Set up the API key and base URL from the config
os.environ["OPENAI_API_KEY"] = A4F_API_KEY
os.environ["OPENAI_API_BASE"] = A4F_BASE_URL

# Ensure necessary directories exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(CHROMA_DB_DIR, exist_ok=True)


# --- Core Logic Functions (from document_loader, embedding_store, qa_chain, retriever) ---

def get_embedding_function():
    return OpenAIEmbeddings(
        openai_api_key=A4F_API_KEY,
        openai_api_base=A4F_BASE_URL,
        model=EMBEDDING_MODEL_NAME
    )

def get_vectorstore():
    embedding_function = get_embedding_function()
    db = Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=embedding_function,
        persist_directory=CHROMA_DB_DIR
    )
    return db

def store_embeddings(documents, filename=None):
    embedding_function = get_embedding_function()
    db = Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=embedding_function,
        persist_directory=CHROMA_DB_DIR
    )
    if filename:
        cleaned_filename = os.path.basename(filename)
        for doc in documents:
            doc.metadata["source"] = cleaned_filename
            doc.metadata["upload_date"] = datetime.now().strftime("%Y-%m-%d")
            print(f"📦 Metadata for doc: {doc.metadata}")

    db.add_documents(documents)
    db.persist()
    return db

def load_documents_from_folder(folder_path, metadata=None):
    all_documents = []
    for file_name in os.listdir(folder_path):
        file_path = os.path.join(folder_path, file_name)
        if file_name.lower().endswith(".pdf"):
            loader = PyPDFLoader(file_path)
        elif file_name.lower().endswith(".txt"):
            loader = TextLoader(file_path)
        elif file_name.lower().endswith((".docx", ".doc")):
            loader = UnstructuredWordDocumentLoader(file_path)
        else:
            continue
        documents = loader.load()
        for doc in documents:
            doc.metadata["source"] = file_name
            if metadata:
                doc.metadata.update(metadata)
            doc.metadata["upload_date"] = str(datetime.now())
        all_documents.extend(documents)
    return all_documents

def split_documents(documents):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP
    )
    return text_splitter.split_documents(documents)

def get_retriever(source_file: str = None):
    db = get_vectorstore()
    if source_file:
        search_kwargs = {
            "k": 4,
            "filter": {"source": source_file}
        }
    else:
        search_kwargs = {"k": 6}
    retriever = db.as_retriever(
        search_type="similarity",
        search_kwargs=search_kwargs
    )
    return retriever

def get_qa_chain(source_file=None):
    retriever = get_retriever(source_file)
    llm = ChatOpenAI(
        openai_api_key=A4F_API_KEY,
        openai_api_base=A4F_BASE_URL,
        model_name=LLM_MODEL_NAME,
        temperature=0.2
    )
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True
    )
    return qa_chain

def get_all_metadata():
    db = get_vectorstore()
    results = db.get(include=["metadatas"])
    files = {}
    for metadata in results["metadatas"]:
        file = metadata.get("source", "Unknown")
        date = metadata.get("upload_date", "Unknown")
        files[file] = date
    return [{"file": f, "upload_date": d} for f, d in files.items()]

def clear_all_data():
    # 1. Delete all files from docs/
    for filename in os.listdir(UPLOAD_FOLDER):
        file_path = os.path.join(UPLOAD_FOLDER, filename)
        if os.path.isfile(file_path):
            os.remove(file_path)
    # 2. Delete all embeddings from vectorstore (by recreating empty one)
    db = get_vectorstore()
    db.delete_collection()


# --- Streamlit UI Functions ---
# These functions will now call the core logic directly.
def process_uploaded_files(uploaded_files: List[st.runtime.uploaded_file_manager.UploadedFile]):
    """Saves and processes files for ingestion."""
    if not uploaded_files:
        st.warning("Please upload at least one file to process.")
        return

    # Save the files to the local 'docs' folder first
    with st.spinner("📤 Processing and ingesting documents... This may take a moment."):
        uploaded_filenames = []
        for file in uploaded_files:
            file_location = os.path.join(UPLOAD_FOLDER, file.name)
            with open(file_location, "wb") as buffer:
                shutil.copyfileobj(file, buffer)
            uploaded_filenames.append(file.name)

        # Ingest documents
        docs = load_documents_from_folder(UPLOAD_FOLDER)
        chunks = split_documents(docs)
        store_embeddings(chunks)

    st.success(f"✅ Files processed and knowledge base updated! Added: {', '.join(uploaded_filenames)}")

def answer_questions(questions: List[str], selected_file: str):
    """Retrieves answers for a list of questions."""
    # Convert 'All Documents' to None for the retriever filter
    source_file_filter = None if selected_file == "All Documents" else selected_file

    with st.spinner("🚀 Getting answers..."):
        chain = get_qa_chain(source_file_filter)
        responses = []
        for q in questions:
            result = chain({"query": q})
            responses.append({
                "question": q,
                "answer": result["result"]
            })
    return responses

# --- Streamlit App Layout ---

st.set_page_config(
    page_title="Private PDF Q&A",
    page_icon="📖",
    layout="wide"
)

st.title("📖 Private PDF Q&A")
st.markdown("This app allows you to upload documents and ask questions about their content. Your data remains private and is not shared.")

tab1, tab2, tab3 = st.tabs(["Upload Files", "Q&A Session", "File History"])

# --- TAB 1: FILE UPLOADER ---
with tab1:
    st.header("Upload Documents to the Knowledge Base")
    st.info("💡 Add one or more files to ingest them into the system. The processed files will be used for answering your questions.")

    if "num_files" not in st.session_state:
        st.session_state.num_files = 1
    
    all_uploaded_files = []
    for i in range(st.session_state.num_files):
        uploaded_file = st.file_uploader(
            f"File {i+1}", 
            type=["pdf", "txt", "docx"],
            accept_multiple_files=False,
            key=f"file_uploader_{i}"
        )
        if uploaded_file:
            all_uploaded_files.append(uploaded_file)
            
    col1, col2 = st.columns([1, 5])
    with col1:
        if st.button("➕ Add Another File", type="secondary"):
            st.session_state.num_files += 1
            st.rerun()
    with col2:
        st.markdown("*Click to add a new slot for another file.*")

    st.markdown("---")
    
    if st.button("📤 Process All Files", type="primary"):
        process_uploaded_files(all_uploaded_files)

# --- TAB 2: Q&A SESSION ---
with tab2:
    st.header("Ask Questions about Your Documents")
    
    file_history = get_all_metadata()
    if not file_history:
        st.warning("⚠️ No files uploaded yet. Please upload documents in the 'File Uploader' tab to begin.")
    else:
        st.info("Choose a source file to narrow down your query, or select 'All Documents' to search across your entire library.")
        
        file_options = ["All Documents"] + [file['file'] for file in file_history]
        selected_file = st.selectbox(
            "**🔍 Select Source File:**",
            options=file_options,
            help="This feature allows you to perform Q&A on a single, specific document."
        )

        st.markdown("---")
        st.subheader("📝 Enter Your Questions")
        
        if "questions" not in st.session_state:
            st.session_state.questions = [""]
            
        for i, q in enumerate(st.session_state.questions):
            st.text_area(
                f"Question {i+1}", 
                value=q, 
                height=35,
                key=f"question_input_{i}"
            )
        
        if st.button("➕ Add More Question", type="secondary"):
            st.session_state.questions.append("")
            st.rerun()

        st.markdown("---")

        if st.button("🚀 Get Answers", type="primary"):
            questions_to_ask = [st.session_state.get(f"question_input_{i}", "") for i in range(len(st.session_state.questions)) if st.session_state.get(f"question_input_{i}", "").strip()]
            
            if questions_to_ask:
                responses = answer_questions(questions_to_ask, selected_file)
                st.subheader("📌 Answers:")
                for res in responses:
                    with st.expander(f"**Q:** {res['question']}", expanded=True):
                        st.success(f"**A:** {res['answer']}")
            else:
                st.warning("Please enter at least one question.")

# --- TAB 3: FILE HISTORY ---
with tab3:
    st.header("Uploaded Documents History")
    
    file_history = get_all_metadata()
    if file_history:
        st.info("Here is a list of all documents currently in the knowledge base, along with their upload dates.")
        st.dataframe(file_history, use_container_width=True)
        st.markdown("---")
        
        st.subheader("🧹 Clear All Data")
        st.warning("⚠️ **Warning:** Clicking this button will permanently delete all uploaded files and their embeddings from the system.")
        if st.button("🗑️ Clear All Data", type="secondary"):
            clear_all_data()
            st.rerun()
    else:
        st.info("No files have been uploaded yet.")