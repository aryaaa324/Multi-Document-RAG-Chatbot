from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
import os

# loading the embedding model
embeddings = HuggingFaceEmbeddings()

def vectorize_documents():
    # Check if data directory exists
    if not os.path.exists("data"):
        print("Error: 'data' directory not found!")
        return None
    
    # Check if there are PDF files in data directory
    pdf_files = [f for f in os.listdir("data") if f.endswith('.pdf')]
    if not pdf_files:
        print("Error: No PDF files found in 'data' directory!")
        return None
    
    print(f"Found PDF files: {pdf_files}")
    
    # Use PyPDFLoader instead of UnstructuredFileLoader
    loader = DirectoryLoader(
        path="data",
        glob="*.pdf",
        loader_cls=PyPDFLoader,  # Changed to PyPDFLoader
        show_progress=True
    )
    
    try:
        documents = loader.load()
        print(f"Loaded {len(documents)} documents")
        
        text_splitter = CharacterTextSplitter(
            chunk_size=2000,
            chunk_overlap=500
        )
        text_chunks = text_splitter.split_documents(documents)
        print(f"Created {len(text_chunks)} text chunks")
        
        vectordb = Chroma.from_documents(
            documents=text_chunks,
            embedding=embeddings,
            persist_directory="vector_db_dir"
        )
        
        print("Documents successfully vectorized and stored in vector_db_dir")
        return vectordb
        
    except Exception as e:
        print(f"Error during vectorization: {e}")
        return None

# Only run if this script is executed directly
if __name__ == "__main__":
    vectorize_documents()