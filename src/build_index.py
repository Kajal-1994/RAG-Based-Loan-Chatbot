# Build vector index from docs

import os
import glob
import pickle 
import numpy as np
from sentence_transformers import SentenceTransformer
from pypdf import PdfReader


# ---------- 1. Load raw documents from docs/ ----------

def load_text_from_file(path: str) -> str:
    """Load text from a file based on its extension."""
    
    ext = os.path.splitext(path)[1].lower()
    
    if ext == '.txt':
        with open(path, 'r', encoding='utf-8',errors = "ignore") as f:
            return f.read()
        
    if ext == '.pdf':
        text = []
        reader = PdfReader(path)
        for page in reader.pages:
            page_text = page.extract_text() or ""
            text.append(page_text)
        return "\n".join(text)
    

    return ""  # Unsupported file type


def load_documents (docs_folder : str = "docs") -> list[dict]:
    """
      Returns: list of dicts with keys:
    - text: full document text
    - source: file name
    """
    
    patterns = ["*.txt", "*.pdf"]
    file_paths = []
    for p in patterns:
        file_paths.extend(glob.glob(os.path.join(docs_folder, p)))

    
    documents = []
    for path in file_paths:
        text = load_text_from_file(path)
        if text and text.strip():
            documents.append({
                "text": text,
                "source": os.path.basename(path)
            })
            print(f"Loaded document: {path}")
        else:
            print(f"Skipped empty or unsupported document: {path}")

    
    print(f"\nTotal documents loaded: {len(documents)}")
    return documents


# ---------- 2. Split documents into chunks ----------

def split_into_chunks(
        text: str,
        chunk_size: int = 600,
        chunk_overlap: int = 150,
   ) -> list[str]:
    

    """
    Splits long text into overlapping chunks (by characters).
    Example: 600 chars per chunk, 150 overlap."
    """

    text = text.replace("\n"," ").strip()
    chunks = []
    start = 0


    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end]
        chunks.append(chunk)
        start = end - chunk_overlap  # Overlap by better context

    return chunks


#---------- 3. Build vector store ----------


def build_vector_store(
        docs_folder : str = "Data",
        output_path : str = "loan_vector_store.pkl"

):
    """
    Main function:
    - Load docs
    - Split into chunks
    - Create embeddings
    - Save vector store with pickle
    """
    documents = load_documents(docs_folder)
    if not documents:
        print("No documents found in 'docs' folder. Please add SBI loan PDFs/TXTs and try again.")
        return
    
    print("\nLoading embedding model...")
    model = SentenceTransformer('all-MiniLM-L6-v2')

    all_chunks = []
    metadata = []

    print("\nSplitting documents into chunks...")
    for doc in documents:
        doc_chunks = split_into_chunks(doc["text"])
        for idx,chunk_text in enumerate(doc_chunks):
            all_chunks.append(chunk_text)
            metadata.append({
                "source": doc["source"],
                "chunk_id": idx,            
            })

    print(f"Total chunks: {len(all_chunks)}")


    print("\ncreating embeddings for all chunks")
    embeddings = model.encode(
        all_chunks,
        batch_size = 32, 
        show_progress_bar = True,
        convert_to_numpy = True,
        normalize_embeddings = True,
    )

    embeddings = np.array(embeddings,dtype= "float32")


    vector_store = {
        "embeddings": embeddings,
        "chunks": all_chunks,
        "metadata": metadata,
        "model_name":"all-MiniLM-L6-v2",
    }


    with open(output_path,"wb") as f:
        pickle.dump(vector_store,f)

    print(f"\nVector store saved to : {output_path}")
    print( "Index building completed✅")
    
    
if __name__ == "__main__":
    build_vector_store()
    

    
        



