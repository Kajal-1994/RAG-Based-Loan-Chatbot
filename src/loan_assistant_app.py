import os
import pickle
import numpy as np
from sentence_transformers import SentenceTransformer
from transformers import pipeline

# ---------- 1. Load vector store ----------

def load_vector_store(path: str = "loan_vector_store.pkl") -> dict:
    if not os.path.exists(path):
        print("Loan_vector_store.pkl not found. Please create the vector store first.")
        raise FileNotFoundError("loan_vector_store.pkl file not found.")
    with open(path,"rb") as f:
        store = pickle.load(f)
    return store

# ---------- 2. Embeddings & retrieval ----------

def load_embedding_model(model_name: str) -> SentenceTransformer:
    print(f"loading embedding model : {model_name}")
    return SentenceTransformer(model_name)


def embed_text(model: SentenceTransformer, text: str) -> np.ndarray:
    emb  = model.encode(
        [text],
        convert_to_numpy = True,
        normalize_embeddings = True,
    )
    return emb[0]  # Return the first (and only) embedding


def retrieve_top_k(
    query_embedding: np.ndarray,
    store: dict,
    k : int = 3
) -> list[dict]:
    
    """
    Returns list of top-k chunks:
       [
        { "text": ..., "source": ..., "score": ... },
       ] 
    """ 
    embeddings = store["embeddings"]  # shape (num_chunks, embedding_dim)
    scores = np.dot(embeddings,query_embedding) # cosine similarity 

    top_k_idx = np.argsort(scores)[-k:][::-1]

    
    results = []
    for idx in top_k_idx:
        results.append({
            "text": store["chunks"][idx],
            "source": store["metadata"][idx]["source"],
            "score": float(scores[idx]),
        })

    return results


# ---------- 3. Local QA model (HuggingFace)---------- 

def load_qa_pipeline():
    """
    Loads a free, local question-answering model.
    First time it will download from HuggingFace.
    Later it works offline.

    """

    print("Loading QA Model: deepset/roberta-base-squad2..")
    qa = pipeline(("question-answering"), model="deepset/roberta-base-squad2")
    return qa

def answer_with_qa(
        qa_pipeline,
        question: str,
        chunks: list[dict]
) -> str:
    """
    Uses QA model to answer the question based on the retrieved SBI documents..
    
    """

    # Combine chunks into a single context

    context = "\n\n".join(ch["text"] for ch in chunks)

    if not context.strip():
        return "I could not find any relevant information in the SBI loan documents."
    
    result = qa_pipeline(question = question, context = context)

    answer = result.get("answer","").strip()
    score = result.get("score",0.0)

    if not answer:
        return "Sorry I could not find any relevant information in the SBI loan documents."
    
    # Wrap answer in a friendly sentence

    wrapped_answer = (
        f"According to the SBI loan documents,{answer}"
        f"(confidence: {score:.2f})."
    )
    return wrapped_answer


# ---------- 4. CLI loop ----------


def main():
    # load vector store
    try:
        store = load_vector_store("loan_vector_store.pkl")
    except FileNotFoundError:
        return
    

    # load embedding model
    
    embedding_model_name = store.get("model_name","all-MiniLM-L6-v2")
    embed_model = load_embedding_model(embedding_model_name)

   
    # load QA Pipeline

    qa = load_qa_pipeline()

    print("\n=========== SBI LOAN ASSISTANT (CLI) ================")
    print("You can ask questions about SBI loans based on the provided documents in 'docs'.")
    print("Example: 'What is the maximum tenure for SBI home loan?'\n")
    print("Type 'exit' to quit.\n")

    while True:
        user_q = input("You: ").strip()
        if not user_q:
            continue
        if user_q.lower() in ["exit","quit","q"]:
            print("Goodbye!")
            break

        # Embed query
        
        q_emb = embed_text(embed_model,user_q)

        # Retrieve most relevant chunks
        
        top_chunks = retrieve_top_k(q_emb,store, k=3)
        
        # Get answer from QA Model
       
        print("\nThinking based on SBI Loan documents...\n")
        answer = answer_with_qa(qa,user_q,top_chunks)


        # Show Answer
        print("Assistant:", answer)
        print("(Sources used:",{c["source"] for c in top_chunks},")\n")


if __name__ == "__main__":
    main()

              








