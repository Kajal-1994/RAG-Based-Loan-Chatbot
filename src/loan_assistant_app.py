import os
import pickle
import numpy as np
import streamlit as st
from sentence_transformers import SentenceTransformer
from transformers import pipeline
os.environ["CUDA_VISIBLE_DEVICES"] = ""


# ---------- Helper: load CSS & HTML from files ----------


def project_root() -> str:
    """Return absolute path to project root (folder above src/)."""
    return os.path.dirname(os.path.dirname(__file__))


def load_css(path: str = "static/style.css"):
    """Load CSS from an external file.."""
    if os.path.exists(path):
        with open(path,'r', encoding= "utf-8")as f:
            css = f.read()
        st.markdown(f"<style>{css}</style>",unsafe_allow_html = True)
    else:
        st.warning("style.css not found. Using default Streamlit styles.")



def load_header_html(path: str = "templates/header.html"):
    """Load header HTML from an external file."""
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            html = f.read()
        st.markdown(html, unsafe_allow_html=True)
    else:
        st.title("SBI Loan Assistant")
        st.write("RAG-based AI chatbot for SBI loan documents.")



# ---------- 2. Cached loaders ----------

@st.cache_resource
def load_vector_store(path: str = "loan_vector_store.pkl") -> dict:
    
    """Load the pre-built vector store from disk."""
    
    full_path = os.path.join(project_root(), path)
    
    if not os.path.exists(path):
        st.error("loan_vector_store.pkl not found.\nPlease run buils_index.py file first.")
        st.stop()
    with open(full_path,"rb") as f:
        store = pickle.load(f)
        return store


@st.cache_resource
def load_embedding_model(model_name: str) -> SentenceTransformer:
    return SentenceTransformer(model_name,device = "cpu")


@st.cache_resource
def load_qa_pipeline():
     """
    Free local QA model from HuggingFace.
    Downloads once, then works offline.
     """
     return pipeline(
        task ="question-answering",
        model = "deepset/roberta-base-squad2",
        device = -1,
    )


# ========= 3. RAG Helper Functions =========


def embed_text(model:SentenceTransformer, text: str) -> np.ndarray:
    emb = model.encode(
        [text],
        convert_to_numpy = True,
        normalize_embeddings = True
    )
    return emb[0]


def retrieve_top_k(
    query_embedding: np.ndarray,
    store: dict,
    k : int = 3
) -> list[dict]:
    embeddings = store["embeddings"]
    scores = np.dot(embeddings,query_embedding)


    top_k_idx = np.argsort(scores)[-k:][::-1]

    
    results = []
    for idx in top_k_idx:
        results.append({
            "text": store["chunks"][idx],
            "source": store["metadata"][idx]["source"],
            "score": float(scores[idx]),
        })
    return results


def answer_with_qa(
        qa_pipeline,
        question: str,
        chunks: list[dict]
) -> str:
    context = "\n\n".join(ch["text"] for ch in chunks)

    
    if not context.strip():
        return "I could not find any releavant information in the SBI loan documents."
    
    result = qa_pipeline(question = question, context = context)

    answer = result.get("answer","").strip()
    score = result.get("score",0.0)


    if not answer:
        return "Sorry I could not find a clear answer in the SBI loan documets."

    return f"According to the SBI loan documents, {answer} (confidence: {score:.2f})."


# ---------- 3. Streamlit UI ----------


def main():

    st.set_page_config(page_title = "SBI Loan Assistant",page_icon="🏦", layout="centered")

    # Load external CSS and HTML header
    
    load_css()
    load_header_html()
    
    st.subheader("Loan Assistant(SBI)")
    st.write("Ask any question about SBI loan documents.The assistant will extract answers from the uploaded information..")
    
    # Load models and vector store
    
    with st.spinner("Loading models..Please wait."):
        store = load_vector_store()
        embed_model = load_embedding_model(store["model_name"])
        qa = load_qa_pipeline()

    st.success("Models loaded.You can ask question about SBI Loans.")
    
    
    # User input
     
    question = st.text_area(
        "Your question about SBI loans:",
        placeholder="Example: What is the maximum tenure for SBI home loan?",
        height=80,
    )

    top_k = st.slider(
        "Number of chunks to retrieve (Top-K):", 
        min_value=1, 
        max_value=10, 
        value=3)

    if st.button("Ask the Assistant"):
        if not question.strip():
            st.warning("Please enter a question.")
            return

        with st.spinner("Thinking based on SBI loan documents..."):
            q_emb = embed_text(embed_model, question)
            top_chunks = retrieve_top_k(q_emb, store, k=top_k)
            answer = answer_with_qa(qa, question, top_chunks)

        # Show answer
        
        st.markdown("<div class='answer-box'>", unsafe_allow_html=True)
        st.markdown("<div class='answer-title'>🧠 Assistant Response</div>", unsafe_allow_html=True)
        st.write(answer)

        # Show sources
        
        st.markdown("<div class='sources-title'>📄 Sources Used:</div>", unsafe_allow_html=True)
        sources = {c["source"] for c in top_chunks}
        if sources:
            pills_html = "".join(f"<span class='source-pill'>{src}</span>" for src in sources)
            st.markdown(pills_html, unsafe_allow_html=True)
        else:
            st.write("No sources found.")
        
        st.markdown("</div>", unsafe_allow_html=True)


if __name__ == "__main__":
    main()