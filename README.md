SBI Loan Assistant — RAG Chatbot (State Bank of India)

This project implements a Retrieval-Augmented Generation (RAG) chatbot that answers user questions about SBI loan products.
It processes official SBI loan PDFs, builds a searchable vector knowledge base, and uses a local QA model (RoBERTa) to generate accurate, context-aware, source-cited responses — completely offline, without OpenAI or paid APIs.

RAG Pipeline Overview :

PDF Documents → Text Extraction → Cleaning → Chunking → Embeddings → Vector S
                            
                             Query Embedding + Retrieval
                                                     ↓
                           Local QA Model (RoBERTa) → Final Answer


Environment Setup:

1) Create and activate environment:
python -m venv venv
venv\Scripts\activate        # Windows

2) Install dependencies:
pip install -r requirements.txt

3) Folder Structure
Data/                  → Contains SBI loan PDFs  
src/                   →Projec      source code  
    build_index.py     → Builds    vector store from PDFs  
    chat_bot.py        → Command-line chatbot  
    loan_assistant_app.py → Streamlit web app  

static/style.css       → UI styling  
templates/header.html  → Custom HTML header  
loan_vector_store.pkl  → Generated vector embeddings  

* How to Build the Vector Store:

Run this script to extract text, chunk it, and generate embeddings:

python src/build_index.py


This will create:

loan_vector_store.pkl

Run the Streamlit App
streamlit run src/loan_assistant_app.py


➡ Opens at: http://localhost:8501

You can then ask questions like:

“What is the maximum tenure for SBI home loan?”

“What are the documents required for student loan?”

* Run the CLI Chatbot:
python src/chat_bot.py


* Example:

You: What is the maximum home loan tenure?
Assistant: According to the SBI loan documents, 30 years (confidence: 0.78).

* Technologies Used:

1.Sentence-Transformers (MiniLM) — Embeddings

2.NumPy — Retrieval (Cosine similarity)

3.RoBERTa (deepset/squad2) — Question Answering model

4.Streamlit — Web Interface

5.Python, HTML, CSS


* RAG System Behavior:

1.Extracts text from PDFs

2.Splits into readable chunks

3.Generates semantic embeddings

4.Retrieves top-K relevant chunks

5.Uses QA model to extract precise answer

6.Displays sources used

* Example Workflow (Internal Logic):

1.User Question → Embed → Retrieve Relevant PDF Chunks → RoBERTa QA → Final Answer

2.Add SBI Loan PDFs

3.Place your PDF documents here:

4.Data/
    SBI Personal Loan...
    Student Loan...
    Terms-and-Conditions.pdf

* Future Enhancements:

1.Chat history

2.Multiple bank support

3.FAISS ANN index

4.PDF upload directly in UI

5.Deployment on Streamlit Cloud