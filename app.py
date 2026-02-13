import streamlit as st
import PyPDF2
import numpy as np
import torch
from transformers import pipeline
from sentence_transformers import SentenceTransformer, util

st.set_page_config(page_title="Document QA System", layout="centered")

st.title("AI Document Question Answering System")
st.caption("Upload a PDF and ask questions about its content.")

# Load models
@st.cache_resource
def load_models():
    qa_pipeline = pipeline("question-answering")
    embedder = SentenceTransformer('all-MiniLM-L6-v2')
    return qa_pipeline, embedder

qa_pipeline, embedder = load_models()

# Upload PDF
uploaded_file = st.file_uploader("Upload PDF Document", type=["pdf"])

if uploaded_file is not None:
    pdf_reader = PyPDF2.PdfReader(uploaded_file)
    text = ""

    for page in pdf_reader.pages:
        text += page.extract_text() + "\n"

    if len(text) < 10:
        st.error("Unable to extract text from PDF.")
    else:
        st.success("PDF successfully loaded!")

        # Split text into chunks
        chunks = text.split("\n")
        chunks = [chunk for chunk in chunks if len(chunk) > 30]

        # Create embeddings
        chunk_embeddings = embedder.encode(chunks, convert_to_tensor=True)

        # Question input
        question = st.text_input("Ask a question about the document:")

        if question:
            question_embedding = embedder.encode(question, convert_to_tensor=True)

            # Find most similar chunk
            scores = util.cos_sim(question_embedding, chunk_embeddings)
            best_match_idx = torch.argmax(scores)

            context = chunks[best_match_idx]

            result = qa_pipeline(question=question, context=context)

            st.subheader("Answer")
            st.write(result['answer'])

            st.subheader("Confidence Score")
            st.write(round(result['score'] * 100, 2), "%")
