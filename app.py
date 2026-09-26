import streamlit as st
import PyPDF2
import torch
from transformers import pipeline
from sentence_transformers import SentenceTransformer, util

# Page Config
st.set_page_config(
    page_title="AI Document Question Answering System",
    layout="centered"
)

st.title("AI Document Question Answering System")
st.caption("Upload a PDF and ask questions about its content.")

# Load Models
@st.cache_resource
def load_models():

    qa_pipeline = pipeline(
        task="question-answering",
        model="distilbert-base-cased-distilled-squad",
        tokenizer="distilbert-base-cased-distilled-squad"
    )

    embedder = SentenceTransformer("all-MiniLM-L6-v2")

    return qa_pipeline, embedder


qa_pipeline, embedder = load_models()

# PDF Upload
uploaded_file = st.file_uploader(
    "Upload PDF Document",
    type=["pdf"]
)

if uploaded_file is not None:

    try:
        pdf_reader = PyPDF2.PdfReader(uploaded_file)

        text = ""

        for page in pdf_reader.pages:
            page_text = page.extract_text()

            if page_text:
                text += page_text + "\n"

        if len(text.strip()) < 10:
            st.error("Unable to extract text from PDF.")
            st.stop()

        st.success("PDF successfully loaded!")

        # Split into chunks
        chunks = [
            chunk.strip()
            for chunk in text.split("\n")
            if len(chunk.strip()) > 30
        ]

        if len(chunks) == 0:
            st.error("No meaningful text found in the PDF.")
            st.stop()

        # Create embeddings
        chunk_embeddings = embedder.encode(
            chunks,
            convert_to_tensor=True
        )

        question = st.text_input(
            "Ask a question about the document:"
        )

        if question:

            question_embedding = embedder.encode(
                question,
                convert_to_tensor=True
            )

            scores = util.cos_sim(
                question_embedding,
                chunk_embeddings
            )

            best_match_idx = torch.argmax(scores).item()

            context = chunks[best_match_idx]

            result = qa_pipeline(
                question=question,
                context=context
            )

            st.subheader("Answer")
            st.write(result["answer"])

            st.subheader("Confidence Score")
            st.write(f"{result['score'] * 100:.2f}%")

            with st.expander("Context Used"):
                st.write(context)

    except Exception as e:
        st.error(f"Error: {str(e)}")
