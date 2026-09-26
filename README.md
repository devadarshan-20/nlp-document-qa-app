# NLP Document Q&A App

A Transformer-based document question answering application that lets users upload a PDF and ask questions about its contents using semantic retrieval and NLP-based answer extraction.

This project uses Streamlit for the interface, SentenceTransformers for semantic matching, and Hugging Face Transformers for question answering.

## Overview

The app is designed to make it easier to search and understand large documents without manually reading through every page. Users can upload a PDF, the system extracts the text, identifies the most relevant sections, and then answers a natural-language question using a transformer-based question-answering model.

This repository demonstrates a practical workflow for:

- PDF text extraction
- Semantic chunk retrieval
- Document-aware question answering
- Lightweight, browser-based interaction using Streamlit

## Features

- PDF upload and text extraction
- Automatic document chunking
- Semantic similarity search using embeddings
- Transformer-based question answering
- Confidence score for answers
- Simple and interactive UI
- Fast local prototyping in Python

## How It Works

1. A user uploads a PDF file.
2. The app reads the document using PyPDF2.
3. Extracted text is split into manageable chunks.
4. Each chunk is converted into an embedding using SentenceTransformers.
5. The user asks a question.
6. The question is embedded and matched against document chunks using cosine similarity.
7. The most relevant chunk is passed to a Hugging Face QA pipeline.
8. The model returns the most likely answer and a confidence score.

## Tech Stack

- Python
- Streamlit
- PyPDF2
- Hugging Face Transformers
- SentenceTransformers
- PyTorch
- NumPy

## Project Structure

```text
nlp-document-qa-app/
├── app.py                 # Streamlit application logic
├── requirements.txt       # Python dependencies
├── README.md              # Project documentation
└── .venv/                 # Optional local virtual environment
```

## Requirements

The project requires Python 3.9+ and the packages listed in `requirements.txt`.

## Installation

Clone the repository:

```bash
git clone https://github.com/devadarshan-20/nlp-document-qa-app.git
cd nlp-document-qa-app
```

Create and activate a virtual environment:

```bash
python -m venv .venv
source .venv/bin/activate   # Linux/macOS
# or
.venv\Scripts\activate      # Windows
```

Install dependencies:

```bash
pip install -r requirements.txt
```

## Run the App

Start the Streamlit application:

```bash
streamlit run app.py
```

Then open the local URL displayed in the terminal in your browser.

## Usage

1. Upload a PDF document.
2. Wait for the app to extract and process the text.
3. Enter a question related to the document.
4. Review the answer and confidence score returned by the model.

## Example

A sample workflow:

- Upload a research paper or report
- Ask: "What is the main conclusion of the document?"
- The app finds the most relevant text chunk and answers based on that context

## Notes

- Model downloads may take time on first run.
- Larger PDFs may require more processing time.
- The project is best suited for lightweight document Q&A and experimentation rather than enterprise-scale retrieval systems.

## Future Improvements

Potential enhancements for the project include:

- Multi-document support
- Better chunking strategies
- RAG-style retrieval improvements
- PDF summarization
- Search history and saved answers
- Better evaluation and benchmarking

## License

This project does not currently declare a license.
