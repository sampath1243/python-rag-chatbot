# Python RAG Chatbot

A Python-based Retrieval-Augmented Generation (RAG) application that answers user questions using information retrieved from a predefined knowledge base.

## Project Overview

This project implements a RAG pipeline that combines document processing, text embeddings, semantic search, and Large Language Models (LLMs) to generate context-aware answers.

## Project Structure

```text
python_rag_chatbot/
│
├── knowledge/
│   ├── class_object.txt
│   ├── OOPs Python.txt
│   └── Python_Decorators.txt
│
├── rag_core/
│   ├── __init__.py
│   ├── config.py
│   ├── loader.py
│   ├── chunker.py
│   ├── embeddings.py
│   ├── vector_store.py
│   ├── retriever.py
│   └── generator.py
│
├── tests/
│   ├── __init__.py
│   ├── test_chunker.py
│   ├── test_vector_store.py
│   └── test_retriever.py
│
├── .env.example
├── .gitignore
├── README.md
├── requirements.txt
└── run.py
```

## Technologies Used

- Python
- OpenAI API
- OpenAI Embeddings
- NumPy
- Cosine Similarity
- Retrieval-Augmented Generation (RAG)

## How It Works

1. Load documents from the knowledge folder.
2. Split documents into smaller chunks.
3. Generate embeddings for each chunk.
4. Store chunks and their embeddings in a vector store.
5. Accept a question from the user.
6. Generate an embedding for the question.
7. Retrieve the most relevant chunks using cosine similarity.
8. Pass the retrieved context to an LLM.
9. Generate and display the final answer.

## Knowledge Base

The current knowledge base contains:

- Python Classes and Objects
- Python OOP Concepts
- Python Decorators

## Installation

### 1. Clone the repository

```bash
git clone YOUR_GITHUB_REPOSITORY_URL
cd python_rag_chatbot
```

### 2. Create a virtual environment

```bash
python -m venv venv
```

### 3. Activate the environment

Windows:

```bash
venv\Scripts\activate
```

### 4. Install dependencies

```bash
pip install -r requirements.txt
```

### 5. Configure API key

Set your OpenAI API key as the OPENAI_API_KEY environment variable.

### 6. Run the application

```bash
python run.py
```

## Core Components

| Module | Responsibility |
|---|---|
| loader.py | Document loading |
| chunker.py | Text chunking |
| embeddings.py | Embedding generation |
| vector_store.py | Vector storage and similarity search |
| retriever.py | Relevant chunk retrieval |
| generator.py | LLM answer generation |
| run.py | Application execution |

## Author

**Sampath Kumar**

Python Developer | Generative AI | RAG