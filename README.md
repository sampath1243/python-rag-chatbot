# Python RAG Chatbot

A Python-based Retrieval-Augmented Generation (RAG) chatbot that answers user questions using information retrieved from a predefined Python knowledge base.

## Project Overview

This project demonstrates how Retrieval-Augmented Generation (RAG) combines document processing, text chunking, embeddings, semantic search, and Large Language Models (LLMs) to generate context-aware answers.

Instead of relying only on the model's existing knowledge, the chatbot retrieves relevant information from local documents and uses that information to generate answers.

## Technologies Used

- Python
- OpenAI API
- OpenAI Embeddings (`text-embedding-3-small`)
- NumPy
- Cosine Similarity
- python-dotenv
- Pytest
- Git and GitHub

## Project Structure

```text
python_rag_chatbot/
│
├── knowledge/
│   ├── Iterators&Generators Dunder_methods.txt
│   ├── OOps Python.txt
│   ├── Python_Decorators.txt
│   └── class_object.txt
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
│   ├── conftest.py
│   ├── test_chunker.py
│   ├── test_embeddings.py
│   ├── test_generator.py
│   ├── test_loader.py
│   ├── test_retriever.py
│   ├── test_run.py
│   └── test_vector_store.py
│
├── .env
├── .env.example
├── .gitignore
├── README.md
├── requirements.txt
└── run.py
```

## How It Works

1. **Load Documents:** Load knowledge-base documents from the `knowledge/` folder.
2. **Text Chunking:** Split documents into smaller text chunks.
3. **Generate Embeddings:** Convert each text chunk into a numerical vector using OpenAI embeddings.
4. **Store Vectors:** Store chunks and their embeddings in an in-memory vector store.
5. **Accept User Question:** Receive a question from the user.
6. **Generate Query Embedding:** Convert the user's question into an embedding.
7. **Semantic Search:** Compare the query embedding with stored vectors using cosine similarity.
8. **Retrieve Relevant Context:** Select the most relevant text chunks.
9. **Generate Answer:** Pass the retrieved context and question to the LLM to generate an answer.

## Core Components

| Module | Responsibility |
|---|---|
| `config.py` | Manages project paths and environment configuration |
| `loader.py` | Loads knowledge-base documents |
| `chunker.py` | Splits documents into smaller chunks |
| `embeddings.py` | Generates text embeddings using OpenAI |
| `vector_store.py` | Stores vectors and performs cosine similarity search |
| `retriever.py` | Retrieves relevant chunks for a user query |
| `generator.py` | Generates answers using an LLM |
| `run.py` | Runs the chatbot application |

## Knowledge Base

The current knowledge base contains information about:

- Python Classes and Objects
- Python OOP Concepts
- Python Decorators
- Iterators and Generators
- Python Dunder Methods

## Text Chunking Configuration

The project currently uses:

- Chunk Size: 500 characters
- Chunk Overlap: 50 characters

Overlapping helps preserve context between consecutive chunks.

## Installation and Setup

### 1. Clone the Repository

```bash
git clone https://github.com/sampath1243/python-rag-chatbot.git
cd python-rag-chatbot
```

### 2. Create a Virtual Environment

```bash
python -m venv venv
```

### 3. Activate the Virtual Environment

Windows PowerShell:

```powershell
.\venv\Scripts\Activate.ps1
```

### 4. Install Dependencies

```bash
pip install -r requirements.txt
```

### 5. Configure the OpenAI API Key

Create a `.env` file in the project root directory.

Add your API key:

```env
OPENAI_API_KEY=your_api_key_here
```

**Important:** Never upload your actual `.env` file or API key to GitHub. The `.gitignore` file excludes `.env`.

### 6. Run the Application

```bash
python run.py
```

## Automated Testing

This project uses **Pytest** to verify individual components and application behavior.

Run all automated tests:

```bash
python -m pytest tests/ -v
```

### Test Coverage

The automated tests cover:

- Document loading
- Text chunking
- Embedding generation
- Vector storage
- Cosine similarity search
- Document retrieval
- Answer generation
- Main application behavior

**Latest Test Result:** 26 tests passed.

OpenAI API calls are mocked during automated testing, so API credits are not required to run the test suite.

## Current Project Status

- Document loading implemented
- Text chunking implemented
- OpenAI embedding integration implemented
- In-memory vector store implemented
- Cosine similarity search implemented
- Relevant document retrieval implemented
- LLM answer generation implemented
- Automated tests implemented

## Author

**Sampath Kumar Puttapaka**

Python Developer | Generative AI | Retrieval-Augmented Generation (RAG)

GitHub: https://github.com/sampath1243

GitHub Repository: https://github.com/sampath1243/python-rag-chatbot