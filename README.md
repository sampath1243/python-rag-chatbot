# Python RAG Chatbot

A Python-based Retrieval-Augmented Generation (RAG) chatbot that answers user questions using information retrieved from a predefined knowledge base.

## Project Overview

This project demonstrates how RAG combines document processing, text embeddings, semantic search, and Large Language Models (LLMs) to generate context-aware answers.

## Technologies Used

- Python
- OpenAI API
- OpenAI Embeddings (`text-embedding-3-small`)
- NumPy
- Cosine Similarity
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
├── .env.example
├── .gitignore
├── README.md
├── requirements.txt
└── run.py
```

## How It Works

1. Load documents from the knowledge folder.
2. Split documents into smaller text chunks.
3. Generate embeddings for each chunk.
4. Store chunks and embeddings in an in-memory vector store.
5. Accept a question from the user.
6. Generate an embedding for the question.
7. Retrieve relevant chunks using cosine similarity.
8. Pass retrieved context to the LLM.
9. Generate an answer based on the retrieved information.

## Core Components

| Module | Responsibility |
|---|---|
| loader.py | Loads knowledge-base documents |
| chunker.py | Splits documents into chunks |
| embeddings.py | Generates text embeddings |
| vector_store.py | Stores vectors and performs similarity search |
| retriever.py | Retrieves relevant chunks |
| generator.py | Generates answers using an LLM |
| run.py | Runs the chatbot application |

## Knowledge Base

The current knowledge base contains:

- Python Classes and Objects
- Python OOP Concepts
- Python Decorators
- Iterators and Generators
- Python Dunder Methods

## Installation and Setup

### 1. Clone the repository

```bash
git clone https://github.com/sampath1243/python-rag-chatbot.git
cd python-rag-chatbot
```

### 2. Create a virtual environment

```bash
python -m venv venv
```

### 3. Activate the virtual environment

Windows PowerShell:

```powershell
.\venv\Scripts\Activate.ps1
```

### 4. Install dependencies

```bash
pip install -r requirements.txt
```

### 5. Configure the API key

Create a `.env` file in the project root and add your OpenAI API key:

```env
OPENAI_API_KEY=your_api_key_here
```

Never upload your actual `.env` file or API key to GitHub.

### 6. Run the application

```bash
python run.py
```

## Author

**Sampath Kumar Puttapaka**

Python Developer | Generative AI | Retrieval-Augmented Generation (RAG)

GitHub: https://github.com/sampath1243