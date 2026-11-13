# from rag_core.loader import load_documents
# from rag_core.chunker import chunk_document
# from rag_core.embeddings import create_embedding


# # --------------------------------------------------
# # STEP 1: LOAD DOCUMENTS
# # --------------------------------------------------

# documents = load_documents()

# print(f"Loaded {len(documents)} documents")


# # --------------------------------------------------
# # STEP 2: CHUNK DOCUMENTS
# # --------------------------------------------------

# for document in documents:

#     chunks = chunk_document(
#         document,
#         chunk_size=500,
#         overlap=50
#     )

#     print(f"\nSOURCE: {document.source}")
#     print(f"CHUNKS: {len(chunks)}")

#     for chunk in chunks[:3]:

#         print(
#             f"\nChunk ID: {chunk.chunk_id}"
#         )

#         print(chunk.text[:200])


# # --------------------------------------------------
# # STEP 3: TEST EMBEDDING
# # --------------------------------------------------

# first_document = documents[0]

# chunks = chunk_document(
#     first_document,
#     chunk_size=500,
#     overlap=50
# )

# first_chunk = chunks[0]


# # Send one chunk to the embedding model
# vector = create_embedding(first_chunk.text)


# # --------------------------------------------------
# # STEP 4: DISPLAY EMBEDDING RESULT
# # --------------------------------------------------

# print("\n==============================")
# print("Embedding created successfully")
# print("==============================")

# print("Source:", first_chunk.source)
# print("Chunk ID:", first_chunk.chunk_id)
# print("Vector length:", len(vector))
# print("First 5 values:", vector[:5])


'''

from rag_core.loader import load_documents
from rag_core.chunker import chunk_document
from rag_core.embeddings import create_embedding
from rag_core.vector_store import VectorStore


# --------------------------------------------------
# STEP 1: LOAD DOCUMENTS
# --------------------------------------------------

documents = load_documents()

print(f"Loaded {len(documents)} documents")


# --------------------------------------------------
# STEP 2: CREATE VECTOR STORE
# --------------------------------------------------

vector_store = VectorStore()


# --------------------------------------------------
# STEP 3: CHUNK + EMBED + STORE
# --------------------------------------------------

total_chunks = 0

for document in documents:

    chunks = chunk_document(
        document,
        chunk_size=500,
        overlap=50
    )

    print(f"\nSOURCE: {document.source}")
    print(f"CHUNKS: {len(chunks)}")

    for chunk in chunks:

        # Convert chunk text into a vector
        vector = create_embedding(chunk.text)

        # Store chunk + vector
        vector_store.add(
            chunk,
            vector
        )

        total_chunks += 1


# --------------------------------------------------
# STEP 4: DISPLAY RESULT
# --------------------------------------------------

print("\n==============================")
print("Vector Store Created")
print("==============================")

print("Total chunks:", total_chunks)
print("Stored items:", vector_store.count())
'''



# from rag_core.loader import load_documents
# from rag_core.chunker import chunk_document
# from rag_core.embeddings import create_embedding
# from rag_core.vector_store import VectorStore
# from rag_core.retriever import retrieve


# # Load documents
# documents = load_documents()

# print(f"Loaded {len(documents)} documents")


# # Create vector store
# vector_store = VectorStore()


# # Create embeddings and store them
# for document in documents:

#     chunks = chunk_document(
#         document,
#         chunk_size=500,
#         overlap=50
#     )

#     for chunk in chunks:

#         vector = create_embedding(chunk.text)

#         vector_store.add(
#             chunk,
#             vector
#         )


# print("\nVector store ready!")
# print("Stored items:", vector_store.count())


# # Ask a question
# question = input("\nAsk a Python question: ")


# # Retrieve relevant chunks
# results = retrieve(
#     question,
#     vector_store,
#     top_k=3
# )


# # Display results
# print("\n==============================")
# print("RETRIEVED KNOWLEDGE")
# print("==============================")


# for index, result in enumerate(results, start=1):

#     print(f"\nResult {index}")
#     print(f"Score: {result['score']:.4f}")
#     print(f"Source: {result['chunk'].source}")
#     print(f"Chunk ID: {result['chunk'].chunk_id}")
#     print(f"Text:\n{result['chunk'].text[:500]}")





from rag_core.loader import load_documents
from rag_core.chunker import chunk_document
from rag_core.embeddings import create_embedding
from rag_core.vector_store import VectorStore
from rag_core.retriever import retrieve
from rag_core.generator import generate_answer


# ============================================================
# STEP 1: LOAD DOCUMENTS
# ============================================================

documents = load_documents()

print(f"Loaded {len(documents)} documents")


# ============================================================
# STEP 2: CREATE VECTOR STORE
# ============================================================

vector_store = VectorStore()


# ============================================================
# STEP 3: CHUNK DOCUMENTS + CREATE EMBEDDINGS
# ============================================================

for document in documents:

    chunks = chunk_document(
        document,
        chunk_size=500,
        overlap=50
    )

    for chunk in chunks:

        # Convert chunk text into an embedding vector
        vector = create_embedding(chunk.text)

        # Store chunk + vector
        vector_store.add(
            chunk,
            vector
        )


print("\nVector store ready!")
print("Stored items:", vector_store.count())


# ============================================================
# STEP 4: ASK USER A QUESTION
# ============================================================

question = input("\nAsk a Python question: ")


# ============================================================
# STEP 5: RETRIEVE RELEVANT CHUNKS
# ============================================================

results = retrieve(
    question,
    vector_store,
    top_k=3
)


# ============================================================
# STEP 6: DISPLAY RETRIEVED KNOWLEDGE
# ============================================================

print("\n==============================")
print("RETRIEVED KNOWLEDGE")
print("==============================")


for index, result in enumerate(results, start=1):

    print(f"\nResult {index}")
    print(f"Score: {result['score']:.4f}")
    print(f"Source: {result['chunk'].source}")
    print(f"Chunk ID: {result['chunk'].chunk_id}")

    print("\nText:")
    print(result["chunk"].text[:500])


# ============================================================
# STEP 7: GENERATE FINAL ANSWER USING LLM
# ============================================================

answer = generate_answer(
    question,
    results
)


# ============================================================
# STEP 8: DISPLAY FINAL ANSWER
# ============================================================

print("\n==============================")
print("FINAL ANSWER")
print("==============================")

print(answer)
