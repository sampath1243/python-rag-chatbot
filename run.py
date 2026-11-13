
from rag_core.loader import load_documents
from rag_core.chunker import chunk_document
from rag_core.embeddings import create_embedding
from rag_core.vector_store import VectorStore
from rag_core.retriever import retrieve
from rag_core.generator import generate_answer


def main():

    # ==================================================
    # STEP 1: LOAD DOCUMENTS
    # ==================================================

    documents = load_documents()

    print(f"Loaded {len(documents)} documents")


    # ==================================================
    # STEP 2: CREATE VECTOR STORE
    # ==================================================

    vector_store = VectorStore()


    # ==================================================
    # STEP 3: CHUNK DOCUMENTS + CREATE EMBEDDINGS
    # ==================================================

    for document in documents:

        chunks = chunk_document(
            document,
            chunk_size=500,
            overlap=50
        )

        for chunk in chunks:

            # Convert chunk text into an embedding vector
            vector = create_embedding(chunk.text)

            # Store chunk and vector
            vector_store.add(
                chunk,
                vector
            )


    print("\nVector store ready!")
    print("Stored items:", vector_store.count())


    # ==================================================
    # STEP 4: ASK USER A QUESTION
    # ==================================================

    question = input("\nAsk a Python question: ")


    # ==================================================
    # STEP 5: RETRIEVE RELEVANT CHUNKS
    # ==================================================

    results = retrieve(
        question,
        vector_store,
        top_k=3
    )


    # ==================================================
    # STEP 6: DISPLAY RETRIEVED KNOWLEDGE
    # ==================================================

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


    # ==================================================
    # STEP 7: GENERATE FINAL ANSWER USING LLM
    # ==================================================

    answer = generate_answer(
        question,
        results
    )


    # ==================================================
    # STEP 8: DISPLAY FINAL ANSWER
    # ==================================================

    print("\n==============================")
    print("FINAL ANSWER")
    print("==============================")

    print(answer)


# ======================================================
# APPLICATION ENTRY POINT
# ======================================================

if __name__ == "__main__":
    main()
