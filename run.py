
from rag_core.loader import load_documents
from rag_core.chunker import chunk_document
from rag_core.embeddings import create_embedding
from rag_core.vector_store import VectorStore
from rag_core.retriever import retrieve
from rag_core.generator import generate_answer


def main():

    # STEP 1: LOAD DOCUMENTS
    documents = load_documents()

    print(f"Loaded {len(documents)} documents")

    # STEP 2: CREATE VECTOR STORE
    vector_store = VectorStore()

    # STEP 3: CHUNK, EMBED AND STORE
    for document in documents:

        chunks = chunk_document(
            document,
            chunk_size=500,
            overlap=50
        )

        for chunk in chunks:

            vector = create_embedding(chunk.text)

            vector_store.add(chunk, vector)

    print("\nVector store ready!")
    print("Stored items:", vector_store.count())

    # STEP 4: INITIALIZE CONVERSATION HISTORY
    conversation_history = []

    print("\nChatbot is ready!")
    print("Type 'exit' to stop the chatbot.")

    # STEP 5: CONTINUOUS QUESTION-ANSWER LOOP
    while True:

        question = input("\nYou: ").strip()

        if question.lower() == "exit":
            print("Chatbot session ended.")
            break

        if not question:
            print("Please enter a question.")
            continue

        # STEP 6: RETRIEVE RELEVANT CHUNKS
        results = retrieve(
            question,
            vector_store,
            top_k=3
        )

        # STEP 7: DISPLAY RETRIEVED KNOWLEDGE
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

        # STEP 8: GENERATE ANSWER USING HISTORY
        answer = generate_answer(
            question,
            results,
            conversation_history
        )

        # STEP 9: DISPLAY ANSWER
        print("\n==============================")
        print("ANSWER")
        print("==============================")
        print(answer)

        # STEP 10: SAVE CONVERSATION
        conversation_history.append({
            "role": "user",
            "content": question
        })

        conversation_history.append({
            "role": "assistant",
            "content": answer
        })


if __name__ == "__main__":
    main()
