import os

from openai import OpenAI


client = OpenAI(
    api_key=os.getenv("OPENAI_API_KEY")
)


def generate_answer(question, results):

    context_parts = []

    for result in results:
        context_parts.append(
            result["chunk"].text
        )

    context = "\n\n---\n\n".join(context_parts)

    response = client.responses.create(
        model="gpt-5.6",
        instructions=(
            "You are a Python knowledge assistant. "
            "Answer the user's question using the provided context. "
            "If the answer is not present in the context, "
            "say that the information is not available in the provided knowledge."
        ),
        input=(
            f"Context:\n{context}\n\n"
            f"Question:\n{question}"
        )
    )

    return response.output_text