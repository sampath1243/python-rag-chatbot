
import os
from openai import OpenAI


# Initialize OpenAI client
client = OpenAI(
    api_key=os.getenv("OPENAI_API_KEY")
)


def generate_answer(
    question,
    results,
    conversation_history=None
):

    # STEP 1: COLLECT RETRIEVED CONTEXT
    context_parts = []

    for result in results:
        context_parts.append(result["chunk"].text)

    context = "\n\n---\n\n".join(context_parts)

    # STEP 2: PREPARE CONVERSATION HISTORY
    history_text = ""

    if conversation_history:

        for message in conversation_history:

            history_text += (
                f"{message['role']}: "
                f"{message['content']}\n"
            )

    # STEP 3: GENERATE ANSWER
    response = client.responses.create(

        model="gpt-5.6",

        instructions=(
            "You are a Python knowledge assistant. "
            "Answer the user's question using the provided context "
            "and conversation history. "
            "Use conversation history to understand follow-up questions. "
            "If the answer is not present in the provided context, "
            "say that the information is not available."
        ),

        input=(
            f"Previous Conversation:\n{history_text}\n\n"
            f"Retrieved Context:\n{context}\n\n"
            f"Current Question:\n{question}"
        )
    )

    # STEP 4: RETURN GENERATED ANSWER
    return response.output_text
