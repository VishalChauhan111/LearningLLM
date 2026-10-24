
from nemotron import ask_nemotron, MODEL

messages = [
    {
        "role": "system",
        "content": "You are a helpful AI assistant. "
                   "Explain answers clearly with examples."
    }
]

print(f"\nNemotron CMD Chat | {MODEL}")
print("Type 'exit' to quit.\n")

while True:
    try:
        user_input = input("You: ").strip()

        if user_input.lower() == "exit":
            print("Goodbye!")
            break

        if not user_input:
            continue

        messages.append({
            "role": "user",
            "content": user_input
        })

        try:
            answer = ask_nemotron(messages)
            print(f"\nNemotron: {answer}\n")

            messages.append({
                "role": "assistant",
                "content": answer
            })

        except Exception as error:
            messages.pop()
            print(f"Request failed: {error}")

    except (KeyboardInterrupt, EOFError):
        print("\nGoodbye!")
        break
