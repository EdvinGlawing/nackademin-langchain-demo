import argparse
import asyncio

from agents.it_helpdesk_agent import (
    build_agent,
    extract_final_answer,
    run_conversation,
)


async def interactive_chat():
    agent = await build_agent()
    history = []

    print("\nIT-helpdesk-agenten är igång.")
    print("Skriv din fråga.")
    print("Skriv 'exit' eller 'quit' för att avsluta.")
    print("Skriv 'clear' för att rensa minnet i den här sessionen.\n")

    while True:
        user_input = input("Du: ").strip()

        if not user_input:
            continue

        if user_input.lower() in {"exit", "quit"}:
            print("Avslutar.")
            break

        if user_input.lower() == "clear":
            history = []
            print("Sessionens minne är rensat.\n")
            continue

        history.append(
            {
                "role": "user",
                "content": user_input,
            }
        )

        try:
            result = await run_conversation(agent, history)
            history = result["messages"]
            answer = extract_final_answer(result)
            print(f"\nAgent:\n{answer}\n")
        except Exception as exc:
            print(f"\nFel: {exc}\n")


async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--prompt",
        type=str,
        help="Kör en enstaka prompt utan interaktivt läge",
    )
    args = parser.parse_args()

    if args.prompt:
        agent = await build_agent()
        result = await run_conversation(
            agent,
            [{"role": "user", "content": args.prompt}],
        )
        answer = extract_final_answer(result)
        print("\n=== AGENTSVAR ===\n")
        print(answer)
    else:
        await interactive_chat()


if __name__ == "__main__":
    asyncio.run(main())