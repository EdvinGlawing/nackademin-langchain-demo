from langchain.agents import create_agent
from langchain.tools import tool

from util.models import get_model
from util.streaming_utils import STREAM_MODES, handle_stream
from util.pretty_print import get_user_input


@tool
def generate_schedule(user_input: str) -> str:
    """
    Skapar ett veckoschema baserat på användarens tillgänglighet och mål.
    """
    return f"""
Skapa ett detaljerat veckoschema baserat på följande krav:

{user_input}

Schemat ska:
- vara realistiskt
- inkludera pauser
- balansera arbete, studier och fritid
- vara uppdelat dag för dag
"""



def run():
    model = get_model()

    agent = create_agent(
        model=model,
        tools=[generate_schedule],
        system_prompt=(
            """
            <CONTEXT>
            Du är en planeringsassistent som hjälper användare att skapa scheman.
            Du har tillgång till ett verktyg för att generera scheman.
            </CONTEXT>

            <OBJECTIVE>
            Skapa ett realistiskt och balanserat schema baserat på användarens livssituation.
            Använd verktyget när användaren vill ha ett schema.
            </OBJECTIVE>

            <STYLE>
            Strukturerad och tydlig.
            </STYLE>

            <TONE>
            Praktisk och hjälpsam.
            </TONE>

            <AUDIENCE>
            Personer som vill organisera sin tid bättre.
            </AUDIENCE>

            <RESPONSE_FORMAT>
            Veckoschema (måndag–söndag)
            </RESPONSE_FORMAT>

            <USER_INPUT>
            Användaren beskriver sitt schema och behov.
            </USER_INPUT>
"""
        ),
    )

    messages = []

    while True:
        user_input = get_user_input("Beskriv din situation (skriv 'exit' för att avsluta)")

        if user_input.lower() in ["exit", "quit"]:
            break

        if not user_input.strip():
            print("Du måste skriva något.")
            continue

        messages.append({"role": "user", "content": user_input})

        process_stream = agent.stream(
            {"messages": messages},
            stream_mode=STREAM_MODES,
        )

        handle_stream(process_stream)

        messages.append({
            "role": "assistant",
            "content": "Schema genererat"
        })


if __name__ == "__main__":
    run()