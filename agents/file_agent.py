from langchain.agents import create_agent
from langchain.tools import tool

from util.models import get_model
from util.streaming_utils import STREAM_MODES, handle_stream
from util.pretty_print import get_user_input



@tool
def read_file(file_path: str) -> str:
    """
    Läser innehållet i en lokal textfil.
    """
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()

        # begränsa längd
        return content[:5000]

    except Exception as e:
        return f"Kunde inte läsa filen: {str(e)}"



def run():
    model = get_model()

    agent = create_agent(
        model=model,
        tools=[read_file],
        system_prompt=(
            """
            <CONTEXT>
            Du är en assistent som kan läsa och analysera filer.
            Du har tillgång till ett verktyg för att läsa filer från datorn.
            </CONTEXT>

            <OBJECTIVE>
            Hjälp användaren att:
            - förstå innehåll i filer
            - sammanfatta text
            - förklara innehåll

            Använd verktyget när en fil behöver läsas.
            </OBJECTIVE>

            <STYLE>
            Tydlig och informativ.
            </STYLE>

            <TONE>
            Neutral och hjälpsam.
            </TONE>

            <AUDIENCE>
            Personer som arbetar med dokument eller studier.
            </AUDIENCE>

            <RESPONSE_FORMAT>
            - Sammanfattning
            - Viktiga punkter
            - Förklaring
            </RESPONSE_FORMAT>

            <USER_INPUT>
            Användaren anger en fil eller frågar om innehåll.
            </USER_INPUT>
"""
        ),
    )

    messages = []

    while True:
        user_input = get_user_input("Ange fil eller fråga (skriv 'exit' för att avsluta)")

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
            "content": "Fil analyserad"
        })


if __name__ == "__main__":
    run()