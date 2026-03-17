from langchain.agents import create_agent
from langchain.tools import tool

import requests
import re

from util.models import get_model
from util.streaming_utils import STREAM_MODES, handle_stream
from util.pretty_print import get_user_input



@tool
def search_resources(topic: str) -> str:
    """
    Söker efter artiklar och tutorials och returnerar relevanta länkar.
    """
    try:
        url = f"https://duckduckgo.com/html/?q={topic}+tutorial"
        response = requests.get(url, timeout=5)

        html = response.text

        # hitta länkar
        links = re.findall(r'href="(https?://[^"]+)"', html)

        clean_links = []
        for link in links:
            if "duckduckgo" not in link:
                clean_links.append(link)

            if len(clean_links) >= 5:
                break

        if not clean_links:
            return f"Kunde inte hitta resurser för {topic}."

        result = f"Resurser för att lära dig {topic}:\n\n"
        for link in clean_links:
            result += f"- {link}\n"

        return result

    except Exception as e:
        return f"Fel vid webbsökning: {str(e)}"



def run():
    model = get_model()

    agent = create_agent(
        model=model,
        tools=[search_resources],
        system_prompt=(
            """
            <CONTEXT>
            Du är en studiecoach som hjälper användare att planera sina studier.
            Du har tillgång till ett verktyg för att söka efter resurser och artiklar på webben.
            </CONTEXT>

            <OBJECTIVE>
            Hjälp användaren att lära sig ett ämne genom att:
            - skapa planer
            - ge tips
            - hitta relevanta resurser online

            Använd webverktyget när användaren behöver material att läsa.
            </OBJECTIVE>

            <STYLE>
            Strukturerad och pedagogisk.
            </STYLE>

            <TONE>
            Stöttande.
            </TONE>

            <AUDIENCE>
            Studenter.
            </AUDIENCE>

            <RESPONSE_FORMAT>
            - Mål
            - Plan
            - Tips
            - Resurser (om relevant)
            </RESPONSE_FORMAT>

            <USER_INPUT>
            Användaren berättar vad den behöver hjälp med.
            </USER_INPUT>
"""
        ),
    )

    messages = []

    while True:
        user_input = get_user_input("Du")

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

        # streaming output
        handle_stream(process_stream)

        # enkel minnesplaceholder
        messages.append({
            "role": "assistant",
            "content": "Svar genererat"
        })


if __name__ == "__main__":
    run()