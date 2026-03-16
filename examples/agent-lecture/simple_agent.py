from langchain.agents import create_agent
from langchain.tools import tool

from util.models import get_model
from util.streaming_utils import STREAM_MODES, handle_stream
from util.pretty_print import get_user_input
from util.tools import get_web_search_tool


# Tool: Calculate
@tool
def calculate(expression: str) -> str:
    """
    Räknar ut ett matematiskt uttryck.
    Exempel: '2 + 2 * 10'
    """
    try:
        result = eval(expression)
        return str(result)
    except Exception as e:
        return f"Fel vid beräkning: {e}"


# Tool: Read file
@tool
def read_file(file_path: str) -> str:
    """
    Läser innehållet i en textfil på datorn.
    Ange fullständig filväg.
    """
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            return f.read()
    except Exception as e:
        return f"Kunde inte läsa filen: {e}"


web_tools = get_web_search_tool()


def run():
    # Get predefined attributes
    model = get_model()

    # Lista med verktyg
    tools = [
        calculate,
        read_file,
        *web_tools
    ]

    # Create agent
    agent = create_agent(
        model=model,
        tools=tools,
        system_prompt=(
            "Du är en hjälpsam assistent som svarar på användarens frågor."
            "Om användaren ber om en matematisk beräkning ska du använda verktyget calculate."
            "Om användaren ber dig läsa en fil ska du använda verktyget read_file."
            "Om användaren ber dig läsa en webbsida ska du använda requests_get."
            "Svara alltid på svenska och var koncis men informativ."
        ),
    )

    # Memory for conversation
    chat_history = []

    while True:
        user_input = get_user_input("Du")

        if user_input.lower() in ["exit", "quit"]:
            print("Avslutar konversationen...")
            break

        # Lägg till användarens meddelande i minnet
        chat_history.append({"role": "user", "content": user_input})

        # Skicka hela historiken till agenten
        process_stream = agent.stream(
            {"messages": chat_history},
            stream_mode=STREAM_MODES,
        )

        # Visa svaret
        response = handle_stream(process_stream)

        # Lägg till agentens svar i minnet
        chat_history.append({"role": "assistant", "content": response})


if __name__ == "__main__":
    run()