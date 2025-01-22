from typing import Literal


def redirect_condition(state) -> Literal["generate", "chatbot"]:
    """
    Determines which node to go next

    Args:
        state (messages): The current state

    Returns:
        str: decision
    """

    message = state["messages"][-1]

    if message.name == "get_weather":
        print("----redirected to chatbot------")
        return "chatbot"
    else:
        print("----redirected to generate------")
        return "generate"
