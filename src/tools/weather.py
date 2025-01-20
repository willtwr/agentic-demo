from langchain_core.tools import tool


@tool
def get_weather(location: str) -> str:
    """Call to get the current weather"""
    if location.lower() in ["kl", "kuala lumpur"]:
        return "It is 30 degree."
    elif location.lower() in ["sj", "subang jaya"]:
        return "It is 27 degree."
    else:
        return "Data not available."
