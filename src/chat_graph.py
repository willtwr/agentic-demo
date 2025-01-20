from langgraph.graph import StateGraph, MessagesState
from langgraph.checkpoint.memory import MemorySaver
from langgraph.prebuilt import ToolNode, tools_condition
# from langchain_community.tools.tavily_search import TavilySearchResults

from agents.chatbot.chatbot import ChatBot
from tools.weather import get_weather
# from tools.math import add, multiply


class ChatGraph:
    def __init__(self):
        self.chatbot = ChatBot("gemini")
        self.build_graph()

    def build_graph(self) -> None:
        memory = MemorySaver()
        graph_builder = StateGraph(MessagesState)

        # search_tool = TavilySearchResults(max_results=2)
        # tools = [get_weather, search_tool]

        tools = [get_weather]
        tool_node = ToolNode(tools=tools)
        self.chatbot.bind_tools(tools)

        graph_builder.add_node("chatbot", self.chatbot)
        graph_builder.add_node("tools", tool_node)
        
        graph_builder.set_entry_point("chatbot")
        graph_builder.add_conditional_edges("chatbot", tools_condition)
        graph_builder.add_edge("tools", "chatbot")

        self.chat_graph = graph_builder.compile(checkpointer=memory)

    def __call__(self):
        return self.chat_graph
