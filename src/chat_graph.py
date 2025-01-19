from langgraph.graph import StateGraph, START, END, MessagesState
from langgraph.checkpoint.memory import MemorySaver
from langchain_community.tools.tavily_search import TavilySearchResults
from langgraph.prebuilt import ToolNode, tools_condition

from agents.chatbot import ChatBot


class ChatGraph:
    def __init__(self):
        self.chatbot = ChatBot("llama-3.2")
        self.build_graph()

    def build_graph(self) -> None:
        memory = MemorySaver()
        graph_builder = StateGraph(MessagesState)

        search_tool = TavilySearchResults(max_results=2)
        tools = [search_tool]
        tool_node = ToolNode(tools=tools)
        self.chatbot.bind_tools(tools)

        graph_builder.add_node("chatbot", self.chatbot)
        graph_builder.add_node("tools", tool_node)
        
        graph_builder.add_edge(START, "chatbot")
        graph_builder.add_conditional_edges("chatbot", tools_condition)
        graph_builder.add_edge("tools", "chatbot")
        graph_builder.add_edge("chatbot", END)

        self.chat_graph = graph_builder.compile(checkpointer=memory)

    def __call__(self):
        return self.chat_graph
